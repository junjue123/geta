"""蒸馏输入体检: 不训练, 只检查"喂给三个机制的东西"是否合理。

回答四个问题:
  A 教师输入   —— 候选快照本身可靠吗(精度、与学生的距离、预测一致率)
  B 视图输入   —— 决策层看到的 G 个视图, 对这个模型到底是不是"不同的输入"
  C 特征输入   —— 表征层抽的 tap 是否可分辨、是否互相重复
  D 参数输入   —— 投影的 ε 预算相对"单步更新幅度"是紧还是松, 层间差多少倍

用法: python scripts/probe_inputs.py <student_ckpt> [teacher_ckpt ...]
产物: outputs/probe_inputs/inputs.json
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train as T                                      # noqa: E402
from only_train_once.distillation.runtime import _TapCapture, _bn_eval   # noqa: E402
from only_train_once.distillation.views import make_views, SUPPORTED      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
TAPS = ["layer1", "layer2", "layer3", "avg_pool"]


def load(path):
    m, _ = T.build_model("resnet20", DEV)
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck.get("epoch")


@torch.no_grad()
def gather(loader, n_batch=4):
    xs, ys = [], []
    for i, (X, y) in enumerate(loader):
        if i >= n_batch:
            break
        xs.append(X)
        ys.append(y)
    return torch.cat(xs).to(DEV), torch.cat(ys).to(DEV)


@torch.no_grad()
def val_acc(model, loader):
    c = n = 0
    for X, y in loader:
        X, y = X.to(DEV), y.to(DEV)
        c += (model(X).argmax(1) == y).sum().item()
        n += y.numel()
    return c / n


# ---------------------------------------------------------------- A 教师输入
@torch.no_grad()
def check_teacher(student, teachers, X, testloader):
    out = []
    for path, (t, ep) in teachers.items():
        acc_s = student(X).argmax(1)
        acc_t = t(X).argmax(1)
        agree = (acc_s == acc_t).float().mean().item()
        kl = F.kl_div(F.log_softmax(student(X), 1), F.log_softmax(t(X), 1),
                      log_target=True, reduction="batchmean").item()
        # 教师自己的精度: 准入条件是 acc(T) >= acc(theta_t) - delta, 低于学生
        # 的教师被当成"可靠教师"就是输入不合理。
        t_acc = val_acc(t, testloader)
        # 参数距离: 只看存活权重, 与投影的判据一致
        d2 = 0.0
        for (n1, p1), (n2, p2) in zip(student.named_parameters(),
                                      t.named_parameters()):
            if n1 == n2 and p1.numel() == p2.numel():
                d2 += float(((p1 - p2) ** 2).sum())
        rel = d2 ** 0.5 / (sum(float(p.pow(2).sum()) for p in student.parameters()
                               if p.numel()) ** 0.5 + 1e-12)
        out.append({"ckpt": os.path.basename(path), "epoch": ep,
                    "val_acc": t_acc,
                    "pred_agree": agree, "kl_s_t": kl, "rel_l2_to_student": rel})
    return out


# ---------------------------------------------------------------- B 视图输入
@torch.no_grad()
def check_views(model, X, y, mode="mid", G=8):
    """视图是否真的构成"不同输入": 看像素差、预测变化率、组内退化率。"""
    res = []
    for sev in ("light", "mid", "heavy"):
        views = make_views(X, ["identity", "crop", "flip", "color", "cutout", "rotate"],
                           sev, group_size=G, seed=0)
        base_logits = model(X)
        base_pred = base_logits.argmax(1)
        per_mode = {}
        for name, v in views:
            if v.shape[0] != X.shape[0]:
                continue
            pix = float((v - X).abs().mean())
            pred = model(v).argmax(1)
            per_mode[name] = {
                "pixel_l1": pix,
                "pred_change": float((pred != base_pred).float().mean()),
                "acc": float((pred == y).float().mean()),
            }
        # 组内退化: 每条样本的 G 个视图里"全对"或"全错"的比例
        cor = torch.stack([(model(v).argmax(1) == y).float() for _, v in views], 1)
        degener = float(((cor.sum(1) == 0) | (cor.sum(1) == cor.size(1))).float().mean())
        res.append({"severity": sev, "G": len(views), "per_mode": per_mode,
                    "D_group": degener,
                    "indep_bound": float(
                        (base_logits.softmax(1).max(1).values ** len(views) +
                         (1 - base_logits.softmax(1).max(1).values) ** len(views)).mean())})
    return res


# ---------------------------------------------------------------- C 特征输入
@torch.no_grad()
def check_feats(model, X):
    taps = _TapCapture(model, TAPS)
    with _bn_eval(model), taps:
        model(X)
    store = dict(taps.store)
    res, kept = {}, []
    for n in TAPS:
        if n not in store:
            continue
        f = store[n]
        dist = f.mean(dim=(2, 3)) if f.dim() == 4 else f
        p = F.softmax(dist, 1)
        # 熵: 越接近 log C 说明逐通道分布被抹平, KL 拿不到信息
        ent = float(-(p * torch.log(p + 1e-8)).sum(1).mean())
        dup = None
        for m, q in kept:
            if q.shape == p.shape and torch.allclose(q, p, atol=1e-6, rtol=1e-4):
                dup = m
        if dup is None:
            kept.append((n, p))
        res[n] = {"shape": list(f.shape), "entropy": ent,
                  "log_C": float(torch.log(torch.tensor(float(p.size(1))) )),
                  "dup_of": dup}
    return res


# ---------------------------------------------------------------- D 参数输入
def check_weight(model, eps=0.01, lr=1e-3):
    rows = []
    for n, p in model.named_parameters():
        if p.numel() == 0:
            continue
        a = p.detach()
        an = float(a.norm())
        if an == 0.0:
            continue
        # Adam 每个元素一步约走 lr, 所以单步更新范数 ≈ lr*sqrt(元素数)
        step = lr * (p.numel() ** 0.5)
        rows.append({"param": n, "norm": an, "budget_eps": eps * an,
                     "adam_step": step, "ratio": (eps * an) / (step + 1e-12)})
    # 只打印被投影的"层"级(张量近 200 个), 汇总成统计量
    ratios = [r["ratio"] for r in rows]
    norms = [r["norm"] for r in rows]
    return {"n_tensors": len(rows),
            "norm_min": min(norms), "norm_max": max(norms),
            "norm_spread": max(norms) / (min(norms) + 1e-12),
            "ratio_min": min(ratios), "ratio_max": max(ratios),
            "ratio_median": sorted(ratios)[len(ratios) // 2],
            "tight_tensors": sum(1 for r in ratios if r < 1.0),
            "example": rows[:4] + rows[-2:]}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    student_path, teacher_paths = sys.argv[1], sys.argv[2:]
    trainloader, testloader = T.get_dataloaders(
        os.path.join(ROOT, "datasets/cifar10"), 128, 2, 1, False, True,
        MEAN, STD, 4, True, 256)
    X, y = gather(testloader)
    student, s_ep = load(student_path)
    report = {"student": {"ckpt": os.path.basename(student_path), "epoch": s_ep,
                          "val_acc": val_acc(student, testloader)}}

    if teacher_paths:
        teachers = {p: load(p) for p in teacher_paths}
        report["teacher"] = check_teacher(student, teachers, X, testloader)
        print("\nA 教师输入")
        print(f"  学生 {report['student']['ckpt']} val_acc={report['student']['val_acc']*100:.2f}%")
        for r in report["teacher"]:
            flag = "  <-- 比学生差, 不可靠" if r["val_acc"] < report["student"]["val_acc"] else ""
            print(f"  {r['ckpt']:32s} ep={r['epoch']:<3} val_acc={r['val_acc']*100:5.2f}%  "
                  f"预测一致={r['pred_agree']*100:5.1f}%  KL(S||T)={r['kl_s_t']:.3f}  "
                  f"相对参数距离={r['rel_l2_to_student']:.3f}{flag}")

    report["views"] = check_views(student, X, y)
    print("\nB 视图输入(学生模型上, 组内 G=8)")
    for r in report["views"]:
        print(f"  severity={r['severity']:<6} D_group={r['D_group']:.3f} "
              f"(独立理论上界 {r['indep_bound']:.3f})")
        for k, v in r["per_mode"].items():
            print(f"    {k:9s} 像素L1={v['pixel_l1']:.3f} "
                  f"预测变化={v['pred_change']*100:5.1f}% acc={v['acc']:.3f}")

    report["feats"] = check_feats(student, X)
    print("\nC 特征输入")
    for k, v in report["feats"].items():
        print(f"  {k:9s} shape={v['shape']} 熵={v['entropy']:.3f} "
              f"(log C={v['log_C']:.3f}) 与谁重复={v['dup_of']}")

    report["weight"] = check_weight(student)
    w = report["weight"]
    print("\nD 参数输入(ε=0.01 的预算 vs Adam 单步)")
    print(f"  {w['n_tensors']} 个张量, ||a|| 范围 {w['norm_min']:.4f}~{w['norm_max']:.4f} "
          f"(层间差 {w['norm_spread']:.1f}x)")
    print(f"  预算/单步 比例: 中位 {w['ratio_median']:.3f}, "
          f"最小 {w['ratio_min']:.3f}, 最大 {w['ratio_max']:.3f}; "
          f"预算小于单步的张量 {w['tight_tensors']}/{w['n_tensors']}")

    out = os.path.join(ROOT, "outputs/probe_inputs")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "inputs.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {out}/inputs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
