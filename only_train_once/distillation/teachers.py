"""第3章: 教师池(T_hist / T_cf)。

两类教师的分工是固定的, 不要混用:
  T_hist —— 压缩轨迹上"还比较可靠"的历史快照(热身前段)。负责软标签、
            特征锚、ref 位。它可靠, 所以可以做"标准答案"。
  T_cf   —— 当前模型的冻结影子副本。它**不可靠**(和学生在同一压缩点上),
            所以只做逐视图的密集认可度奖励, 不做软标签、不做特征锚。

设计约束(踩过的坑):
  - 教师必须整份 eval + requires_grad=False, 否则 BN 的 running stats 会被
    学生的 batch 污染, 且反传会把梯度灌进教师参数。
  - checkpoint 可能是续训格式 {'model':...} 也可能是最终格式
    {'model_state_dict':...}, 两种都要能吃。加载后要核对 key 是否对得上,
    对不上直接报错而不是静默半加载。
"""

import copy
import glob
import os
import re

import torch


_CKPT_EP = re.compile(r"_ep(\d+)\.pt$")


def discover_hist_candidates(out_dir, model_name, mode,
                             prefix_mode=None, max_epoch=None):
    """扫描 {out}/checkpoints 下按 epoch 命名的快照, 返回 [(epoch, path), ...]。

    prefix_mode 默认跟当前 run 一致; 跨目录取教师时(例如用 warmup_100 里的
    baseline 快照当教师)需要显式给。
    """
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    pat = os.path.join(ckpt_dir, f"{model_name}_{prefix_mode or mode}_ep*.pt")
    out = []
    for p in glob.glob(pat):
        m = _CKPT_EP.search(os.path.basename(p))
        if m:
            ep = int(m.group(1))
            if max_epoch is not None and ep > max_epoch:
                continue
            out.append((ep, p))
    return sorted(out)


def _extract_state(obj):
    """兼容三种保存格式, 返回纯 model state dict。"""
    if isinstance(obj, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # 本身就是 state dict(键里带 "." 或以 weight/bias 结尾)
        if any("." in str(k) for k in obj.keys()):
            return obj
    raise ValueError("checkpoint 里找不到模型权重(model / model_state_dict)")


def load_teacher(model_factory, path, device):
    """用 model_factory 新建一份模型, 灌入 ckpt 权重, 冻结后返回。"""
    teacher = model_factory()
    teacher.to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = _extract_state(ckpt)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    # 量化模型的 state dict 里带 d_quant_wt/t_quant_wt 这类派生量, 缺几个不算错;
    # 但原生参数缺了就是加载错了, 必须报出来。
    fatal = [k for k in missing if "quant" not in k and "scale" not in k]
    if fatal:
        raise RuntimeError(f"{os.path.basename(path)} 加载失败, 缺关键参数: {fatal[:6]}")
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def quick_acc(model, loader, device, max_batches=None):
    """教师准入用的快速精度。只跑验证集, 不进梯度, 一次性开销。"""
    if loader is None:
        return None
    was_training = model.training
    model.eval()
    c = n = 0
    for i, (X, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        X, y = X.to(device), y.to(device)
        c += (model(X).argmax(1) == y).sum().item()
        n += y.numel()
    if was_training:
        model.train()
    return c / max(n, 1)


class TeacherPool:
    """按开关组装教师集合; 缺资源时降级并记录原因, 不静默。"""

    def __init__(self, cfg, model, out_dir, model_name, mode, device,
                 model_factory, log=None, val_loader=None):
        self.cfg = cfg
        self.device = device
        self.model = model      # 学生; 准入判据要拿它当参照(此前漏赋值, 准入一开就 AttributeError)
        self.hist = []          # [(epoch, model)]
        self.cf = None
        self.notes = []
        self._log = log or (lambda tag, msg: None)
        self.val_loader = val_loader

        t = cfg["teacher"]
        self._build_hist(t["hist"], out_dir, model_name, mode, model_factory)
        self._build_cf(t["cf"], model)

    # ---------------- T_hist ----------------
    def _build_hist(self, hcfg, out_dir, model_name, mode, model_factory):
        if not hcfg["enabled"]:
            return
        # 教师来源目录: 默认取本次 run 的 out_dir。跨目录(把"压缩前的起点模型"
        # 当教师)时由配置指定, 前缀里的 mode 也要跟着变。
        src_dir = hcfg.get("source_dir") or out_dir
        src_mode = hcfg.get("source_mode") or mode
        cands = discover_hist_candidates(src_dir, model_name, src_mode)
        if not cands:
            # 必须硬失败。没有教师时软标签/特征锚/ref 位会全部静默关闭, 训练照跑、
            # 日志一切正常, 等于三项机制里的两项没开 —— 这种"实验白跑"比报错贵得多。
            ckpt_dir = os.path.join(src_dir, "checkpoints")
            near = []
            if os.path.isdir(ckpt_dir):
                near = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))[:8]
            elif os.path.isdir(src_dir):
                near = sorted(os.listdir(src_dir))[:8]
            raise RuntimeError(
                f"T_hist 无候选: 找不到 {ckpt_dir}/{model_name}_{src_mode}_ep*.pt。"
                f" 该目录现有: {near or '目录不存在或为空'}。"
                f" 检查 source_dir / source_mode(跨目录取教师时要写那边的 mode, "
                f"例如 baseline) 与 ckpt 前缀是否对得上。")
        k = max(1, int(hcfg["max_count"]))
        if len(cands) < k:
            self._log("DISTILL",
                      f"T_hist 候选只有 {len(cands)} 个, 少于 max_count={k}, "
                      f"按实际 {len(cands)} 个取")
        picked = self._select(cands, k, hcfg["require_warmup_mid"],
                              hcfg.get("coordinate", "compression"),
                              hcfg.get("pick", "earliest"))
        for ep, path in picked:
            try:
                m = load_teacher(model_factory, path, self.device)
                self.hist.append((ep, m))
            except Exception as e:                      # 单个坏 ckpt 不该拖垮整次训练
                self._log("DISTILL", f"教师 ep{ep} 加载失败, 跳过: {e}")
        if not self.hist:
            raise RuntimeError(
                f"T_hist 候选 {len(cands)} 个但全部加载失败(见上面的 [DISTILL] 失败原因), "
                "教师池为空会让软标签/特征锚/ref 位静默关闭。")
        if self.hist:
            delta = float(hcfg.get("admit_acc_delta", 0.0))
            # 准入判据: 教师精度不得低于学生太多。实测学生 ep15 val 49.57%,
            # 而 ep5/ep10 只有 43.74%/47.78% —— 两个"可靠教师"其实都比学生弱
            # 5.8/1.8 pt, 拿它们的软标签与特征锚去拽学生, 方向是错的。
            if self.val_loader is not None and delta > 0:
                s_acc = quick_acc(self.model, self.val_loader, self.device)
                kept = []
                for ep, m in self.hist:
                    t_acc = quick_acc(m, self.val_loader, self.device)
                    if t_acc is not None and t_acc >= s_acc - delta:
                        kept.append((ep, m))
                        self._log("DISTILL",
                                  f"教师 ep{ep} 准入通过: acc {t_acc:.4f} "
                                  f"(学生 {s_acc:.4f}, 容差 {delta})")
                    else:
                        self._log("DISTILL",
                                  f"教师 ep{ep} 准入被拒: acc {t_acc:.4f} < "
                                  f"学生 {s_acc:.4f} - {delta}")
                self.hist = kept
            eps = [e for e, _ in self.hist]
            if self.hist:
                self.notes.append(f"T_hist 就绪: {len(self.hist)} 个(ep={eps})")
                self._log("DISTILL", self.notes[-1])
            else:
                # 同样必须硬失败: 准入把教师全拒了, 后续特征层/ref 位会静默关闭。
                raise RuntimeError(
                    "T_hist 全部被准入判据拒绝: 每个候选教师都劣于学生 "
                    f"(admit_acc_delta={delta})。要么把起点模型换成更强的快照, "
                    "要么放宽 admit_acc_delta, 不要带着空教师池继续跑。")

    @staticmethod
    def _select(cands, k, require_mid, coordinate="compression", pick="earliest"):
        """选教师快照。E4 要比较三种坐标口径, 所以这里把"怎么排序"和"从哪起选"
        分开: coordinate 决定间距口径, pick 决定锚点是整条轨迹的哪一端。
        """
        if coordinate == "epoch_even" and k > 1:
            # 按 epoch 等距取点(含首尾): 与"压缩更少优先"的口径做对照
            n = len(cands)
            idxs = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
            chosen = [cands[i] for i in idxs]
            if cands[-1] not in chosen:
                chosen.append(cands[-1])
            return chosen[:k]

        anchor = {"earliest": 0, "mid": len(cands) // 2,
                  "end": len(cands) - 1}.get(pick, 0)
        chosen = [cands[anchor]]
        if k >= 2 and require_mid and len(cands) > 1:
            mid = cands[len(cands) // 2]
            if mid[0] != chosen[0][0]:
                chosen.append(mid)
        for c in cands:
            if len(chosen) >= k:
                break
            if c[0] not in [x[0] for x in chosen]:
                chosen.append(c)
        return chosen[:k]

    # ---------------- T_cf ----------------
    def _build_cf(self, ccfg, model):
        """S1 影子副本: 结构、剪枝掩码、量化参数全都照抄当前模型。"""
        if not ccfg["enabled"]:
            return
        self.cf = copy.deepcopy(model)
        self.cf.requires_grad_(False)
        self.cf.eval()
        if ccfg["implement"] != "shadow":
            self.notes.append(f"teacher.cf.implement={ccfg['implement']} 尚未实现, "
                              "本次按 shadow 处理")
            self._log("DISTILL", self.notes[-1])

    # ---------------- 查询 ----------------
    @property
    def has_hist(self):
        return len(self.hist) > 0

    @property
    def has_cf(self):
        return self.cf is not None

    def primary_hist(self):
        """软标签/特征锚用第一个(最早 = 压缩最少 = 最可靠)。"""
        return self.hist[0][1] if self.hist else None

    def describe(self):
        return " ".join(self.notes) if self.notes else "教师池为空"
