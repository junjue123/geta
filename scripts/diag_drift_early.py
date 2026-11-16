"""临时诊断: 对比 DACO 与原版在 joint 早期/后期的 val_acc 抖动与原型漂移。

只读输出目录, 不写盘除 stdout。
用法: python diag_drift_early.py
"""
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "outputs", "framework_compare")
RUNS = {"daco": ("ours", "daco_history.jsonl", "resnet20_daco_protos.npy"),
        "origin": ("origin", "baseline_history.jsonl", "resnet20_baseline_protos.npy")}


def load(hist, proto):
    recs = [json.loads(l) for l in open(hist, encoding="utf-8") if l.strip()]
    p = np.load(proto)
    return recs, p


def jitter(acc, lo, hi):
    """窗口 [lo,hi] 内: 平均 |Δacc|, 以及相对 3 点滑动均值的残差(std)。"""
    a = np.asarray(acc[lo:hi], dtype=float)
    if len(a) < 3:
        return None
    d = np.abs(np.diff(a)).mean()
    res = a[1:-1] - (a[:-2] + a[1:-1] + a[2:]) / 3.0
    return d, res.std(), a.min(), a.max()


def drift(p):
    """相邻 epoch 类别原型位移: 每类 L2 平均, 再除以该 epoch 原型平均范数归一化。"""
    out = []
    for t in range(1, len(p)):
        d = np.linalg.norm(p[t] - p[t - 1], axis=1).mean()
        n = np.linalg.norm(p[t], axis=1).mean() + 1e-8
        out.append(d / n)
    return np.array(out)  # index i 对应 epoch (i+2) 相对 (i+1)


for tag, (sub, hf, pf) in RUNS.items():
    recs, p = load(os.path.join(ROOT, sub, hf), os.path.join(ROOT, sub, "prototypes", pf))
    acc = [r["val_acc"] for r in recs]
    sp = [r["group_sparsity"] for r in recs]
    print(f"\n===== {tag}  ({len(recs)} epochs, protos {p.shape}) =====")

    # 1) 剪枝绝对增量 vs 相对增量
    print("  sparsity 绝对增量/相对增量:")
    for e in (17, 21, 25, 29):
        if e < len(sp):
            di = sp[e] - sp[e - 1]
            print(f"    ep{e:3d}: spars={sp[e]:.4f}  +{di:.4f}  (相对 +{100*di/max(sp[e-1],1e-9):.1f}%)")

    # 2) val_acc 抖动分段
    for lo, hi, name in ((15, 25, "joint 早期 ep16-25"), (25, len(acc), "joint 后期 ep26+")):
        r = jitter(acc, lo, hi)
        if r:
            print(f"  {name}: 平均|Δacc|={r[0]*100:.2f}pp  去趋势抖动std={r[1]*100:.2f}pp  "
                  f"范围 {r[2]:.4f}~{r[3]:.4f}")

    # 3) 原型漂移分段（归一化）
    d = drift(p)
    for lo, hi, name in ((0, 14, "warmup"), (14, 24, "joint 早期"), (24, len(d), "joint 后期")):
        seg = d[lo:min(hi, len(d))]
        if len(seg):
            print(f"  proto漂移 {name}: n={len(seg)} mean={seg.mean()*100:.2f}%  "
                  f"max={seg.max()*100:.2f}%  std={seg.std()*100:.2f}%")

# 4) 重叠区间严格同 epoch 对比
print("\n===== 同 epoch 对比 (ep16 起) =====")
D = {}
for tag, (sub, hf, pf) in RUNS.items():
    recs, p = load(os.path.join(ROOT, sub, hf), os.path.join(ROOT, sub, "prototypes", pf))
    D[tag] = (recs, p)
n = min(len(v[0]) for v in D.values())
print(f"{'ep':>4} {'daco_acc':>9} {'orig_acc':>9} {'daco_bit':>9} {'orig_bit':>9} {'daco_drift':>11} {'orig_drift':>11}")
dp = {k: drift(v[1]) for k, v in D.items()}
for e in range(16, n + 1):
    i = e - 1
    dd = dp["daco"][i - 1] * 100 if i - 1 < len(dp["daco"]) else float("nan")
    od = dp["origin"][i - 1] * 100 if i - 1 < len(dp["origin"]) else float("nan")
    print(f"{e:>4} {D['daco'][0][i]['val_acc']:>9.4f} {D['origin'][0][i]['val_acc']:>9.4f} "
          f"{D['daco'][0][i]['avg_weight_bit']:>9.2f} {D['origin'][0][i]['avg_weight_bit']:>9.2f} "
          f"{dd:>10.2f}% {od:>10.2f}%")
