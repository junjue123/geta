#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨框架对比可视化: 原始 GETA vs 我们的框架(DACO)
================================================
读取两个独立实验目录的输出:
  origin_dir/  <- train.py --root <microsoft/geta> --modes baseline
  ours_dir/    <- train.py --modes daco

生成:
  1. fig_curves.png          (a) Top-1 准确率  (b) 稀疏度/剪枝率  (c) 整体压缩率
  2. fig_tsne_trajectories.png  逐步 t-SNE 类别原型迁移轨迹 (特征漂移)
  3. fig_drift_metric.png    每 epoch 的特征漂移幅度 (轨迹位移范数), 量化"平滑 vs 震荡"

科研规范格式: 无标题, 保留刻度线, 规范图例线条

用法:
  python plot_framework_compare.py \
      --origin-dir outputs/framework_compare/origin \
      --ours-dir  outputs/framework_compare/ours \
      --out       outputs/framework_compare
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ---- 科研图样式 ----
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 10
rcParams["axes.linewidth"] = 0.8
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["xtick.major.size"] = 3
rcParams["ytick.major.size"] = 3
rcParams["legend.frameon"] = False
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# Okabe-Ito 色盲安全配色 (期刊通用, 灰阶打印可辨)
C_ORIGIN = "#D55E00"   # baseline: 朱红
C_OURS = "#0072B2"     # DACO: 蓝
C_ORIGIN_L = C_ORIGIN
C_OURS_L = C_OURS
LABEL_ORIGIN = "baseline"
LABEL_OURS = "DACO"
DPI = 200   # 出图用; 论文排版按 200 dpi 导出即可


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}, 请先跑对应实验")
    with open(path) as f:
        return json.load(f)


def load_pair(origin_dir, ours_dir, mode_origin="baseline", mode_ours="daco"):
    o = _load_json(os.path.join(origin_dir, f"{mode_origin}.json"))
    u = _load_json(os.path.join(ours_dir, f"{mode_ours}.json"))
    return o, u


def load_protos(d, mode, model):
    p = os.path.join(d, "prototypes", f"{model}_{mode}_protos.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)  # (T, 10, D)


# ===========================================================================
# 1. 准确率 / 稀疏度 / 压缩率
# ===========================================================================
def plot_curves(o, u, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.2))
    warmup = o.get("warmup_epochs", 15)

    # 压缩率: 优先用论文 BOPs 口径 cr_bops = 32*32/((1-s)*w*a);
    # 老的 overall_compress 是 (1-s)*16/w, 基准写成 16 且漏掉激活位宽, 不可用。
    has_cr = ("cr_bops" in o["history"]) and ("cr_bops" in u["history"])
    c_key = "cr_bops" if has_cr else "overall_compress"
    c_label = "Compression Rate (BOPs, x)" if has_cr else "Overall Compression (x)"

    specs = [
        ("val_acc", "Top-1 Accuracy", "(a)"),
        ("prune_ratio", "Sparsity (pruned ratio)", "(b)"),
        (c_key, c_label, "(c)"),
    ]
    for ax, (key, ylabel, tag) in zip(axes[:3], specs):
        ep_o = np.array(o["history"]["epoch"])
        ep_u = np.array(u["history"]["epoch"])
        ax.plot(ep_o, o["history"][key], color=C_ORIGIN, linewidth=1.2,
                label=LABEL_ORIGIN, alpha=0.95)
        ax.plot(ep_u, u["history"][key], color=C_OURS, linewidth=1.2,
                label=LABEL_OURS, alpha=0.95)
        if warmup > 0:
            ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.6,
                       alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.text(0.02, 0.95, tag, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")

    # (d) 位宽双线: 权重实线 / 激活虚线, baseline 与 DACO 各一组
    ax = axes[3]
    ep_o = np.array(o["history"]["epoch"])
    ep_u = np.array(u["history"]["epoch"])
    ax.plot(ep_o, o["history"]["avg_weight_bit"], color=C_ORIGIN, linewidth=1.2,
            linestyle="-", label=f"{LABEL_ORIGIN} W", alpha=0.95)
    ax.plot(ep_o, o["history"]["avg_act_bit"], color=C_ORIGIN, linewidth=1.2,
            linestyle="--", label=f"{LABEL_ORIGIN} A", alpha=0.95)
    ax.plot(ep_u, u["history"]["avg_weight_bit"], color=C_OURS, linewidth=1.2,
            linestyle="-", label=f"{LABEL_OURS} W", alpha=0.95)
    ax.plot(ep_u, u["history"]["avg_act_bit"], color=C_OURS, linewidth=1.2,
            linestyle="--", label=f"{LABEL_OURS} A", alpha=0.95)
    if warmup > 0:
        ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average bit-width")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.text(0.02, 0.95, "(d)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_curves.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")


# ===========================================================================
# 2. 特征空间漂移: 联合 t-SNE 出坐标, 每个 epoch 取「该 epoch 全部类坐标的
#    均值」作为这一 epoch 在特征空间中的代表点, 代表点的逐 epoch 位移即漂移。
# ===========================================================================
def _align_dim(a, b):
    D = min(a.shape[2], b.shape[2])
    out = []
    for arr in (a, b):
        if arr.shape[2] > D:
            T, C, Dd = arr.shape
            flat = PCA(n_components=D, random_state=42).fit_transform(
                arr.reshape(T * C, Dd))
            out.append(flat.reshape(T, C, D))
        else:
            out.append(arr)
    return out[0], out[1]


def _tsne_center_drift(o_protos, u_protos, perplexity=30):
    """返回 (oc, uc, do, du): 两边的 epoch 代表点轨迹与逐 epoch 位移。"""
    o_protos, u_protos = _align_dim(o_protos, u_protos)
    To, C = o_protos.shape[0], o_protos.shape[1]
    Tu = u_protos.shape[0]

    o_flat = o_protos.reshape(To * C, -1)
    u_flat = u_protos.reshape(Tu * C, -1)
    all_flat = np.concatenate([o_flat, u_flat], axis=0)
    if all_flat.shape[1] > 50:
        all_flat = PCA(n_components=50, random_state=42).fit_transform(all_flat)

    perp = min(perplexity, max(5, all_flat.shape[0] // 10))
    print(f"  [t-SNE] {all_flat.shape[0]} 点, perplexity={perp} ...")
    emb = TSNE(n_components=2, perplexity=perp, random_state=42,
               init="pca", learning_rate="auto", n_iter=1500).fit_transform(all_flat)
    o2d = emb[:To * C].reshape(To, C, 2)
    u2d = emb[To * C:].reshape(Tu, C, 2)

    # 每个 epoch 取该 epoch 全部类坐标的均值 -> (T, 2) 代表点
    oc = o2d.mean(axis=1)
    uc = u2d.mean(axis=1)
    do = np.linalg.norm(np.diff(oc, axis=0), axis=1)
    du = np.linalg.norm(np.diff(uc, axis=0), axis=1)
    return oc, uc, do, du


def plot_tsne_center_drift(o_protos, u_protos, out_dir, warmup, perplexity=30):
    oc, uc, do, du = _tsne_center_drift(o_protos, u_protos, perplexity)

    ep_o = np.arange(2, len(do) + 2)
    ep_u = np.arange(2, len(du) + 2)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))

    # (a) 代表点在 t-SNE 平面上的移动轨迹
    ax = axes[0]
    ax.plot(oc[:, 0], oc[:, 1], color=C_ORIGIN, linewidth=1.1,
            label=LABEL_ORIGIN, alpha=0.9)
    ax.plot(uc[:, 0], uc[:, 1], color=C_OURS, linewidth=1.1,
            label=LABEL_OURS, alpha=0.9)
    for cc, col in ((oc, C_ORIGIN), (uc, C_OURS)):
        ax.scatter(cc[0, 0], cc[0, 1], color=col, s=34, marker="o",
                   edgecolors="black", linewidths=0.4, zorder=3)
        ax.scatter(cc[-1, 0], cc[-1, 1], color=col, s=38, marker="^",
                   edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

    # (b) 代表点的逐 epoch 位移
    ax = axes[1]
    ax.plot(ep_o, do, color=C_ORIGIN, linewidth=1.1, label=LABEL_ORIGIN, alpha=0.95)
    ax.plot(ep_u, du, color=C_OURS, linewidth=1.1, label=LABEL_OURS, alpha=0.95)
    if warmup > 0:
        ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Displacement of epoch mean (t-SNE)")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

    j_o = warmup if warmup < len(do) else 0
    j_u = warmup if warmup < len(du) else 0
    ax.text(0.98, 0.80,
            f"joint mean: {LABEL_ORIGIN}={do[j_o:].mean():.3f}, "
            f"{LABEL_OURS}={du[j_u:].mean():.3f}",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color="#424242")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_feature_drift.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")

    print(f"  [DRIFT] {LABEL_ORIGIN}: mean={do.mean():.3f} max={do.max():.3f} "
          f"@ep{ep_o[int(do.argmax())]} | total path={np.linalg.norm(np.diff(oc, axis=0), axis=1).sum():.2f}")
    print(f"  [DRIFT] {LABEL_OURS}: mean={du.mean():.3f} max={du.max():.3f} "
          f"@ep{ep_u[int(du.argmax())]} | total path={np.linalg.norm(np.diff(uc, axis=0), axis=1).sum():.2f}")
    return do, du


def plot_combined(o, u, o_protos, u_protos, out_dir, warmup, perplexity=30):
    """总图: 左 = 准确率(左轴) + 压缩率(右轴); 右 = t-SNE 表示的特征漂移。"""
    oc, uc, do, du = _tsne_center_drift(o_protos, u_protos, perplexity)
    ep_o = np.array(o["history"]["epoch"])
    ep_u = np.array(u["history"]["epoch"])
    d_ep_o = np.arange(2, len(do) + 2)
    d_ep_u = np.arange(2, len(du) + 2)

    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.2),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.34})

    # ---------------- 左: 准确率 + 压缩率 ----------------
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(ep_o, o["history"]["val_acc"], color=C_ORIGIN, linewidth=1.6)
    ax.plot(ep_u, u["history"]["val_acc"], color=C_OURS, linewidth=1.6)
    ax2.plot(ep_o, o["history"]["cr_bops"], color=C_ORIGIN, linewidth=1.1,
             linestyle=(0, (4, 2.5)))
    ax2.plot(ep_u, u["history"]["cr_bops"], color=C_OURS, linewidth=1.1,
             linestyle=(0, (4, 2.5)))
    if warmup > 0:
        ax.axvline(warmup, color="#8a8a8a", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 accuracy")
    ax2.set_ylabel("Compression rate (BOPs, ×)")
    ax.set_ylim(0.15, 0.70)
    ax2.set_ylim(8, 42)
    ax.text(0.012, 0.985, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

    # ---------------- 右: t-SNE 特征漂移 ----------------
    ax = axes[1]
    ax.plot(d_ep_o, do, color=C_ORIGIN, linewidth=1.2)
    ax.plot(d_ep_u, du, color=C_OURS, linewidth=1.2)
    if warmup > 0:
        ax.axvline(warmup, color="#8a8a8a", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Drift of epoch-mean (t-SNE)")
    ax.set_ylim(-4, do.max() * 1.32)
    ax.text(0.012, 0.985, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")
    for tag, d in ((LABEL_ORIGIN, do), (LABEL_OURS, du)):
        i = int(d.argmax())
        ax.annotate(f"{d.max():.1f} @ep{i + 2}", xy=(i + 2, d.max()),
                    xytext=(i + 2, d.max() + d.max() * 0.06), fontsize=6.8,
                    color="#4a4a4a", ha="center")

    # 左上角内嵌: epoch 代表点在 t-SNE 平面上的路径
    # 抽稀到 ~14 个点并让标记沿时间变大, 否则 100 个点挤在这么小的框里看不清走向
    axi = ax.inset_axes([0.045, 0.52, 0.40, 0.42])
    for c, col in ((oc, C_ORIGIN), (uc, C_OURS)):
        step = max(1, len(c) // 14)
        idx = list(range(0, len(c), step))
        if idx[-1] != len(c) - 1:
            idx.append(len(c) - 1)
        p = c[idx]
        axi.plot(p[:, 0], p[:, 1], color=col, linewidth=1.0, alpha=0.9)
        sizes = np.linspace(5, 30, len(p))
        axi.scatter(p[:, 0], p[:, 1], s=sizes, color=col, edgecolors="white",
                    linewidths=0.5, zorder=3)
        axi.scatter(p[0, 0], p[0, 1], s=26, facecolors="none", edgecolors=col,
                    linewidths=1.0, zorder=4)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.tick_params(length=0)
    for sp in axi.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#9a9a9a")
    axi.set_title("prototype drift path", fontsize=6.5, pad=2.5, color="#4a4a4a")

    # 图底统一图例: 颜色=方法, 线型=指标
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C_ORIGIN, lw=1.6, label=LABEL_ORIGIN),
        Line2D([0], [0], color=C_OURS, lw=1.6, label=LABEL_OURS),
        Line2D([0], [0], color="#4a4a4a", lw=1.1, ls=(0, (4, 2.5)),
               label="dashed: compression (right axis)"),
        Line2D([0], [0], color="#4a4a4a", lw=0.8, ls=":", label="warmup end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.045),
               handlelength=2.0, columnspacing=1.6)

    plt.savefig(os.path.join(out_dir, "fig_combined.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {os.path.join(out_dir, 'fig_combined.png')}")
    return do, du


def main():
    p = argparse.ArgumentParser(description="原始 GETA vs 我们框架 可视化")
    p.add_argument("--origin-dir", default="outputs/framework_compare/origin")
    p.add_argument("--ours-dir", default="outputs/framework_compare/ours")
    p.add_argument("--out", default="outputs/framework_compare")
    p.add_argument("--mode-origin", default="baseline")
    p.add_argument("--mode-ours", default="daco")
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    o, u = load_pair(args.origin_dir, args.ours_dir,
                     args.mode_origin, args.mode_ours)
    model = o.get("model", "resnet20")
    warmup = o.get("warmup_epochs", 15)

    print(f"{LABEL_ORIGIN}: best_acc={o['best_val_acc']} "
          f"final_acc={o['final_val_acc']} "
          f"sparsity={o['final_prune_ratio']} "
          f"compress={o['final_overall_compress']}x")
    print(f"{LABEL_OURS}:   best_acc={u['best_val_acc']} "
          f"final_acc={u['final_val_acc']} "
          f"sparsity={u['final_prune_ratio']} "
          f"compress={u['final_overall_compress']}x")

    plot_curves(o, u, args.out)

    try:
        op = load_protos(args.origin_dir, args.mode_origin, model)
        up = load_protos(args.ours_dir, args.mode_ours, model)
        print(f"  [PROTO] origin {op.shape}, ours {up.shape}")
        plot_combined(o, u, op, up, args.out, warmup, args.tsne_perplexity)
    except FileNotFoundError as e:
        print(f"[WARN] 原型缺失, 跳过特征漂移图: {e}")

    print(f"\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
