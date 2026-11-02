#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨框架对比可视化: 原始 GETA vs 我们的框架(DACO)
================================================
读取两个独立实验目录的输出:
  origin_dir/  <- run_daco_vs_baseline.py --root <microsoft/geta> --modes baseline
  ours_dir/    <- run_daco_vs_baseline.py --modes daco

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

C_ORIGIN = "#9e9e9e"
C_OURS = "#1976d2"
LABEL_ORIGIN = "Original GETA"
LABEL_OURS = "Ours (DACO)"


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
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    warmup = o.get("warmup_epochs", 15)

    specs = [
        ("val_acc", "Top-1 Accuracy", "(a)"),
        ("prune_ratio", "Sparsity (pruned ratio)", "(b)"),
        ("overall_compress", "Overall Compression (x)", "(c)"),
    ]
    for ax, (key, ylabel, tag) in zip(axes, specs):
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

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"[FIG] {path}")


# ===========================================================================
# 2. t-SNE 类别原型迁移轨迹
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


def plot_tsne_trajectories(o_protos, u_protos, out_dir, warmup,
                           perplexity=30):
    To, Tu = o_protos.shape[0], u_protos.shape[0]
    o_protos, u_protos = _align_dim(o_protos, u_protos)

    o_flat = o_protos.reshape(To * 10, -1)
    u_flat = u_protos.reshape(Tu * 10, -1)
    all_flat = np.concatenate([o_flat, u_flat], axis=0)
    if all_flat.shape[1] > 50:
        all_flat = PCA(n_components=50, random_state=42).fit_transform(all_flat)

    perp = min(perplexity, max(5, all_flat.shape[0] // 10))
    print(f"  [t-SNE] {all_flat.shape[0]} 点, perplexity={perp} ...")
    emb = TSNE(n_components=2, perplexity=perp, random_state=42,
               init="pca", learning_rate="auto", n_iter=1500).fit_transform(all_flat)
    o2d = emb[:To * 10].reshape(To, 10, 2)
    u2d = emb[To * 10:].reshape(Tu, 10, 2)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, p2d, T, title in [
        (axes[0], o2d, To, f"{LABEL_ORIGIN} (hard projection)"),
        (axes[1], u2d, Tu, f"{LABEL_OURS} (RCAJS+MCSS+DGD)"),
    ]:
        for c in range(10):
            traj = p2d[:, c, :]
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[c], linewidth=0.9,
                    alpha=0.7, zorder=1)
            s = max(warmup, 0)
            if s < T:
                ax.scatter(traj[s, 0], traj[s, 1], color=COLORS[c], s=30,
                           marker="o", edgecolors="black", linewidths=0.4,
                           zorder=3)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS[c], s=35,
                       marker="^", edgecolors="black", linewidths=0.4,
                       zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title, fontsize=10)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

    handles = [Line2D([0], [0], color=COLORS[c], lw=1.5, marker="o",
                      markersize=5, markeredgecolor="black",
                      markeredgewidth=0.3, label=CIFAR10_CLASSES[c])
               for c in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out_dir, "fig_tsne_trajectories.png")
    plt.savefig(path)
    plt.close()
    print(f"[FIG] {path}")
    return o2d, u2d


# ===========================================================================
# 3. 特征漂移幅度: 每个 epoch 内类别原型的平均位移
#    轨迹越"抖", 曲线越高; DACO 期望更平滑
# ===========================================================================
def plot_drift(o2d, u2d, out_dir, warmup):
    def step_disp(a):
        # (T,10,2) -> (T-1,) 相邻 epoch 的类别原型平均欧氏位移
        d = np.linalg.norm(np.diff(a, axis=0), axis=2)  # (T-1,10)
        return d.mean(axis=1)

    do, du = step_disp(o2d), step_disp(u2d)
    ep = np.arange(2, len(do) + 2)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ep, do, color=C_ORIGIN, linewidth=1.1, label=LABEL_ORIGIN, alpha=0.95)
    ax.plot(ep, du, color=C_OURS, linewidth=1.1, label=LABEL_OURS, alpha=0.95)
    if warmup > 0:
        ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Prototype displacement (t-SNE space)")
    ax.legend(loc="best", fontsize=8)

    # 联合阶段(log10 尺度)的平均漂移, 直接量化"平滑度"
    j = warmup
    if j < len(do):
        mo, mu = do[j:].mean(), du[j:].mean()
        ax.text(0.98, 0.95,
                f"joint-phase mean: {LABEL_ORIGIN}={mo:.3f}, {LABEL_OURS}={mu:.3f}",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="#424242")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_drift_metric.png")
    plt.savefig(path)
    plt.close()
    print(f"[FIG] {path}")


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
        o2d, u2d = plot_tsne_trajectories(op, up, args.out, warmup,
                                          args.tsne_perplexity)
        plot_drift(o2d, u2d, args.out, warmup)
    except FileNotFoundError as e:
        print(f"[WARN] 原型缺失, 跳过 t-SNE: {e}")

    print(f"\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
