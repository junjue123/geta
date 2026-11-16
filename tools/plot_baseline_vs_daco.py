#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline vs DACO 可视化 (对齐 TCSVT 论文 Fig.ourDGD)
=====================================================
读取 train.py 输出, 生成:
  1. 准确率 / 压缩率 折线图 (baseline vs DACO, 100 epoch 曲线)
     - 前 15 epoch warmup 两线重合, 之后分叉
     - DACO 压缩更快但准确率震荡更明显
  2. t-SNE prototype migration 轨迹图 (论文 Fig.ourDGD 样式)
     - 把所有 epoch 的 10 个类别原型(64维) 一起 t-SNE 到 2D
     - 每个类别的轨迹按时间顺序连线: baseline 跳跃震荡, DACO 平滑

科研规范格式: 无标题, 保留刻度线, 规范图例线条

用法:
  python plot_baseline_vs_daco.py --out outputs/baseline_vs_daco
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
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


def load_results(out_dir):
    b_path = os.path.join(out_dir, "baseline.json")
    d_path = os.path.join(out_dir, "daco.json")
    if not os.path.exists(b_path) or not os.path.exists(d_path):
        raise FileNotFoundError("缺少 baseline.json 或 daco.json, 请先跑实验")
    with open(b_path) as f:
        b = json.load(f)
    with open(d_path) as f:
        d = json.load(f)
    return b, d


def load_protos(out_dir, mode, model):
    p = os.path.join(out_dir, "prototypes", f"{model}_{mode}_protos.npy")
    return np.load(p)  # (T, 10, D)


# ===========================================================================
# 1. 准确率 / 压缩率 折线图
# ===========================================================================
def plot_curves(b, d, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # (a) Val Accuracy vs epoch
    ax = axes[0]
    ep = np.array(b["history"]["epoch"])
    warmup = b["warmup_epochs"]
    ax.plot(ep, b["history"]["val_acc"], color="#9e9e9e", linewidth=1.2,
            label="Baseline", alpha=0.9)
    ax.plot(ep, d["history"]["val_acc"], color="#1976d2", linewidth=1.2,
            label="DACO", alpha=0.9)
    if warmup > 0:
        ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.6,
                   alpha=0.6)
        ax.text(warmup + 1, ax.get_ylim()[0] + 0.01, "warmup end",
                fontsize=7, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

    # (b) Overall Compression vs epoch
    ax = axes[1]
    ax.plot(ep, b["history"]["overall_compress"], color="#9e9e9e",
            linewidth=1.2, label="Baseline", alpha=0.9)
    ax.plot(ep, d["history"]["overall_compress"], color="#1976d2",
            linewidth=1.2, label="DACO", alpha=0.9)
    if warmup > 0:
        ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.6,
                   alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Overall Compression (x)")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"[FIG] {path}")


# ===========================================================================
# 2. t-SNE prototype migration 轨迹图 (论文 Fig.ourDGD 样式)
# ===========================================================================
def plot_tsne_trajectories(b_protos, d_protos, out_dir, warmup, perplexity=30):
    """
    b_protos / d_protos: (T, 10, D)  T 个 epoch, 10 类, D 维特征
    做法: 把 baseline 和 daco 的所有原型拼成 (2*T*10, D) 一起 t-SNE 到 2D,
          保证两图坐标系一致, 然后每个类别按时间顺序连线成轨迹。
    warmup: 热身 epoch 数, 轨迹起点用 warmup 后第一个点高亮。
    """
    Tb, Td = b_protos.shape[0], d_protos.shape[0]
    Db, Dd = b_protos.shape[2], d_protos.shape[2]
    # 对齐维度 (若不同则各自 PCA 到 min(Db,Dd))
    Dmin = min(Db, Dd)
    def pca_if_needed(arr):
        if arr.shape[2] > Dmin:
            T, C, D = arr.shape
            flat = arr.reshape(T * C, D)
            flat = PCA(n_components=Dmin, random_state=42).fit_transform(flat)
            return flat.reshape(T, C, Dmin)
        return arr
    b_protos = pca_if_needed(b_protos)
    d_protos = pca_if_needed(d_protos)

    # 拼接所有点一起降维 (保证坐标系一致)
    b_flat = b_protos.reshape(Tb * 10, -1)
    d_flat = d_protos.reshape(Td * 10, -1)
    all_flat = np.concatenate([b_flat, d_flat], axis=0)  # (Tb*10 + Td*10, D)
    # 先 PCA 到 50 加速
    if all_flat.shape[1] > 50:
        all_flat = PCA(n_components=50, random_state=42).fit_transform(all_flat)
    print(f"  [t-SNE] fitting {all_flat.shape[0]} points, perplexity={perplexity} ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="pca", learning_rate="auto", n_iter=1500)
    all2d = tsne.fit_transform(all_flat)
    b2d = all2d[:Tb * 10].reshape(Tb, 10, 2)
    d2d = all2d[Tb * 10:].reshape(Td, 10, 2)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))

    for ax, protos2d, T, title in [
        (axes[0], b2d, Tb, "Baseline (hard projection)"),
        (axes[1], d2d, Td, "DACO (DGD-guided)"),
    ]:
        for c in range(10):
            traj = protos2d[:, c, :]  # (T, 2)
            # 轨迹线
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[c],
                    linewidth=0.9, alpha=0.7, zorder=1)
            # 起点warmup后第一个点) 用圆圈
            start = max(warmup, 0)
            if start < T:
                ax.scatter(traj[start, 0], traj[start, 1], color=COLORS[c],
                           s=30, marker="o", edgecolors="black",
                           linewidths=0.4, zorder=3)
            # 终点用三角
            ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS[c],
                       s=35, marker="^", edgecolors="black",
                       linewidths=0.4, zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title, fontsize=11)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # 共享图例 (10 个类别) 放底部
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[c], lw=1.5,
                      marker="o", markersize=5, markeredgecolor="black",
                      markeredgewidth=0.3, label=CIFAR10_CLASSES[c])
               for c in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out_dir, "fig_tsne_trajectories.png")
    plt.savefig(path)
    plt.close()
    print(f"[FIG] {path}")


def main():
    p = argparse.ArgumentParser(description="Baseline vs DACO 可视化")
    p.add_argument("--out", default="outputs/baseline_vs_daco")
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = p.parse_args()

    b, d = load_results(args.out)
    model = b.get("model", "resnet20")
    warmup = b.get("warmup_epochs", 15)
    print(f"Baseline: best_acc={b['best_val_acc']}, "
          f"final_compress={b['final_overall_compress']}x")
    print(f"DACO:     best_acc={d['best_val_acc']}, "
          f"final_compress={d['final_overall_compress']}x")

    plot_curves(b, d, args.out)

    try:
        b_protos = load_protos(args.out, "baseline", model)
        d_protos = load_protos(args.out, "daco", model)
        print(f"  [PROTO] baseline {b_protos.shape}, daco {d_protos.shape}")
        plot_tsne_trajectories(b_protos, d_protos, args.out, warmup,
                               perplexity=args.tsne_perplexity)
    except FileNotFoundError as e:
        print(f"[WARN] 原型文件缺失, 跳过 t-SNE: {e}")

    print(f"\n输出目录: {args.out}")


if __name__ == "__main__":
    main()
