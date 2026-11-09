#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收集 AutoDL 实验结果 → 生成 make_hyper_figs.py 所需数据格式
=============================================================

读取 outputs/autodl_experiments/results/*.json，
按超参数分组汇总，输出:
  1. hparam_results.json    — 结构化结果 (供 make_hyper_figs.py 直接填入)
  2. hparam_summary.md      — Markdown 汇总报告
  3. plot_hyper_beta.png    — β_p 敏感性曲线 (验证精度)
  4. plot_hyper_phi.png     — φ_b 敏感性曲线 (位宽过渡)
  5. plot_hyper_kappa.png   — κ 敏感性曲线 (验证精度)

使用方式:
  python collect_results.py --out outputs/autodl_experiments
"""

import os
import sys
import json
import argparse
from datetime import datetime


def load_results(results_dir):
    """加载所有实验结果 JSON"""
    results = []
    if not os.path.exists(results_dir):
        print(f"  [WARN] 结果目录不存在: {results_dir}")
        return results
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path, "r") as f:
            results.append(json.load(f))
    return results


def group_by_hparam(results):
    """按超参类型分组"""
    groups = {"beta_p": [], "phi_b": [], "kappa": [], "none": []}
    for r in results:
        hp = r.get("hparam", "none")
        if hp in groups:
            groups[hp].append(r)
        else:
            groups["none"].append(r)
    # 按取值排序
    for key in groups:
        groups[key].sort(key=lambda x: float(x.get("hparam_value", 0)) if x.get("hparam_value") != "default" else 0)
    return groups


def generate_fig_data(groups):
    """生成 make_hyper_figs.py 所需的数据格式"""
    fig_data = {
        "rcajs": {
            "beta": {"x": None, "y": None, "optimal_idx": None},
            "phi": {"x": None, "y": None, "optimal_idx": None},
        },
        "mcss": {"x": None, "y": None, "optimal_idx": None, "opt_range": None},
    }

    # β_p → 验证精度
    beta_results = groups["beta_p"]
    if beta_results:
        ok = [r for r in beta_results if r.get("status") == "OK"]
        if ok:
            fig_data["rcajs"]["beta"]["x"] = [float(r["hparam_value"]) for r in ok]
            fig_data["rcajs"]["beta"]["y"] = [r["best_val_acc"] * 100 for r in ok]  # 转百分比
            best_idx = max(range(len(ok)), key=lambda i: ok[i]["best_val_acc"])
            fig_data["rcajs"]["beta"]["optimal_idx"] = best_idx

    # φ_b → 最终平均位宽 (位宽过渡速度的间接指标)
    phi_results = groups["phi_b"]
    if phi_results:
        ok = [r for r in phi_results if r.get("status") == "OK"]
        if ok:
            fig_data["rcajs"]["phi"]["x"] = [float(r["hparam_value"]) for r in ok]
            fig_data["rcajs"]["phi"]["y"] = [r.get("avg_weight_bit", 0) for r in ok]
            # 最优 = 最低位宽 (过渡最快) 且精度可接受
            best_idx = min(range(len(ok)), key=lambda i: ok[i].get("avg_weight_bit", 999))
            fig_data["rcajs"]["phi"]["optimal_idx"] = best_idx

    # κ → 验证精度
    kappa_results = groups["kappa"]
    if kappa_results:
        ok = [r for r in kappa_results if r.get("status") == "OK"]
        if ok:
            fig_data["mcss"]["x"] = [float(r["hparam_value"]) for r in ok]
            fig_data["mcss"]["y"] = [r["best_val_acc"] * 100 for r in ok]
            best_idx = max(range(len(ok)), key=lambda i: ok[i]["best_val_acc"])
            fig_data["mcss"]["optimal_idx"] = best_idx
            # 最优区间: best ± 0.1
            best_val = float(ok[best_idx]["hparam_value"])
            fig_data["mcss"]["opt_range"] = [max(0, best_val - 0.1), min(1.0, best_val + 0.1)]

    return fig_data


def generate_markdown(groups, fig_data):
    """生成 Markdown 汇总报告"""
    lines = [
        "# DACO 超参数实验结果汇总",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 数据集: CIFAR-10",
        f"> 训练: 200 epochs (joint) + 10 warmup + 20 cooldown",
        "",
    ]

    for hp_name, hp_label in [("beta_p", "RCAJS β_p (阻尼系数)"),
                               ("phi_b", "RCAJS φ_b (收缩比)"),
                               ("kappa", "MCSS κ (平衡系数)")]:
        sub = groups[hp_name]
        if not sub:
            continue
        lines += [
            f"## {hp_label}",
            "",
            "| 取值 | Best Val Acc (%) | Final Val Acc (%) | 组稀疏度 | 平均位宽 | 耗时(h) | 状态 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sub:
            best = r.get("best_val_acc", 0)
            final = r.get("final_val_acc", 0)
            sp = r.get("final_group_sparsity", 0)
            awb = r.get("avg_weight_bit", 0)
            hrs = r.get("elapsed_hours", 0)
            status = r.get("status", "?")
            if isinstance(best, float):
                best = f"{best*100:.2f}"
            if isinstance(final, float):
                final = f"{final*100:.2f}"
            lines.append(f"| {r.get('hparam_value', '?')} | {best} | {final} | {sp} | {awb} | {hrs} | {status} |")
        lines.append("")

    # 最优值汇总
    lines += ["## 最优超参取值", ""]
    if fig_data["rcajs"]["beta"]["optimal_idx"] is not None:
        idx = fig_data["rcajs"]["beta"]["optimal_idx"]
        x = fig_data["rcajs"]["beta"]["x"][idx]
        y = fig_data["rcajs"]["beta"]["y"][idx]
        lines.append(f"- **β_p** 最优 = {x} (val_acc = {y:.2f}%)")
    if fig_data["rcajs"]["phi"]["optimal_idx"] is not None:
        idx = fig_data["rcajs"]["phi"]["optimal_idx"]
        x = fig_data["rcajs"]["phi"]["x"][idx]
        y = fig_data["rcajs"]["phi"]["y"][idx]
        lines.append(f"- **φ_b** 最优 = {x} (avg_bit = {y:.2f})")
    if fig_data["mcss"]["optimal_idx"] is not None:
        idx = fig_data["mcss"]["optimal_idx"]
        x = fig_data["mcss"]["x"][idx]
        y = fig_data["mcss"]["y"][idx]
        lines.append(f"- **κ** 最优 = {x} (val_acc = {y:.2f}%)")

    return "\n".join(lines)


def plot_results(groups, out_dir):
    """绘制敏感性曲线"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib 不可用，跳过绘图")
        return

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 0.8

    # β_p
    beta = [r for r in groups["beta_p"] if r.get("status") == "OK"]
    if beta:
        beta.sort(key=lambda r: float(r["hparam_value"]))
        xs = [float(r["hparam_value"]) for r in beta]
        ys = [r["best_val_acc"] * 100 for r in beta]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(xs, ys, "-o", color="#1f77b4", markersize=5, linewidth=1.5, markerfacecolor="white")
        best_i = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[best_i], ys[best_i], "*", color="#d62728", markersize=12, zorder=5)
        ax.set_xlabel(r"Damping coefficient $\beta_p$")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_xscale("log")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        path = os.path.join(out_dir, "plot_hyper_beta.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PNG] {path}")

    # φ_b
    phi = [r for r in groups["phi_b"] if r.get("status") == "OK"]
    if phi:
        phi.sort(key=lambda r: float(r["hparam_value"]))
        xs = [float(r["hparam_value"]) for r in phi]
        ys = [r.get("avg_weight_bit", 0) for r in phi]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(xs, ys, "-o", color="#2ca02c", markersize=5, linewidth=1.5, markerfacecolor="white")
        ax.set_xlabel(r"Contraction ratio $\varphi_b$")
        ax.set_ylabel("Avg. Bit-Width (post-training)")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        path = os.path.join(out_dir, "plot_hyper_phi.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PNG] {path}")

    # κ
    kappa = [r for r in groups["kappa"] if r.get("status") == "OK"]
    if kappa:
        kappa.sort(key=lambda r: float(r["hparam_value"]))
        xs = [float(r["hparam_value"]) for r in kappa]
        ys = [r["best_val_acc"] * 100 for r in kappa]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(xs, ys, "-o", color="#d62728", markersize=5, linewidth=1.5, markerfacecolor="white")
        best_i = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(xs[best_i], ys[best_i], "*", color="#d62728", markersize=12, zorder=5)
        ax.set_xlabel(r"Balance coefficient $\kappa$")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        path = os.path.join(out_dir, "plot_hyper_kappa.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PNG] {path}")


def main():
    parser = argparse.ArgumentParser(description="收集 DACO 超参数实验结果")
    parser.add_argument("--out", type=str, default="outputs/autodl_experiments",
                        help="实验输出目录")
    args = parser.parse_args()

    results_dir = os.path.join(args.out, "results")
    print(f"加载结果: {results_dir}")
    results = load_results(results_dir)
    print(f"  共 {len(results)} 个实验结果")

    if not results:
        print("  [WARN] 没有找到结果文件，请先运行实验")
        return

    groups = group_by_hparam(results)
    for hp, sub in groups.items():
        if sub:
            print(f"  {hp}: {len(sub)} 个")

    # 生成 fig data
    fig_data = generate_fig_data(groups)
    fig_path = os.path.join(args.out, "hparam_results.json")
    with open(fig_path, "w") as f:
        json.dump(fig_data, f, indent=2, ensure_ascii=False)
    print(f"\n[JSON] {fig_path}")

    # 生成 markdown
    md = generate_markdown(groups, fig_data)
    md_path = os.path.join(args.out, "hparam_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"[MD  ] {md_path}")

    # 绘图
    plot_results(groups, args.out)

    # 打印最优值
    print("\n" + "=" * 50)
    print("  最优超参取值:")
    if fig_data["rcajs"]["beta"]["optimal_idx"] is not None:
        idx = fig_data["rcajs"]["beta"]["optimal_idx"]
        print(f"    β_p = {fig_data['rcajs']['beta']['x'][idx]} "
              f"(acc = {fig_data['rcajs']['beta']['y'][idx]:.2f}%)")
    if fig_data["rcajs"]["phi"]["optimal_idx"] is not None:
        idx = fig_data["rcajs"]["phi"]["optimal_idx"]
        print(f"    φ_b = {fig_data['rcajs']['phi']['x'][idx]} "
              f"(avg_bit = {fig_data['rcajs']['phi']['y'][idx]:.2f})")
    if fig_data["mcss"]["optimal_idx"] is not None:
        idx = fig_data["mcss"]["optimal_idx"]
        print(f"    κ   = {fig_data['mcss']['x'][idx]} "
              f"(acc = {fig_data['mcss']['y'][idx]:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
