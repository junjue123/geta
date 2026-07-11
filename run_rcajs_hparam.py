# -*- coding: utf-8 -*-
"""
DACO 论文「Hyperparameter Analysis」补充实验脚本
=================================================

论文 (DACO.tex, IEEE TCSVT) Section IV "Hyperparameter Analysis" 显式分析了
三个关键超参的敏感性，但 run_hparam_sweep.py 只覆盖了其中 MCSS 的 κ
(即 smooth_factor)，而 RCAJS 的两个超参此前并未被扫描：

  · β_p  (damping coefficient, 阻尼系数)
        → 论文 Fig. hyper_rcajs(a): Impact of β_p on validation accuracy.
        → GETA 优化器内部已接线 RCAJSController，β_p 通过
          r_p = r_base * exp(-β_p · Δ_p) 调节自适应剪枝率
          (geta.py:RCAJSController.compute_adaptive_prune_rate, L156)。
          本脚本在构造优化器后注入 opt.rcajs.beta_p 即可真实生效。

  · φ_b  (contraction ratio, 收缩比)
        → 论文 Fig. hyper_rcajs(b): Impact of φ_b on bit-width transition speed.
        → 论文规则: 当「低于 max_bit 的层比例」φ_b_measured > φ_b_threshold 时，
          触发一次位宽收缩 (max_bit → max_bit - bit_reduction)，并受 Guard
          bit-width b_ε (= min_bit_wt) 硬下界保护。
        → 由于 GETA 的投影阶段在剪枝之前、层尚未被量化到低位，
          内部 update_bit_adaptive 在短时训练里几乎不触发，故本脚本在训练
          循环外部实现论文的 φ_b 门控调度器，直接驱动 opt.max_bit_wt
          (compute_gamma_d 每步以 [min_bit_wt, max_bit_wt] 为搜索上界，
           见 geta.py L1286，因此外部调低 max_bit_wt 能真实改变量化位宽)。

说明:
  · 训练使用随机数据 + VGG7 + GETA (W&A 量化)，单 trial ~ 数秒，CPU 可跑。
  · 「验证精度」为随机数据下的代理指标 (proxy = 1/(1+final_loss))，
    仅用于观察 β_p / φ_b 的相对趋势，论文真实曲线需用 CIFAR-10 全量训练替换
    run_beta_trial / run_phi_trial 中的训练循环。
  · β_p 与 φ_b 均固定 target_group_sparsity=0.5 以隔离单变量效应
    (论文主实验默认 φ_b=0.9, b_ε=4, κ=0.1, W=5)。

----------------------------------------------------------------------
启用方式 (How to run)
----------------------------------------------------------------------
  # 1) 跑全部 RCAJS 超参实验 (β_p + φ_b)
  CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py

  # 2) 只扫 β_p
  CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --mode beta

  # 3) 只扫 φ_b
  CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --mode phi

  # 4) 快速冒烟 (各取 3 个值)
  CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --quick

  # 5) 自定义步数 / 输出目录 / 跳过绘图
  CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --mode all --steps 20 \
      --out outputs/rcajs_hparam --no-plot
"""

import os
import sys
import csv
import math
import time
import argparse
import traceback
from datetime import datetime

# ---- 强制 CPU、关掉 CUDA，保证轻量可复现 -------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

# 复用 run_hparam_sweep 的模型/优化器构造与指标采集
from run_hparam_sweep import (
    build_trial,
    set_seed,
    reset_mcss_history,
    _collect_metrics,
    BASE_PARAMS,
)


# ===========================================================================
# 0. 论文 Hyperparameter Analysis 的扫描取值
# ===========================================================================
# β_p: 论文讨论从「极低 (退化为刚性静态调度, 性能崩溃)」到「极高 (过度保守,
#      收敛迟缓)」的连续区间；取对数尺度采样覆盖。
BETA_SWEEP = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]

# φ_b: 论文讨论 high (0.95) stalls、low (0.1) premature；取 0.1~0.9 覆盖。
PHI_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]

# 单变量隔离用的固定目标稀疏度 (论文主实验 φ_b=0.9 等，此处仅用于隔离 β_p/φ_b)
TARGET_SPARSITY = 0.5


# ===========================================================================
# 1. φ_b 外部门控调度器 (论文 Section III-A-2 / Fig hyper_rcajs(b))
# ===========================================================================
def rcajs_phi_scheduler(opt, phi_b_threshold, bit_reduction, min_bit_wt):
    """
    论文规则: 当「当前权重位宽低于 max_bit 的层比例」φ_b_measured > φ_b_threshold 时，
    触发一次位宽收缩 max_bit_wt -= bit_reduction，并以 min_bit_wt (Guard b_ε) 为硬下界。

    返回 True 表示本次触发了收缩 (用于统计过渡速度)。
    """
    bit_layers = getattr(opt, "bit_layers", None)
    if not bit_layers:
        return False
    below = 0
    total = 0
    for v in bit_layers.values():
        if isinstance(v, dict) and v.get("weight") is not None:
            total += 1
            if v["weight"] < opt.max_bit_wt:
                below += 1
    if total == 0:
        return False
    phi_b_measured = below / total
    if phi_b_measured > phi_b_threshold and opt.max_bit_wt > min_bit_wt:
        opt.max_bit_wt = max(int(opt.max_bit_wt) - int(bit_reduction), int(min_bit_wt))
        return True
    return False


# ===========================================================================
# 2. 单个 trial 运行 (β_p 或 φ_b)
# ===========================================================================
def run_trial(var_name, var_value, device, num_steps, seed_base=20240706,
              max_retry=3, bit_reduction=2, min_bit_wt=2):
    """
    运行单个 RCAJS 超参 trial。
    var_name: "beta_p" 或 "phi_b_threshold"
    var_value: 对应超参取值
    返回指标字典 (含逐 step 轨迹)。
    """
    # 构造配置: 固定 target_group_sparsity 隔离单变量
    config = dict(BASE_PARAMS)
    config["target_group_sparsity"] = TARGET_SPARSITY
    config["bit_reduction"] = bit_reduction
    config["min_bit_wt"] = min_bit_wt
    if var_name == "beta_p":
        config["beta_p"] = var_value
    else:
        config["phi_b_threshold"] = var_value

    traj = {"loss": [], "group_sparsity": [], "avg_weight_bit": [], "min_weight_bit": []}
    last_loss = float("nan")
    ok, err, used_seed = False, "", None
    model = opt = criterion = None
    t0 = time.time()

    for attempt in range(max_retry):
        used_seed = seed_base + attempt
        set_seed(used_seed)
        reset_mcss_history()
        try:
            model, opt, criterion = build_trial(config, device)
            # ---- 注入 RCAJS 超参到已接线的 RCAJSController ----
            if var_name == "beta_p":
                opt.rcajs.beta_p = float(var_value)
            else:
                opt.rcajs.phi_b_threshold = float(var_value)

            model.train()
            X = torch.randn(2, 3, 32, 32)
            y = torch.randint(0, 10, (2,))
            for step in range(num_steps):
                loss = criterion(model(X), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # φ_b 模式: 每步外部门控调度 (β_p 模式不调用，保持默认 φ_b=0.8)
                if var_name == "phi_b_threshold":
                    rcajs_phi_scheduler(opt, float(var_value), bit_reduction, min_bit_wt)
                last_loss = loss.item()
                sp, nzg, awb, mwb = _collect_metrics(opt)
                traj["loss"].append(round(last_loss, 4))
                traj["group_sparsity"].append(round(sp, 4) if isinstance(sp, (int, float)) else None)
                traj["avg_weight_bit"].append(round(awb, 2) if isinstance(awb, (int, float)) else None)
                traj["min_weight_bit"].append(mwb if isinstance(mwb, (int, float)) else None)
            ok = True
            err = ""
            break
        except Exception as e:
            tb = traceback.format_exc().splitlines()
            err = (tb[-2] + " | " + tb[-1])[:200] if len(tb) >= 2 else repr(e)
            if attempt < max_retry - 1:
                print(f"(retry {attempt+1}, seed={used_seed})", end=" ", flush=True)
            continue
    elapsed = time.time() - t0

    group_sparsity = traj["group_sparsity"][-1] if traj["group_sparsity"] else float("nan")
    avg_wb = traj["avg_weight_bit"][-1] if traj["avg_weight_bit"] else float("nan")
    min_wb = traj["min_weight_bit"][-1] if traj["min_weight_bit"] else float("nan")

    # 代理验证精度 (随机数据下): proxy = 1/(1+final_loss)，越高越好
    if ok and isinstance(last_loss, (int, float)) and math.isfinite(last_loss):
        proxy_acc = 1.0 / (1.0 + max(last_loss, 0.0))
    else:
        proxy_acc = 0.0

    # φ_b 过渡速度代理: 起始 max_bit 与最终 avg 位宽之差 (越大 = 过渡越快)
    init_max = float(config.get("max_bit_wt", 16))
    transition_drop = (init_max - avg_wb) if isinstance(avg_wb, (int, float)) else float("nan")

    return {
        "var": var_name,
        "value": var_value,
        "final_loss": round(last_loss, 4) if ok else None,
        "proxy_acc": round(proxy_acc, 4),
        "group_sparsity": group_sparsity,
        "avg_weight_bit": avg_wb,
        "min_weight_bit": min_wb,
        "transition_drop": transition_drop,
        "steps": num_steps,
        "seconds": round(elapsed, 1),
        "seed": used_seed,
        "status": "OK" if ok else "FAIL",
        "error": err[:120] if err else "",
        "trajectory": traj,
    }


# ===========================================================================
# 3. 模块级扫描
# ===========================================================================
def run_beta(device, num_steps, values=None, limit=None):
    values = values or BETA_SWEEP
    if limit is not None:
        values = values[:limit]
    print(f"\n{'='*70}\n  RCAJS β_p sweep ({len(values)} values, {num_steps} steps)\n{'='*70}")
    results = []
    for i, v in enumerate(values):
        print(f"  [{i+1}/{len(values)}] β_p={v} ...", end=" ", flush=True)
        r = run_trial("beta_p", v, device, num_steps)
        tag = f"loss={r['final_loss']}, sp={r['group_sparsity']}, proxyAcc={r['proxy_acc']}" \
            if r["status"] == "OK" else f"FAIL: {r['error']}"
        print(tag, flush=True)
        results.append(r)
    return results


def run_phi(device, num_steps, values=None, limit=None):
    values = values or PHI_SWEEP
    if limit is not None:
        values = values[:limit]
    print(f"\n{'='*70}\n  RCAJS φ_b sweep ({len(values)} values, {num_steps} steps)\n{'='*70}")
    results = []
    for i, v in enumerate(values):
        print(f"  [{i+1}/{len(values)}] φ_b={v} ...", end=" ", flush=True)
        r = run_trial("phi_b_threshold", v, device, num_steps)
        tag = f"avgBit={r['avg_weight_bit']}, drop={r['transition_drop']}, sp={r['group_sparsity']}" \
            if r["status"] == "OK" else f"FAIL: {r['error']}"
        print(tag, flush=True)
        results.append(r)
    return results


# ===========================================================================
# 4. 输出: CSV / Markdown
# ===========================================================================
FIELDS = [
    "var", "value", "final_loss", "proxy_acc", "group_sparsity",
    "avg_weight_bit", "min_weight_bit", "transition_drop",
    "steps", "seconds", "seed", "status", "error",
]


def write_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n[CSV] 已写入: {path}")


def write_markdown(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    titles = {
        "beta_p": "RCAJS β_p (阻尼系数) 敏感性分析 — 论文 Fig. hyper_rcajs(a)",
        "phi_b_threshold": "RCAJS φ_b (收缩比) 敏感性分析 — 论文 Fig. hyper_rcajs(b)",
    }
    lines = ["# DACO 论文 RCAJS 超参数敏感性实验报告", "",
             f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"> 模型: VGG7 + GETA (W&A 量化), target_group_sparsity=固定 {TARGET_SPARSITY}",
             f"> 注: 验证精度为随机数据代理指标 (proxy=1/(1+final_loss))\n"]
    for var in ("beta_p", "phi_b_threshold"):
        sub = [r for r in results if r["var"] == var]
        if not sub:
            continue
        title = titles.get(var, var)
        lines += [f"## {title}", "",
                  "| 取值 | Final Loss | 代理精度 | 组稀疏度 | 平均权重位宽 | 过渡下降 | 状态 |",
                  "|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(
                f"| {r['value']} | {r['final_loss']} | {r['proxy_acc']} | "
                f"{r['group_sparsity']} | {r['avg_weight_bit']} | {r['transition_drop']} | {r['status']} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[MD ] 已写入: {path}")


# ===========================================================================
# 5. 可视化: 论文风格敏感性曲线
# ===========================================================================
def plot_beta(results, out_dir):
    """论文 Fig. hyper_rcajs(a): β_p 对 (代理) 验证精度 / 组稀疏度 的影响。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    sub = [r for r in results if r["var"] == "beta_p" and r["status"] == "OK"]
    if not sub:
        return None
    sub.sort(key=lambda r: r["value"])
    xs = [r["value"] for r in sub]
    acc = [r["proxy_acc"] for r in sub]
    sp = [r["group_sparsity"] for r in sub]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RCAJS β_p 敏感性分析 (论文 Fig. hyper_rcajs(a))", fontsize=13, fontweight="bold")

    axes[0].plot(xs, acc, marker="o", color="#1f77b4", label="代理验证精度")
    axes[0].set_xlabel("β_p (阻尼系数)")
    axes[0].set_ylabel("代理验证精度 = 1/(1+final_loss)")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(xs, sp, marker="s", color="#ff7f0e", label="最终组稀疏度")
    axes[1].axhline(TARGET_SPARSITY, color="gray", ls="--", lw=1, label=f"目标稀疏度={TARGET_SPARSITY}")
    axes[1].set_xlabel("β_p (阻尼系数)")
    axes[1].set_ylabel("组稀疏度")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "plot_rcajs_beta.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_phi(results, out_dir):
    """论文 Fig. hyper_rcajs(b): φ_b 对位宽过渡速度的影响。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    sub = [r for r in results if r["var"] == "phi_b_threshold" and r["status"] == "OK"]
    if not sub:
        return None
    sub.sort(key=lambda r: r["value"])
    xs = [r["value"] for r in sub]
    avgbit = [r["avg_weight_bit"] for r in sub]
    drop = [r["transition_drop"] for r in sub]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RCAJS φ_b 敏感性分析 (论文 Fig. hyper_rcajs(b))", fontsize=13, fontweight="bold")

    # 左: 最终平均权重位宽 vs φ_b (高位宽=过渡慢/停滞, 低位宽=过渡快/过早)
    axes[0].plot(xs, avgbit, marker="o", color="#2ca02c", label="最终平均权重位宽")
    axes[0].set_xlabel("φ_b (收缩比阈值)")
    axes[0].set_ylabel("最终平均权重位宽 (bit)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # 右: 过渡下降量 vs φ_b (越大=过渡越快)
    axes[1].plot(xs, drop, marker="s", color="#d62728", label="位宽过渡下降量")
    axes[1].set_xlabel("φ_b (收缩比阈值)")
    axes[1].set_ylabel("过渡下降 (初始max_bit - 最终avg_bit)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "plot_rcajs_phi.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_phi_trajectories(results, out_dir):
    """φ_b 各组的平均权重位宽轨迹叠加 (直观看过渡速度差异)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    sub = [r for r in results if r["var"] == "phi_b_threshold" and r["status"] == "OK"]
    if not sub:
        return None
    sub.sort(key=lambda r: r["value"])
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("RCAJS φ_b: 平均权重位宽轨迹 (论文 Fig. hyper_rcajs(b) 轨迹版)",
                 fontsize=13, fontweight="bold")
    for i, r in enumerate(sub):
        tr = r["trajectory"].get("avg_weight_bit", [])
        if not tr:
            continue
        ax.plot(range(len(tr)), tr, marker="o", markersize=3,
                label=f"φ_b={r['value']}", color=colors[i % len(colors)])
    ax.set_xlabel("step")
    ax.set_ylabel("平均权重位宽 (bit)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "plot_rcajs_phi_traj.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ===========================================================================
# 6. 入口
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="DACO 论文 RCAJS 超参数敏感性实验 (β_p / φ_b)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["beta", "phi", "all"],
                        help="扫描哪个 RCAJS 超参 (默认 all)")
    parser.add_argument("--steps", type=int, default=12,
                        help="每个 trial 的训练步数 (默认 12)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (默认 cpu)")
    parser.add_argument("--out", type=str, default="outputs/rcajs_hparam",
                        help="结果输出目录")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: β_p/φ_b 各仅取 3 个代表值")
    parser.add_argument("--no-plot", action="store_true",
                        help="不生成对比图 (仅 CSV + Markdown)")
    args = parser.parse_args()

    beta_limit = 3 if args.quick else None
    phi_limit = 3 if args.quick else None

    all_results = []
    if args.mode in ("beta", "all"):
        all_results += run_beta(args.device, args.steps, limit=beta_limit)
    if args.mode in ("phi", "all"):
        all_results += run_phi(args.device, args.steps, limit=phi_limit)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"rcajs_hparam_{args.mode}_{ts}.csv")
    md_path = os.path.join(args.out, f"rcajs_hparam_{args.mode}_{ts}.md")
    write_csv(all_results, csv_path)
    write_markdown(all_results, md_path)

    if not args.no_plot:
        try:
            paths = []
            if args.mode in ("beta", "all"):
                p = plot_beta(all_results, args.out)
                if p:
                    paths.append(p)
            if args.mode in ("phi", "all"):
                p1 = plot_phi(all_results, args.out)
                p2 = plot_phi_trajectories(all_results, args.out)
                if p1:
                    paths.append(p1)
                if p2:
                    paths.append(p2)
            for p in paths:
                print(f"[PNG] 已生成: {p}")
        except Exception as e:
            print(f"[Warn] 绘图失败 (可加 --no-plot 跳过): {e}")

    print(f"\n完成。共 {len(all_results)} 个 trial。")


if __name__ == "__main__":
    main()
