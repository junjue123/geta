# -*- coding: utf-8 -*-
"""
DACO 超参数实验脚本 (Hyperparameter Sweep for DACO)
====================================================

对 DACO 三大模块的关键超参数进行扫描实验，在轻量 CNN (VGG7) 上跑短时间训练，
收集「压缩率 / 精度 / 位宽」等指标，输出 CSV 对比表 + Markdown 报告。

对应论文三大模块 (详见 DACO_论文与代码对照报告.md)：
  · RCAJS (Resource-Constrained Adaptive Joint Sparsity) 资源约束自适应调度
      可控超参: target_group_sparsity, pruning_periods, bit_reduction,
                max_bit_wt, min_bit_wt
      (注: RCAJS 的 β_p / φ_b 论文超参分析见 run_rcajs_hparam.py ——
       二者在 GETA 优化器内已接线为 RCAJSController，但需外部扫描脚本驱动)
  · MCSS  (Multi-Criteria Calibrated Saliency Score)  多标准校准重要性分数
      可控超参: importance_score_criteria (代理权重), mcss_smooth_factor (κ),
                mcss_history_window (W, 稳定性窗口)
  · DGD   (Diffusion Gradient Descent)  扩散梯度下降 (朗之万噪声)
      可控超参: diffusion_noise_ratio, diffusion_steps, diffusion_noise_init

----------------------------------------------------------------------
启用方式 (How to run)
----------------------------------------------------------------------
  # 1) 单模块扫描 (推荐，最快) —— 只扫 RCAJS
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module rcajs

  # 2) 扫 MCSS (含 smooth_factor / history_window 接线)
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module mcss

  # 3) 扫 DGD
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module dgd

  # 4) 全部模块扫描
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all

  # 5) 快速验证 (每个模块只跑 2 个配置，用于冒烟测试)
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all --quick

  # 6) 自定义步数 / 设备 / 输出目录
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module rcajs --steps 20 --device cpu --out outputs/hparam_sweep

  # 7) 不生成对比图 (仅 CSV + Markdown，适合无 matplotlib 环境)
  CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all --no-plot

说明:
  · 默认 CPU 运行 (CUDA_VISIBLE_DEVICES="")，单 trial ~ 8~15s，适合快速扫参。
  · 每个 trial 是独立的 VGG7 + GETA 短训练，互不影响。
  · 每个 trial 记录逐 step 轨迹 (Loss / 组稀疏度 / 平均权重位宽)，
    结束后按模块生成对比图 plot_{rcajs,mcss,dgd}.png (matplotlib)。
  · MCSS 的 smooth_factor / history_window 原框架未在优化器内接线，
    本脚本通过运行时 patch compute_importance_scores 注入 MCSS 校准，
    保证扫描真实生效 (research 模式)。
"""

import os
import sys
import time
import types
import argparse
import csv
import random
import traceback
import numpy as np
from datetime import datetime

# ---- 强制 CPU、关掉 CUDA，保证轻量可复现 -------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn


def set_seed(seed):
    """固定随机种子，保证单次 trial 可被复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 0. 基础共享参数 (所有 trial 共用，除非被具体配置覆盖)
# ===========================================================================
BASE_PARAMS = dict(
    variant="sgd",
    lr=1e-3,
    lr_quant=1e-3,
    # 剪枝 / 量化阶段调度 (足够短以便快速扫参)
    start_projection_step=0,
    projection_periods=1,
    projection_steps=2,
    start_pruning_step=2,
    pruning_periods=1,
    pruning_steps=6,
    # 默认位宽范围
    bit_reduction=2,
    min_bit_wt=2,
    max_bit_wt=16,
    min_bit_act=2,
    max_bit_act=16,
    # DGD 默认噪声
    diffusion_noise_init=1e-4,
    diffusion_steps=None,          # None -> 默认等于 pruning_steps
    diffusion_noise_ratio=0.5,
    # 默认重要性代理权重 (论文 default)
    importance_score_criteria="default",
    verbose="False",
    device="cpu",
    log_dir="outputs/hparam_sweep_logs",
)


# ===========================================================================
# 1. 扫描配置：每个模块一组待扫超参 (value 直接作为 GETA / MCSS 构造参数)
# ===========================================================================
# 约定:
#   key 以 "mcss_" 开头 -> 仅 MCSS 接线使用 (smooth_factor / history_window)
#   其余 key        -> 直接透传给 GETA 优化器构造器
SWEEP = {
    # ---------------------- RCAJS ----------------------
    "rcajs": [
        {"label": "sparsity=0.3", "target_group_sparsity": 0.3},
        {"label": "sparsity=0.5", "target_group_sparsity": 0.5},
        {"label": "sparsity=0.7", "target_group_sparsity": 0.7},
        {"label": "periods=3",    "pruning_periods": 3, "pruning_steps": 9},
        {"label": "periods=5",    "pruning_periods": 5, "pruning_steps": 10},
        {"label": "bit_red=1",    "bit_reduction": 1},
        {"label": "bit_red=4",    "bit_reduction": 4},
        {"label": "maxbw=8",      "max_bit_wt": 8},
        {"label": "maxbw=32",     "max_bit_wt": 32},
        {"label": "minbw=4",      "min_bit_wt": 4},
    ],
    # ---------------------- MCSS ----------------------
    "mcss": [
        {"label": "kappa=0.4",  "mcss_smooth_factor": 0.4},
        {"label": "kappa=0.8",  "mcss_smooth_factor": 0.8},
        {"label": "kappa=1.0",  "mcss_smooth_factor": 1.0},
        {"label": "W=3",        "mcss_history_window": 3},
        {"label": "W=10",       "mcss_history_window": 10},
        {"label": "mag-only",   "importance_score_criteria": {"magnitude": 1.0}},
        {"label": "cos-only",   "importance_score_criteria": {"cosine_similarity": 1.0}},
        {"label": "taylor-only","importance_score_criteria": {"taylor_first_order": 1.0, "taylor_second_order": 1.0}},
    ],
    # ---------------------- DGD ----------------------
    "dgd": [
        {"label": "noise=0.1",  "diffusion_noise_ratio": 0.1},
        {"label": "noise=0.5",  "diffusion_noise_ratio": 0.5},
        {"label": "noise=1.0",  "diffusion_noise_ratio": 1.0},
        {"label": "dsteps=2",   "diffusion_steps": 2},
        {"label": "dsteps=12",  "diffusion_steps": 12},
        {"label": "ninit=1e-5", "diffusion_noise_init": 1e-5},
        {"label": "ninit=1e-3", "diffusion_noise_init": 1e-3},
    ],
}


# ===========================================================================
# 2. MCSS 运行时接线：patch compute_importance_scores，注入校准
# ===========================================================================
def reset_mcss_history():
    """清空跨 trial 的全局分数历史与边界，避免稳定性因子 / 归一化污染。"""
    try:
        from only_train_once.optimizer.importance_score import _SCORE_HISTORY, _GLOBAL_SCORE_BOUNDS
        _SCORE_HISTORY.clear()
        _GLOBAL_SCORE_BOUNDS.clear()
    except Exception:
        pass


def apply_mcss_wiring(opt, smooth_factor=0.8, history_window=5):
    """
    将 MCSS 校准 (adjust_importance_criteria) 接线进 GETA 优化器。
    原框架 adjust_importance_criteria 未被优化器调用，这里在
    compute_importance_scores 之后补做校准，并重新聚合 'overall' 分数，
    使 smooth_factor / history_window 真正影响剪枝决策。
    """
    from only_train_once.optimizer.importance_score import adjust_importance_criteria

    original = opt.compute_importance_scores

    def wrapped(**kwargs):
        original(**kwargs)
        # 1) 逐组应用 MCSS 校准 (Φ(d) 保真系数 + Ψ 稳定性因子)
        for group in opt.param_groups:
            if group.get("is_prunable") and not group.get("is_auxiliary"):
                adjust_importance_criteria(
                    group,
                    opt.bit_layers,
                    history_window=history_window,
                    smooth_factor=smooth_factor,
                )
        # 2) 重新聚合 overall (校准后分数需重算加权总和)
        for group in opt.param_groups:
            if group.get("is_prunable") and not group.get("is_auxiliary"):
                overall = None
                for proxy_name in opt.importance_score_criteria:
                    if proxy_name not in group["importance_scores"]:
                        continue
                    s = group["importance_scores"][proxy_name]
                    overall = s.clone() if overall is None else overall + s
                if overall is not None:
                    group["importance_scores"]["overall"] = overall

    opt.compute_importance_scores = types.MethodType(wrapped, opt)
    return opt


# ===========================================================================
# 3. 构建模型 + 优化器 (按 config 覆盖 BASE_PARAMS)
# ===========================================================================
def build_trial(config, device):
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO

    # 分离 MCSS 专用参数 (不传给 GETA 构造器，仅用于运行时接线)
    mcss_smooth = config.pop("mcss_smooth_factor", None)
    mcss_window = config.pop("mcss_history_window", None)

    # oto.geta() 是固定签名 wrapper，不会透传 DGD 超参，需在构造后设为属性。
    # 这里把 DGD 相关键摘出，构造完再注入。
    dgd_init = config.pop("diffusion_noise_init", None)
    dgd_steps = config.pop("diffusion_steps", None)
    dgd_ratio = config.pop("diffusion_noise_ratio", None)

    # 合并参数
    params = dict(BASE_PARAMS)
    params.update(config)
    params["device"] = device

    # oto.geta 实际接受的参数白名单 (其余在构造后作为属性注入)
    GETA_ACCEPTED = {
        "lr", "lr_quant", "weight_decay", "first_momentum", "second_momentum",
        "variant", "target_group_sparsity", "start_projection_step",
        "projection_steps", "projection_periods", "start_pruning_step",
        "pruning_steps", "pruning_periods", "dampening", "group_divisible",
        "fixed_zero_groups", "importance_score_criteria", "bit_reduction",
        "min_bit_wt", "max_bit_wt", "min_bit_act", "max_bit_act",
        "grad_clip_min", "grad_clip_max", "verbose", "device", "log_dir",
    }
    geta_params = {k: v for k, v in params.items() if k in GETA_ACCEPTED}

    model = model_to_quantize_model(
        vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
    )
    dummy = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model, dummy_input=dummy)
    opt = oto.geta(**geta_params)

    # ---- 注入 oto.geta 未透传的 DGD 超参 (GETA step 中会读取这些属性) ----
    if dgd_init is not None:
        opt.diffusion_noise_init = dgd_init
    if dgd_ratio is not None:
        opt.diffusion_noise_ratio = dgd_ratio
    if dgd_steps is not None:
        opt.diffusion_steps = dgd_steps
    else:
        # 对齐 GETA.__init__: diffusion_steps 默认 = pruning_steps
        opt.diffusion_steps = opt.pruning_steps

    # MCSS 接线 (若有)
    if mcss_smooth is not None or mcss_window is not None:
        apply_mcss_wiring(
            opt,
            smooth_factor=mcss_smooth if mcss_smooth is not None else 0.8,
            history_window=mcss_window if mcss_window is not None else 5,
        )

    criterion = nn.CrossEntropyLoss()
    return model, opt, criterion


# ===========================================================================
# 4. 运行单个 trial，返回指标字典
# ===========================================================================
def _collect_metrics(opt):
    """从优化器收集当前 (组稀疏度, 零组数, 平均权重位宽, 最小权重位宽)。"""
    group_sparsity, num_zero_groups = float("nan"), 0
    avg_wb, min_wb = float("nan"), float("nan")
    try:
        m = opt.compute_metrics()
        group_sparsity = m.group_sparsity
        num_zero_groups = m.num_zero_groups
    except Exception:
        pass
    try:
        wbits = [v.get("weight") for v in opt.bit_layers.values() if v.get("weight") is not None]
        if wbits:
            avg_wb = sum(wbits) / len(wbits)
            min_wb = min(wbits)
    except Exception:
        pass
    return group_sparsity, num_zero_groups, avg_wb, min_wb


def run_trial(config, device, num_steps, seed_base=20240706, max_retry=3):
    """
    运行单个 trial，并返回逐 step 轨迹 + 最终指标。
    · 每个 trial 用确定性种子 (seed_base + 序号) 初始化，保证可复现。
    · GETA 在极少数随机初始化下会触发潜在的除零 bug (数据相关)，
      故加 retry: 失败则换种子重建重跑，最多 max_retry 次；
      框架层面已对 d_quant 的 bit_width==1 除零做钳制防御。
    """
    label = config.get("label", "?")
    mcss_keys = [k for k in ("mcss_smooth_factor", "mcss_history_window") if k in config]
    module = "mcss" if mcss_keys else (
        "dgd" if any(k.startswith("diffusion") for k in config) else "rcajs"
    )
    # 重建 config (build_trial 会 pop MCSS 键，这里用副本)
    cfg = dict(config)

    # 逐 step 轨迹
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
            model, opt, criterion = build_trial(cfg, device)
            model.train()
            X = torch.randn(2, 3, 32, 32)
            y = torch.randint(0, 10, (2,))
            for step in range(num_steps):
                loss = criterion(model(X), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
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

    # 最终指标 (用轨迹末点)
    group_sparsity = traj["group_sparsity"][-1] if traj["group_sparsity"] else float("nan")
    avg_wb = traj["avg_weight_bit"][-1] if traj["avg_weight_bit"] else float("nan")
    min_wb = traj["min_weight_bit"][-1] if traj["min_weight_bit"] else float("nan")
    num_zero_groups = 0
    final_max_bw = float("nan")
    if ok and opt is not None:
        try:
            num_zero_groups = opt.compute_metrics().num_zero_groups
        except Exception:
            pass
        final_max_bw = getattr(opt, "max_bit_wt", float("nan"))

    # 把覆盖的超参汇成紧凑字符串
    override_str = ", ".join(
        f"{k}={v}" for k, v in config.items() if k != "label"
    )

    return {
        "module": module,
        "label": label,
        "override": override_str,
        "final_loss": round(last_loss, 4) if ok else None,
        "group_sparsity": group_sparsity,
        "num_zero_groups": num_zero_groups,
        "avg_weight_bit": avg_wb,
        "min_weight_bit": min_wb,
        "final_max_bit_wt": final_max_bw,
        "steps": num_steps,
        "seconds": round(elapsed, 1),
        "seed": used_seed,
        "status": "OK" if ok else "FAIL",
        "error": err[:120] if err else "",
        "trajectory": traj,
    }


# ===========================================================================
# 5. 模块级扫描
# ===========================================================================
def run_module(module, device, num_steps, limit=None):
    configs = SWEEP[module]
    if limit is not None:
        configs = configs[:limit]
    print(f"\n{'='*70}\n  Module: {module.upper()}  ({len(configs)} configs, {num_steps} steps)\n{'='*70}")
    results = []
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {cfg.get('label','?')} ...", end=" ", flush=True)
        r = run_trial(cfg, device, num_steps)
        tag = f"loss={r['final_loss']}, sp={r['group_sparsity']}, avgB={r['avg_weight_bit']}" \
              if r["status"] == "OK" else f"FAIL: {r['error']}"
        print(tag, flush=True)
        results.append(r)
    return results


# ===========================================================================
# 6. 输出: 表格 / CSV / Markdown
# ===========================================================================
FIELDS = [
    "module", "label", "override", "final_loss", "group_sparsity",
    "num_zero_groups", "avg_weight_bit", "min_weight_bit",
    "final_max_bit_wt", "steps", "seconds", "seed", "status", "error",
]


def print_table(results):
    cols = ["module", "label", "final_loss", "group_sparsity", "avg_weight_bit",
            "min_weight_bit", "final_max_bit_wt", "seconds", "status"]
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in results)) for c in cols}
    width = {c: min(width[c] + 2, 22) for c in cols}
    line = "  ".join(c.ljust(width[c]) for c in cols)
    print("\n" + "-" * len(line))
    print(line)
    print("-" * len(line))
    for r in results:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    print("-" * len(line))


def write_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        # extrasaction="ignore": trajectory 等大字段不写入 CSV
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n[CSV] 已写入: {path}")


def write_markdown(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# DACO 超参数扫描实验报告", "",
             f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"> 模型: VGG7 + GETA (W&A 量化)",
             f"> Trial 数: {len(results)}", ""]
    # 按模块分组
    for module in ("rcajs", "mcss", "dgd"):
        sub = [r for r in results if r["module"] == module]
        if not sub:
            continue
        title = {"rcajs": "RCAJS (资源约束自适应调度)",
                 "mcss": "MCSS (多标准校准重要性分数)",
                 "dgd": "DGD (扩散梯度下降)"}[module]
        lines += [f"## {title}", "",
                  "| 配置 | 覆盖超参 | Loss | 组稀疏度 | 平均权重位宽 | 最小位宽 | 最终max_bw | 耗时(s) | 状态 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(
                f"| {r['label']} | {r['override']} | {r['final_loss']} | "
                f"{r['group_sparsity']} | {r['avg_weight_bit']} | {r['min_weight_bit']} | "
                f"{r['final_max_bit_wt']} | {r['seconds']} | {r['status']} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[MD ] 已写入: {path}")


# ===========================================================================
# 6b. 可视化: 每模块一张对比图 (Loss / 组稀疏度 / 平均权重位宽 轨迹)
# ===========================================================================
def plot_module(results, module, out_dir):
    """为单个模块绘制超参对比图，返回 PNG 路径或 None。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Windows 中文环境: 使用系统自带中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    sub = [r for r in results if r["module"] == module and r["status"] == "OK"]
    if not sub:
        return None

    titles = {
        "rcajs": "RCAJS (资源约束自适应联合稀疏)",
        "mcss": "MCSS (多标准校准重要性分数)",
        "dgd": "DGD (扩散梯度下降 / 朗之万噪声)",
    }
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(titles.get(module, module.upper()), fontsize=14, fontweight="bold")

    metrics = [
        ("loss", "Loss 轨迹", "loss"),
        ("group_sparsity", "组稀疏度轨迹", "group sparsity"),
        ("avg_weight_bit", "平均权重位宽轨迹", "avg weight bit"),
    ]
    for ax, (key, ylabel, _) in zip(axes, metrics):
        for i, r in enumerate(sub):
            tr = r["trajectory"].get(key, [])
            if not tr:
                continue
            ax.plot(range(len(tr)), tr, marker="o", markersize=3,
                    label=r["label"], color=colors[i % len(colors)])
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"plot_{module}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_all(results, out_dir):
    paths = []
    for module in ("rcajs", "mcss", "dgd"):
        p = plot_module(results, module, out_dir)
        if p:
            paths.append(p)
    return paths


# ===========================================================================
# 7. 入口
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="DACO 超参数扫描实验")
    parser.add_argument("--module", type=str, default="all",
                        choices=["rcajs", "mcss", "dgd", "all"],
                        help="扫描哪个 DACO 模块 (默认 all)")
    parser.add_argument("--steps", type=int, default=12,
                        help="每个 trial 的训练步数 (默认 12)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (默认 cpu)")
    parser.add_argument("--out", type=str, default="outputs/hparam_sweep",
                        help="结果输出目录")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 每个模块只跑前 2 个配置")
    parser.add_argument("--no-plot", action="store_true",
                        help="不生成对比图 (仅 CSV + Markdown)")
    args = parser.parse_args()

    modules = ["rcajs", "mcss", "dgd"] if args.module == "all" else [args.module]
    limit = 2 if args.quick else None

    all_results = []
    for mod in modules:
        all_results += run_module(mod, args.device, args.steps, limit=limit)

    print_table(all_results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"sweep_{args.module}_{ts}.csv")
    md_path = os.path.join(args.out, f"sweep_{args.module}_{ts}.md")
    write_csv(all_results, csv_path)
    write_markdown(all_results, md_path)

    if not args.no_plot:
        try:
            plots = plot_all(all_results, args.out)
            for p in plots:
                print(f"[PNG] 已生成: {p}")
        except Exception as e:
            print(f"[Warn] 绘图失败 (可加 --no-plot 跳过): {e}")

    print(f"\n完成。共 {len(all_results)} 个 trial。")


if __name__ == "__main__":
    main()
