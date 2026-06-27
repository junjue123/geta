from .magnitude import *
from .cosine_similarity import *
from .taylor import *
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# --- 全局变量用于存储绘图历史 ---
# 结构: {'layer_name': {'magnitude': [v1, v2...], 'steps': [1, 2...]}}
_SCORE_HISTORY = {}

# --- 全局变量用于存储重要性分数的全局观测边界 ---
# 结构: {'criterion_name': [global_min, global_max]}
# 用于 _apply_online_normalization() 的滑动最值归一化
_GLOBAL_SCORE_BOUNDS = {}


def _get_layer_name(param_group):
    """辅助函数：统一获取层名称"""
    layer_name = "unknown"
    if 'p_names' in param_group and len(param_group['p_names']) > 0:
        # 简化名称: module.layer.weight -> module_layer
        layer_name = param_group['p_names'][0].replace(".", "_").replace("_weight", "")
    return layer_name


def _record_and_plot(param_group, step=None):
    """
    辅助函数：记录分数到 log 并绘图到 temp 文件夹
    注意：此函数记录的是【原始/Raw】分数，用于后续计算方差。
    """
    # 1. 准备目录
    save_dir = "temp"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "scores_log_raw.txt")

    # 2. 获取层名称
    layer_name = _get_layer_name(param_group)

    # 3. 处理 Step 计数
    if step is None:
        if not hasattr(_record_and_plot, "internal_step"):
            _record_and_plot.internal_step = 0
        _record_and_plot.internal_step += 1
        current_step = _record_and_plot.internal_step
    else:
        current_step = step

    # 4. 初始化历史记录
    if layer_name not in _SCORE_HISTORY:
        _SCORE_HISTORY[layer_name] = {'steps': []}

    # 防止同一个step重复记录 (幂等性检查)
    if current_step in _SCORE_HISTORY[layer_name]['steps']:
        return

    _SCORE_HISTORY[layer_name]['steps'].append(current_step)

    # 5. 提取数据并写入日志
    try:
        with open(log_file, "a") as f:
            log_header = f"Step: {current_step} | Layer: {layer_name}"
            f.write(f"{log_header}\n")

            if 'importance_scores' in param_group:
                scores_dict = param_group['importance_scores']
                for crit_name, score_val in scores_dict.items():
                    # 确保 crit_name 在历史记录中
                    if crit_name not in _SCORE_HISTORY[layer_name]:
                        _SCORE_HISTORY[layer_name][crit_name] = []

                    # 计算均值用于记录和绘图
                    if isinstance(score_val, torch.Tensor):
                        val_scalar = score_val.float().mean().item()
                    else:
                        val_scalar = float(score_val)

                    # 存入历史
                    _SCORE_HISTORY[layer_name][crit_name].append(val_scalar)

                    f.write(f"  - {crit_name}: {val_scalar:.6e}\n")
            f.write("-" * 30 + "\n")
    except Exception as e:
        print(f"[Warning] Logging failed: {e}")

    # 6. 绘图 (为了性能建议控制频率，此处保留原逻辑)
    try:
        # 简单优化：只在 steps 数量为 10 的倍数时绘图，避免频繁 I/O
        if len(_SCORE_HISTORY[layer_name]['steps']) % 10 == 0:
            plt.ioff()
            plt.figure(figsize=(10, 6))
            steps = _SCORE_HISTORY[layer_name]['steps']

            has_data = False
            for key, vals in _SCORE_HISTORY[layer_name].items():
                if key == 'steps': continue
                # 确保长度对齐
                if len(vals) == len(steps):
                    plt.plot(steps, vals, label=key, marker='.')
                    has_data = True

            if has_data:
                plt.title(f"Raw Importance Scores - {layer_name}")
                plt.xlabel("Step")
                plt.ylabel("Score Value (Mean)")
                plt.yscale('symlog')
                plt.legend()
                plt.grid(True, which="both", ls="-", alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{layer_name}_raw_scores.png"))

            plt.close()
    except Exception as e:
        print(f"[Warning] Plotting failed: {e}")


def _apply_online_normalization(param_group, momentum=0.9):
    """
    使用滑动最值对当前层的分数进行在线归一化。
    momentum: 如果你想让最值更偏向于近期观测到的范围，可以引入动量；
    此处默认使用全局绝对最值以保证稳定性。
    """
    global _GLOBAL_SCORE_BOUNDS
    eps = 1e-8

    scores_dict = param_group.get('importance_scores', {})

    for cri_name, score in scores_dict.items():
        # 获取当前张量的局部最值
        if isinstance(score, torch.Tensor):
            curr_min = score.min().item()
            curr_max = score.max().item()
        else:
            curr_min = curr_max = float(score)

        # 更新全局观测边界
        if cri_name not in _GLOBAL_SCORE_BOUNDS:
            _GLOBAL_SCORE_BOUNDS[cri_name] = [curr_min, curr_max]
        else:
            _GLOBAL_SCORE_BOUNDS[cri_name][0] = min(_GLOBAL_SCORE_BOUNDS[cri_name][0], curr_min)
            _GLOBAL_SCORE_BOUNDS[cri_name][1] = max(_GLOBAL_SCORE_BOUNDS[cri_name][1], curr_max)

        # 执行归一化
        g_min, g_max = _GLOBAL_SCORE_BOUNDS[cri_name]
        denom = g_max - g_min + eps

        # 直接修改 param_group 中的 tensor
        param_group['importance_scores'][cri_name] = (score - g_min) / denom


def adjust_importance_criteria(param_group, bit_layers, full_precision=32, history_window=5, smooth_factor = 0.8):
    """
    [DACO - MCSS]
    根据 1. 历史方差波动 (稳定性 Ψ) 和 2. 当前量化位宽 (保真系数 Φ(d))
    对重要性分数进行加权调整，直接修改 param_group['importance_scores']。
    
    公式: adjusted = raw * (1 - smooth) + raw * Φ(d) * Ψ * smooth
    默认 smooth_factor=0.8 使校准因子起主导作用，对齐论文 MCSS 公式。
    """
    if 'importance_scores' not in param_group or not param_group['importance_scores']:
        return

    layer_name = _get_layer_name(param_group)

    # --- 1. 获取位宽因子 (Bit-width Factor) ---
    # 逻辑：位宽越小，越需要保护 (Score 变大)；位宽越大 (接近32)，相对可以被剪枝 (Score 相对变小)
    bit_factor = 1.0
    if bit_layers is not None:
        # 尝试匹配 param_group 中的参数名到 bit_layers
        target_layer_key = None
        for p_name in param_group['p_names']:
            # 假设 bit_layers 键值是 "module.layer" 形式
            candidate = ".".join(p_name.split(".")[:-1])
            if candidate in bit_layers:
                target_layer_key = candidate
                break

        if target_layer_key:
            # 获取权重位宽，默认为 32
            current_bit = bit_layers[target_layer_key].get('weight', full_precision)
            current_bit = max(current_bit, 0.1)  # 防止除零
            # 示例: 4bit -> factor = 32/4 = 8.0 (放大分数，保护)
            bit_factor = 1 + math.log2(full_precision/current_bit)

    # --- 2. 遍历所有评分标准进行调整 ---
    for cri_name in param_group['importance_scores']:
        raw_score = param_group['importance_scores'][cri_name]

        # --- 计算方差因子 (Variance/Stability Factor) ---
        stability_factor = 1.0
        # 从全局历史中获取最近 N 步的数据来计算波动
        if layer_name in _SCORE_HISTORY and cri_name in _SCORE_HISTORY[layer_name]:
            history_vals = _SCORE_HISTORY[layer_name][cri_name]
            if len(history_vals) >= 2:
                # 取最近 window_size 个点
                window = history_vals[-history_window:]
                arr = np.array(window)

                # 计算变异系数 (Coefficient of Variation) = std / mean
                # 使用 CV 而不是单纯的 var，是为了消除不同 score 量级的影响
                mean_val = np.abs(np.mean(arr))
                if mean_val > 1e-9:
                    cv = np.std(arr) / mean_val
                    # 策略: 波动越大 (CV越大)，因子越小。波动为0，因子为1。
                    # 使用 exp(-CV) 进行平滑衰减
                    stability_factor = max(0.2, math.exp(-cv))
                else:
                    stability_factor = 1.0

        # --- 3. 综合调整 ---
        # Final Score = Raw * Stability * BitProtection
        # 1. 越稳定，分数越高 (奖励稳定性)
        # 2. 位宽越低，分数越高 (保护低位宽)

        # 注意: raw_score 可能是 Tensor，保持 Tensor 操作
        adjusted_score = raw_score*(1-smooth_factor) + raw_score * stability_factor * bit_factor * smooth_factor

        # 更新回 param_group
        param_group['importance_scores'][cri_name] = adjusted_score

        # (可选) 打印调试信息，仅在 verbose 时开启
        # if stability_factor < 0.8 or bit_factor > 2.0:
        #     print(f"[{layer_name}][{cri_name}] Raw: {raw_score.mean():.2e} | "
        #           f"CV_Factor: {stability_factor:.2f} | Bit_Factor: {bit_factor:.1f} -> Adj: {adjusted_score.mean():.2e}")


def calculate_importance_score(criteria, param_group, bit_layers=None, step=None):
    """
    Args:
        step (int, optional): 当前训练步数
    """
    param_group['importance_scores'] = dict()
    with torch.no_grad():
        # 1. 计算原始重要性分数
        for cri_name in criteria:
            if 'magnitude' == cri_name:
                importance_score_by_magnitude(param_group)
            elif 'avg_magnitude' == cri_name:
                importance_score_by_avg_magnitude(param_group)
            elif 'cosine_similarity' == cri_name:
                importance_score_by_cosine_similarity(param_group)
            elif 'taylor_first_order' == cri_name:
                importance_score_by_first_order_taylor(param_group)
            elif 'taylor_second_order' == cri_name:
                importance_score_by_second_order_taylor(param_group)

    # 2. 记录原始分数到历史 (用于计算方差) 并绘图
    # 注意：我们先记录 Raw Score，这样 adjust 函数才能读到当前步的数据来计算方差
    _record_and_plot(param_group, step=step)
    # 归一化
    _apply_online_normalization(param_group)

    # 3. [新增] 根据方差和位宽调整分数
    # 这一步会修改 param_group['importance_scores'] 的值供优化器使用
    adjust_importance_criteria(param_group, bit_layers)


def calculate_importance_score_lora(criteria, param_group, global_params, bit_layers=None, step=None):
    """
    Args:
        step (int, optional): 当前训练步数
        bit_layers: (新增参数) 传入位宽字典
    """
    param_group['importance_scores'] = dict()
    with torch.no_grad():
        # 1. 计算原始重要性分数
        for cri_name in criteria:
            if 'magnitude' in cri_name:
                importance_score_by_magnitude_lora(param_group)
            elif 'avg_magnitude' == cri_name:
                importance_score_by_avg_magnitude_lora(param_group)
            elif 'cosine_similarity' in cri_name:
                importance_score_by_cosine_similarity_lora(param_group, global_params)
            elif 'taylor_first_order' in cri_name:
                importance_score_by_first_order_taylor_lora(param_group, global_params)
            elif 'taylor_second_order' in cri_name:
                importance_score_by_second_order_taylor_lora(param_group, global_params)

    # 2. 记录原始分数到历史并绘图
    _record_and_plot(param_group, step=step)

    # 3. [新增] 根据方差和位宽调整分数
    adjust_importance_criteria(param_group, bit_layers)