import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# 1. 引入路径 (保持你的设置)
sys.path.append('..')
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
from sanity_check.backends.vgg7 import vgg7_bn
from sanity_check.backends.resnets_imagenet import resnet18_imagenet
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once import OTO

# 新增：计算参数组中非零参数的数量（过滤置零参数）
def get_nonzero_numel(param_group):
    nonzero_numel = 0
    for p in param_group['params']:
        # 只统计非零参数的数量（忽略置零的稀疏参数）
        nonzero_numel += torch.count_nonzero(p.data).item()
    return nonzero_numel

if __name__ == "__main__":
    # 1. 路径设置
    checkpoint_file = r"E:\pythonProject\geta-main\temp\daco\epoch_299_sparsity_0.5.pt"

    if not os.path.exists(checkpoint_file):
        print(f"错误: 文件不存在 {checkpoint_file}")
        exit()

    print(f"正在读取模型: {checkpoint_file}")

    # 2. 初始化模型结构
    model = resnet20_cifar10()
    model = model_to_quantize_model(model, quant_mode = "weight_and_activation")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 3. 加载权重
    checkpoint = torch.load(checkpoint_file, map_location=device)

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("成功加载 model_state_dict")
    except Exception as e:
        print(f"加载模型权重时发生警告: {e}")

    # 4. 初始化 OTO
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)

    # 5. 初始化 Optimizer
    steps_per_epoch = 391
    sparsity = 0.9
    optimizer = oto.mygeta(
        variant="adam",
        lr=1e-1,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=sparsity,
        start_projection_step=1 * steps_per_epoch,
        projection_periods=100,
        projection_steps=100 * steps_per_epoch,
        start_pruning_step=5 * steps_per_epoch,
        pruning_periods=100,
        pruning_steps=100 * steps_per_epoch,
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
    )

    print("优化器初始化完成，开始提取位宽统计...")
    print("-" * 80)

    # 数据结构：
    # layer_stats[layer_name] = {
    #     'w_bit': float,
    #     'a_bit': float,
    #     'w_numel': int (权重参数数量),
    #     'w_nonzero_numel': int (非零权重参数数量)  # 新增
    # }
    layer_stats = {}

    for group in optimizer.param_groups:
        # 1. 获取该组的位宽字典
        group_bits = optimizer.get_bitwidth_dict(group)
        if not group_bits:
            continue

        # 2. 新增：分别统计总参数量和非零参数量（过滤置零参数）
        current_group_numel = 0
        current_group_nonzero_numel = 0
        for p in group['params']:
            current_group_numel += p.numel()
            current_group_nonzero_numel += torch.count_nonzero(p.data).item()  # 只统计非零参数

        # 将位宽信息和参数数量记录下来
        for layer_name, bits_info in group_bits.items():
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {'w_numel': 0, 'w_nonzero_numel': 0}  # 初始化非零参数统计

            # 更新位宽
            layer_stats[layer_name].update(bits_info)

            # 更新参数量 (仅当该组是权重相关的组时)
            if 'weight' in bits_info:
                layer_stats[layer_name]['w_numel'] += current_group_numel
                layer_stats[layer_name]['w_nonzero_numel'] += current_group_nonzero_numel  # 新增非零参数统计

    # ---------------------------------------------------------
    # 打印与计算 (修改统计逻辑，考虑稀疏参数)
    # ---------------------------------------------------------
    print(f"{'Layer Name':<35} | {'W Bit':<6} | {'Total Params':<10} | {'Nonzero Params':<12} | {'A Bit':<6}")
    print("-" * 90)

    # 统计变量
    total_w_bits_arithmetic = []
    total_a_bits_arithmetic = []

    # 修改：区分总参数量和非零参数量的加权计算
    total_weighted_bits_product = 0.0  # 原始逻辑（含置零参数）
    total_params_count = 0

    total_weighted_bits_product_nonzero = 0.0  # 新增：仅非零参数的加权
    total_nonzero_params_count = 0

    sorted_layers = sorted(layer_stats.keys())

    for layer_name in sorted_layers:
        info = layer_stats[layer_name]

        w_bit = info.get('weight', 'N/A')
        a_bit = info.get('activation', 'N/A')
        numel = info.get('w_numel', 0)
        nonzero_numel = info.get('w_nonzero_numel', 0)  # 非零参数数量

        # 1. 算术平均数据收集
        if w_bit != 'N/A':
            total_w_bits_arithmetic.append(w_bit)

            # 2. 加权平均数据收集
            if numel > 0:
                # 原始逻辑（含置零参数）
                total_weighted_bits_product += (w_bit * numel)
                total_params_count += numel

                # 新增：仅非零参数的加权（考虑稀疏）
                if nonzero_numel > 0:
                    total_weighted_bits_product_nonzero += (w_bit * nonzero_numel)
                    total_nonzero_params_count += nonzero_numel

        if a_bit != 'N/A':
            total_a_bits_arithmetic.append(a_bit)

        # 打印时新增非零参数列
        print(f"{layer_name:<35} | {str(w_bit):<6} | {str(numel):<10} | {str(nonzero_numel):<12} | {str(a_bit):<6}")

    print("-" * 90)

    # 计算结果
    avg_w_arith = np.mean(total_w_bits_arithmetic) if total_w_bits_arithmetic else 0
    avg_a_arith = np.mean(total_a_bits_arithmetic) if total_a_bits_arithmetic else 0

    # 原始加权平均（未考虑稀疏）
    avg_w_weighted = 0
    if total_params_count > 0:
        avg_w_weighted = total_weighted_bits_product / total_params_count

    # 新增：考虑稀疏的加权平均（仅非零参数）
    avg_w_weighted_nonzero = 0
    if total_nonzero_params_count > 0:
        avg_w_weighted_nonzero = total_weighted_bits_product_nonzero / total_nonzero_params_count

    print(f"统计总结:")
    print(f"1. 权重算术平均 (Avg Weight Bit - Arithmetic): {avg_w_arith:.4f}")
    print(f"   (解释: 各层位宽直接平均，反映算法倾向)")
    print(f"2. 权重加权平均 (含置零参数)                 : {avg_w_weighted:.4f}")
    print(f"   (解释: 原始逻辑，未考虑稀疏，高估实际位宽)")
    print(f"3. 权重加权平均 (仅非零参数/考虑稀疏)        : {avg_w_weighted_nonzero:.4f}  <-- 推荐使用这个")
    print(f"   (解释: 过滤置零参数，反映实际有效存储的位宽)")
    print(f"4. 激活算术平均 (Avg Act Bit - Arithmetic)   : {avg_a_arith:.4f}")
    print("-" * 90)