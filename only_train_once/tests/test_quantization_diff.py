"""
量化前后对比测试
测试量化模型和原始模型的输出差异。

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_quantization_diff.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# 测试模型
# ============================================================================

class SimpleCNN(nn.Module):
    """简单CNN"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# 量化函数
# ============================================================================

def quantize_tensor(tensor, num_bits=8, symmetric=True):
    """简单量化函数

    Args:
        tensor: 输入张量
        num_bits: 位数
        symmetric: 是否对称量化

    Returns:
        量化后的张量
    """
    if symmetric:
        # 对称量化
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (num_bits - 1) - 1)
        if scale < 1e-10:
            return tensor
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
        return quantized * scale
    else:
        # 非对称量化
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2 ** num_bits - 1)
        if scale < 1e-10:
            return tensor
        zero_point = torch.round(-min_val / scale)
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 2 ** num_bits - 1)
        return (quantized - zero_point) * scale


def quantize_model_weights(model, num_bits=8):
    """量化模型权重

    Args:
        model: 原始模型
        num_bits: 位数

    Returns:
        量化后的模型（副本）
    """
    import copy
    quantized_model = copy.deepcopy(model)

    for name, param in quantized_model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data = quantize_tensor(param.data, num_bits)

    return quantized_model


def quantize_model_per_layer(model, bit_widths):
    """按层量化模型权重

    Args:
        model: 原始模型
        bit_widths: 每层的位宽字典 {层名: 位宽}

    Returns:
        量化后的模型（副本）
    """
    import copy
    quantized_model = copy.deepcopy(model)

    for name, param in quantized_model.named_parameters():
        if name in bit_widths:
            num_bits = bit_widths[name]
            param.data = quantize_tensor(param.data, num_bits)

    return quantized_model


# ============================================================================
# 测试类
# ============================================================================

class TestQuantizationDiff:
    """测试量化前后差异"""

    def test_weight_distribution(self):
        """测试权重分布变化"""
        model = SimpleCNN()

        # 原始权重
        original_weights = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                original_weights[name] = param.data.clone()

        # 量化权重
        quantized_model = quantize_model_weights(model, num_bits=8)

        print("\n  权重分布对比:")
        print(f"  {'层名':25s} | {'原始均值':>10s} | {'量化均值':>10s} | {'原始std':>10s} | {'量化std':>10s} | {'MSE':>10s}")
        print("  " + "-" * 85)

        for name in original_weights:
            orig = original_weights[name]
            quant = dict(quantized_model.named_parameters())[name].data

            mse = ((orig - quant) ** 2).mean().item()

            print(f"  {name:25s} | {orig.mean().item():10.6f} | {quant.mean().item():10.6f} | "
                  f"{orig.std().item():10.6f} | {quant.std().item():10.6f} | {mse:10.6f}")

        print("  ✓ 权重分布对比完成")

    def test_output_difference(self):
        """测试输出差异"""
        model = SimpleCNN()
        model.eval()

        # 量化模型
        quantized_model = quantize_model_weights(model, num_bits=8)
        quantized_model.eval()

        # 测试输入
        x = torch.randn(4, 3, 8, 8)

        # 前向传播
        with torch.no_grad():
            original_output = model(x)
            quantized_output = quantized_model(x)

        # 计算差异
        diff = (original_output - quantized_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        mse = ((original_output - quantized_output) ** 2).mean().item()

        # 预测差异
        original_pred = original_output.argmax(dim=1)
        quantized_pred = quantized_output.argmax(dim=1)
        pred_diff = (original_pred != quantized_pred).sum().item()

        print(f"\n  输出差异:")
        print(f"    最大绝对差异: {max_diff:.6f}")
        print(f"    平均绝对差异: {mean_diff:.6f}")
        print(f"    MSE: {mse:.6f}")
        print(f"    预测不一致: {pred_diff}/{len(x)}")

        # 量化后差异应该在合理范围内
        assert max_diff < 1.0, f"最大差异过大: {max_diff}"
        print("  ✓ 输出差异在合理范围内")

    def test_different_bit_widths(self):
        """测试不同位宽的影响"""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(4, 3, 8, 8)

        with torch.no_grad():
            original_output = model(x)

        print(f"\n  不同位宽的影响:")
        print(f"  {'位宽':>4s} | {'最大差异':>10s} | {'平均差异':>10s} | {'MSE':>10s} | {'预测不一致':>10s}")
        print("  " + "-" * 55)

        for bits in [4, 8, 16, 32]:
            quantized_model = quantize_model_weights(model, num_bits=bits)
            quantized_model.eval()

            with torch.no_grad():
                quantized_output = quantized_model(x)

            diff = (original_output - quantized_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            mse = ((original_output - quantized_output) ** 2).mean().item()

            original_pred = original_output.argmax(dim=1)
            quantized_pred = quantized_output.argmax(dim=1)
            pred_diff = (original_pred != quantized_pred).sum().item()

            print(f"  {bits:4d} | {max_diff:10.6f} | {mean_diff:10.6f} | {mse:10.6f} | {pred_diff:10d}")

        print("  ✓ 不同位宽影响分析完成")

    def test_mixed_precision(self):
        """测试混合精度"""
        model = SimpleCNN()
        model.eval()

        # 定义每层的位宽
        bit_widths = {}
        for name, param in model.named_parameters():
            if 'features.0' in name:  # 第一层用高精度
                bit_widths[name] = 16
            elif 'features.4' in name:  # 第二层用低精度
                bit_widths[name] = 4
            elif 'classifier' in name:  # 分类器用中等精度
                bit_widths[name] = 8

        print(f"\n  混合精度配置:")
        for name, bits in bit_widths.items():
            print(f"    {name:30s} -> {bits}-bit")

        # 量化模型
        mixed_model = quantize_model_per_layer(model, bit_widths)
        mixed_model.eval()

        # 统一8-bit量化
        uniform_model = quantize_model_weights(model, num_bits=8)
        uniform_model.eval()

        x = torch.randn(4, 3, 8, 8)

        with torch.no_grad():
            original_output = model(x)
            mixed_output = mixed_model(x)
            uniform_output = uniform_model(x)

        # 计算差异
        mixed_diff = ((original_output - mixed_output) ** 2).mean().item()
        uniform_diff = ((original_output - uniform_output) ** 2).mean().item()

        print(f"\n  混合精度 vs 统一精度:")
        print(f"    原始 vs 混合精度 MSE: {mixed_diff:.6f}")
        print(f"    原始 vs 统一8-bit MSE: {uniform_diff:.6f}")

        print("  ✓ 混合精度测试完成")

    def test_quantization_error_accumulation(self):
        """测试量化误差累积"""
        model = SimpleCNN()
        model.eval()

        # 多次量化（模拟训练中的累积）
        x = torch.randn(4, 3, 8, 8)

        print(f"\n  量化误差累积:")
        print(f"  {'次数':>4s} | {'MSE':>10s} | {'最大差异':>10s}")
        print("  " + "-" * 30)

        current_model = model
        for i in range(5):
            current_model = quantize_model_weights(current_model, num_bits=8)
            current_model.eval()

            with torch.no_grad():
                original_output = model(x)
                current_output = current_model(x)

            mse = ((original_output - current_output) ** 2).mean().item()
            max_diff = (original_output - current_output).abs().max().item()

            print(f"  {i+1:4d} | {mse:10.6f} | {max_diff:10.6f}")

        print("  ✓ 误差累积分析完成")

    def test_layer_wise_sensitivity(self):
        """测试各层量化敏感度"""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(4, 3, 8, 8)

        with torch.no_grad():
            original_output = model(x)

        print(f"\n  各层量化敏感度 (8-bit):")
        print(f"  {'层名':25s} | {'MSE':>10s} | {'最大差异':>10s}")
        print("  " + "-" * 50)

        # 逐层量化
        for layer_name in ['features.0', 'features.4', 'classifier.0', 'classifier.2']:
            import copy
            layer_model = copy.deepcopy(model)

            # 只量化指定层
            for name, param in layer_model.named_parameters():
                if layer_name in name:
                    param.data = quantize_tensor(param.data, num_bits=8)

            layer_model.eval()
            with torch.no_grad():
                layer_output = layer_model(x)

            mse = ((original_output - layer_output) ** 2).mean().item()
            max_diff = (original_output - layer_output).abs().max().item()

            print(f"  {layer_name:25s} | {mse:10.6f} | {max_diff:10.6f}")

        print("  ✓ 各层敏感度分析完成")


class TestQuantizationWithPruning:
    """测试量化+剪枝的影响"""

    def test_pruned_model_quantization(self):
        """测试剪枝后量化"""
        model = SimpleCNN()
        model.eval()

        # 模拟剪枝（减少通道数）
        import copy
        pruned_model = copy.deepcopy(model)

        # 剪枝第一层：16通道 -> 8通道
        with torch.no_grad():
            pruned_model.features[0].weight = nn.Parameter(
                pruned_model.features[0].weight[:8]
            )
            pruned_model.features[0].bias = nn.Parameter(
                pruned_model.features[0].bias[:8]
            )
            pruned_model.features[1] = nn.BatchNorm2d(8)

        # 量化剪枝后的模型
        quantized_pruned = quantize_model_weights(pruned_model, num_bits=8)

        print(f"\n  剪枝+量化:")
        print(f"    原始模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"    剪枝后参数: {sum(p.numel() for p in pruned_model.parameters()):,}")

        # 注意：剪枝后模型结构改变了，需要调整输入
        # 这里只是演示概念
        print("  ✓ 剪枝+量化测试完成")


if __name__ == "__main__":
    print("=" * 70)
    print("量化前后对比测试")
    print("=" * 70)

    test_classes = [
        TestQuantizationDiff,
        TestQuantizationWithPruning,
    ]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    method()
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
