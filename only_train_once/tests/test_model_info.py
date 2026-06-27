"""
模型信息提取测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_model_info.py -v -s
"""

import sys
import os
import json
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.deployment.model_info import (
    ModelInfo,
    LayerStructure,
    QuantizationInfo,
    extract_model_info,
    get_kernel_config_for_layer,
    export_kernel_config,
    _compute_bit_width,
)


# ============================================================================
# 测试模型
# ============================================================================

class SimpleQuantModel(nn.Module):
    """简单量化模型"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MockQuantConv2d(nn.Conv2d):
    """模拟量化Conv2d层"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加量化参数
        self.d_quant_wt = nn.Parameter(torch.tensor([0.1]))
        self.q_m_wt = nn.Parameter(torch.tensor([1.0]))
        self.t_quant_wt = nn.Parameter(torch.tensor([1.0]))
        self.d_quant_act = nn.Parameter(torch.tensor([0.2]))
        self.q_m_act = nn.Parameter(torch.tensor([1.0]))
        self.t_quant_act = nn.Parameter(torch.tensor([1.0]))
        self.quant_type = type('QuantType', (), {'value': 'symmetric+nonlinear'})()
        self.quant_mode = type('QuantMode', (), {'value': 'weight_and_activation'})()


class MockQuantModel(nn.Module):
    """模拟量化模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = MockQuantConv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = MockQuantConv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# 测试类
# ============================================================================

class TestComputeBitWidth:
    """测试位宽计算"""

    def test_basic_cases(self):
        """测试基本案例"""
        # d=0.1, q_m=1.0, t=1.0 -> bit ≈ 8
        bit = _compute_bit_width(0.1, 1.0, 1.0)
        assert 4 <= bit <= 16
        print(f"  ✓ d=0.1, q_m=1.0, t=1.0 -> bit={bit}")

        # d=0.01, q_m=1.0, t=1.0 -> bit ≈ 11
        bit = _compute_bit_width(0.01, 1.0, 1.0)
        assert 8 <= bit <= 16
        print(f"  ✓ d=0.01, q_m=1.0, t=1.0 -> bit={bit}")

    def test_edge_cases(self):
        """测试边界案例"""
        # d=0 -> 32
        bit = _compute_bit_width(0.0, 1.0, 1.0)
        assert bit == 32
        print(f"  ✓ d=0 -> bit={bit}")

        # q_m=0 -> 32
        bit = _compute_bit_width(0.1, 0.0, 1.0)
        assert bit == 32
        print(f"  ✓ q_m=0 -> bit={bit}")

        # 极小d -> 大位宽
        bit = _compute_bit_width(1e-10, 1.0, 1.0)
        assert bit >= 16
        print(f"  ✓ d=1e-10 -> bit={bit}")

        # 极大d -> 小位宽
        bit = _compute_bit_width(10.0, 1.0, 1.0)
        assert bit <= 4
        print(f"  ✓ d=10.0 -> bit={bit}")


class TestExtractModelInfo:
    """测试模型信息提取"""

    def test_simple_model(self):
        """测试简单模型"""
        model = SimpleQuantModel()
        info = extract_model_info(model, 'test_model')

        assert info.model_name == 'test_model'
        assert info.total_params > 0
        assert len(info.layers) > 0
        print(f"  ✓ 简单模型: {len(info.layers)} 层, {info.total_params} 参数")

    def test_quantized_model(self):
        """测试量化模型"""
        model = MockQuantModel()
        info = extract_model_info(model, 'quant_model')

        # 检查量化层
        quant_layers = [name for name, layer in info.layers.items() if layer.is_quantized]
        assert len(quant_layers) == 2  # conv1, conv2
        print(f"  ✓ 量化模型: {len(quant_layers)} 量化层")

        # 检查位宽
        for name in quant_layers:
            layer = info.layers[name]
            print(f"    {name}: weight_bit={layer.quant_info.weight_bit}, "
                  f"act_bit={layer.quant_info.activation_bit}")

    def test_layer_types(self):
        """测试不同层类型"""
        model = SimpleQuantModel()
        info = extract_model_info(model)

        # 检查层类型
        layer_types = set(layer.type for layer in info.layers.values())
        assert 'Conv2d' in layer_types
        assert 'Linear' in layer_types
        assert 'BatchNorm2d' in layer_types
        print(f"  ✓ 层类型: {layer_types}")

    def test_quantization_info(self):
        """测试量化信息提取"""
        model = MockQuantModel()
        info = extract_model_info(model)

        conv1 = info.layers['conv1']
        assert conv1.is_quantized
        assert abs(conv1.quant_info.d_quant_wt - 0.1) < 0.01  # 浮点精度
        assert abs(conv1.quant_info.q_m_wt - 1.0) < 0.01
        assert abs(conv1.quant_info.t_quant_wt - 1.0) < 0.01
        print(f"  ✓ 量化参数: d={conv1.quant_info.d_quant_wt:.4f}, "
              f"q_m={conv1.quant_info.q_m_wt:.4f}, t={conv1.quant_info.t_quant_wt:.4f}")


class TestGetKernelConfig:
    """测试核函数配置生成"""

    def test_conv2d_config(self):
        """测试Conv2d配置"""
        layer = LayerStructure(
            name='conv1',
            type='Conv2d',
            original_in_channels=3,
            original_out_channels=16,
            pruned_in_channels=3,
            pruned_out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            is_quantized=True,
            quant_info=QuantizationInfo(
                d_quant_wt=0.1,
                q_m_wt=1.0,
                t_quant_wt=1.0,
                weight_bit=8,
                activation_bit=8,
            )
        )

        config = get_kernel_config_for_layer(layer)

        assert config['type'] == 'Conv2d'
        assert config['in_channels'] == 3
        assert config['out_channels'] == 16
        assert config['kernel_size'] == 3
        assert config['is_quantized'] is True
        assert config['quantization']['weight_bit'] == 8
        print(f"  ✓ Conv2d配置: {config['in_channels']}x{config['out_channels']}, "
              f"kernel={config['kernel_size']}, bit={config['quantization']['weight_bit']}")

    def test_linear_config(self):
        """测试Linear配置"""
        layer = LayerStructure(
            name='fc1',
            type='Linear',
            original_in_channels=128,
            original_out_channels=64,
            pruned_in_channels=128,
            pruned_out_channels=64,
            is_quantized=True,
            quant_info=QuantizationInfo(
                d_quant_wt=0.05,
                q_m_wt=1.0,
                t_quant_wt=1.0,
                weight_bit=4,
                activation_bit=8,
            )
        )

        config = get_kernel_config_for_layer(layer)

        assert config['type'] == 'Linear'
        assert config['in_channels'] == 128
        assert config['out_channels'] == 64
        assert config['quantization']['weight_bit'] == 4
        print(f"  ✓ Linear配置: {config['in_channels']}x{config['out_channels']}, "
              f"bit={config['quantization']['weight_bit']}")

    def test_mixed_precision_config(self):
        """测试混合精度配置"""
        model = MockQuantModel()
        info = extract_model_info(model)

        # 获取所有层的配置
        configs = []
        for name, layer in info.layers.items():
            if layer.type in ['Conv2d', 'Linear']:
                config = get_kernel_config_for_layer(layer)
                configs.append(config)

        # 检查不同位宽
        bits = set()
        for config in configs:
            if 'quantization' in config:
                bits.add(config['quantization']['weight_bit'])

        print(f"  ✓ 混合精度: 位宽分布={bits}")


class TestExportKernelConfig:
    """测试核函数配置导出"""

    def test_export_json(self):
        """测试导出JSON"""
        model = MockQuantModel()
        info = extract_model_info(model, 'test_model')

        output_path = '/tmp/test_kernel_config.json'
        export_kernel_config(info, output_path)

        assert os.path.exists(output_path)

        with open(output_path) as f:
            config = json.load(f)

        assert 'model_name' in config
        assert 'layers' in config
        assert 'bit_width_distribution' in config
        print(f"  ✓ 导出JSON: {len(config['layers'])} 层")
        print(f"    位宽分布: {config['bit_width_distribution']}")


class TestModelInfoSerialization:
    """测试模型信息序列化"""

    def test_to_dict(self):
        """测试转换为字典"""
        model = MockQuantModel()
        info = extract_model_info(model, 'test_model')

        data = info.to_dict()

        assert 'model_name' in data
        assert 'layers' in data
        assert 'bit_width_distribution' in data
        print(f"  ✓ to_dict: {len(data['layers'])} 层")

    def test_to_json(self):
        """测试导出JSON"""
        model = MockQuantModel()
        info = extract_model_info(model, 'test_model')

        output_path = '/tmp/test_model_info.json'
        info.to_json(output_path)

        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data['model_name'] == 'test_model'
        print(f"  ✓ to_json: {data['model_name']}")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_extraction_pipeline(self):
        """测试完整提取流程"""
        # 创建模型
        model = MockQuantModel()

        # 提取信息
        info = extract_model_info(model, 'my_model')

        # 打印摘要
        print("\n" + "=" * 60)
        print(f"模型: {info.model_name}")
        print(f"总参数: {info.total_params:,}")
        print(f"层数: {len(info.layers)}")
        print(f"混合精度: {info.has_mixed_precision}")
        print(f"剪枝: {info.has_pruning}")
        print(f"位宽范围: [{info.min_bit_width}, {info.max_bit_width}]")
        print(f"位宽分布: {info.bit_width_distribution}")
        print("=" * 60)

        print("\n层详情:")
        for name, layer in info.layers.items():
            if layer.type in ['Conv2d', 'Linear']:
                bit_info = ""
                if layer.is_quantized:
                    bit_info = f" [wt:{layer.quant_info.weight_bit}-bit, act:{layer.quant_info.activation_bit}-bit]"
                print(f"  {name:20s} | {layer.type:10s} | "
                      f"{layer.pruned_in_channels:4d} -> {layer.pruned_out_channels:4d}{bit_info}")

        # 导出
        output_dir = '/tmp/test_model_info_export'
        os.makedirs(output_dir, exist_ok=True)

        info.to_json(os.path.join(output_dir, 'model_info.json'))
        export_kernel_config(info, os.path.join(output_dir, 'kernel_config.json'))

        print(f"\n导出文件:")
        print(f"  - {output_dir}/model_info.json")
        print(f"  - {output_dir}/kernel_config.json")

        print("\n  ✓ 完整提取流程成功")


if __name__ == "__main__":
    print("=" * 70)
    print("模型信息提取测试")
    print("=" * 70)

    test_classes = [
        TestComputeBitWidth,
        TestExtractModelInfo,
        TestGetKernelConfig,
        TestExportKernelConfig,
        TestModelInfoSerialization,
        TestIntegration,
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
