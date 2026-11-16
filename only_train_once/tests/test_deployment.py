"""
部署模块测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_deployment.py -v -s
"""

import sys
import os
import json
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.deployment import ModelExporter, DeploymentConfig, KernelSelector
from only_train_once.deployment.exporter import (
    DeploymentTarget, QuantizationType, LayerInfo, QuantizationParams
)
from only_train_once.deployment.kernel_selector import (
    HardwareTarget, KernelType, KernelConfig
)


# ============================================================================
# 测试模型
# ============================================================================

class SimpleCNN(nn.Module):
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
# ModelExporter 测试
# ============================================================================

class TestModelExporter:
    """测试模型导出器"""

    def test_initialization(self):
        """测试初始化"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        assert len(exporter.layer_info) > 0
        print(f"  ✓ 初始化: {len(exporter.layer_info)} 层")

    def test_analyze_model(self):
        """测试模型分析"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        # 检查Conv2d层
        conv_layers = [name for name, info in exporter.layer_info.items()
                       if info.type == 'Conv2d']
        assert len(conv_layers) == 2
        print(f"  ✓ Conv2d层: {conv_layers}")

        # 检查Linear层
        linear_layers = [name for name, info in exporter.layer_info.items()
                         if info.type == 'Linear']
        assert len(linear_layers) == 2
        print(f"  ✓ Linear层: {linear_layers}")

    def test_set_layer_bit_width(self):
        """测试设置层位宽"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        exporter.set_layer_bit_width('features.0', 4)
        assert exporter.layer_info['features.0'].bit_width == 4
        print("  ✓ 设置位宽正确")

    def test_set_quantization_params(self):
        """测试设置量化参数"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        exporter.set_quantization_params(
            'features.0',
            scale=0.1,
            zero_point=0,
            min_val=-1.0,
            max_val=1.0,
            num_bits=8
        )

        assert 'features.0' in exporter.quant_params
        assert exporter.quant_params['features.0'].scale == 0.1
        print("  ✓ 设置量化参数正确")

    def test_export_pytorch(self):
        """测试PyTorch导出"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        config = DeploymentConfig(
            target=DeploymentTarget.PYTORCH,
            input_shape=[3, 8, 8]
        )
        exporter.config = config

        with pytest.MonkeyPatch.context() as m:
            # 临时修改输出目录
            output_dir = '/tmp/test_export'
            os.makedirs(output_dir, exist_ok=True)

            files = exporter.export(output_dir, 'test_model')

            assert 'model' in files
            assert 'state_dict' in files
            assert os.path.exists(files['model'])
            assert os.path.exists(files['state_dict'])

            print(f"  ✓ PyTorch导出: {list(files.keys())}")

    def test_export_metadata(self):
        """测试元数据导出"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        output_dir = '/tmp/test_export'
        os.makedirs(output_dir, exist_ok=True)

        meta_path = os.path.join(output_dir, 'test_meta.json')
        exporter._export_metadata(meta_path)

        assert os.path.exists(meta_path)

        with open(meta_path) as f:
            meta = json.load(f)

        assert 'layer_info' in meta
        assert 'total_params' in meta
        print(f"  ✓ 元数据导出: {meta['total_params']} 参数")

    def test_export_for_kernel(self):
        """测试核函数配置导出"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        # 设置一些量化参数
        exporter.set_quantization_params(
            'features.0',
            scale=0.1,
            zero_point=0,
            min_val=-1.0,
            max_val=1.0,
            num_bits=8
        )

        output_dir = '/tmp/test_export'
        os.makedirs(output_dir, exist_ok=True)

        files = exporter.export_for_kernel(output_dir, 'test_model')

        assert 'kernel_config' in files
        assert os.path.exists(files['kernel_config'])

        with open(files['kernel_config']) as f:
            config = json.load(f)

        assert 'layers' in config
        assert 'global_config' in config
        print(f"  ✓ 核函数配置: {len(config['layers'])} 层")

    def test_get_layer_summary(self):
        """测试层摘要"""
        model = SimpleCNN()
        exporter = ModelExporter(model)

        summary = exporter.get_layer_summary()
        assert 'Conv2d' in summary
        assert 'Linear' in summary
        print(f"  ✓ 层摘要:\n{summary[:200]}...")


# ============================================================================
# KernelSelector 测试
# ============================================================================

class TestKernelSelector:
    """测试核函数选择器"""

    def test_initialization(self):
        """测试初始化"""
        selector = KernelSelector(hardware=HardwareTarget.NVIDIA_GPU)
        assert selector.hardware == HardwareTarget.NVIDIA_GPU
        print("  ✓ 初始化正确")

    def test_select_kernel_conv2d(self):
        """测试Conv2d核函数选择"""
        selector = KernelSelector()

        # INT8卷积
        config = selector.select_kernel(
            layer_type='Conv2d',
            in_channels=16,
            out_channels=32,
            bit_width=8
        )
        assert config.kernel_type == KernelType.CONV2D_INT8
        print(f"  ✓ INT8 Conv2d: {config.kernel_type.value}")

        # FP16卷积
        config = selector.select_kernel(
            layer_type='Conv2d',
            in_channels=16,
            out_channels=32,
            bit_width=16
        )
        assert config.kernel_type == KernelType.CONV2D_FP16
        print(f"  ✓ FP16 Conv2d: {config.kernel_type.value}")

    def test_select_kernel_linear(self):
        """测试Linear核函数选择"""
        selector = KernelSelector()

        config = selector.select_kernel(
            layer_type='Linear',
            in_channels=128,
            out_channels=64,
            bit_width=8
        )
        assert config.kernel_type == KernelType.LINEAR_INT8
        print(f"  ✓ INT8 Linear: {config.kernel_type.value}")

    def test_select_kernel_depthwise(self):
        """测试深度可分离卷积核函数选择"""
        selector = KernelSelector()

        config = selector.select_kernel(
            layer_type='Conv2d',
            in_channels=32,
            out_channels=32,
            bit_width=8,
            groups=32
        )
        assert config.kernel_type == KernelType.DEPTHWISE_INT8
        print(f"  ✓ INT8 Depthwise: {config.kernel_type.value}")

    def test_select_kernel_sparse(self):
        """测试稀疏核函数选择"""
        selector = KernelSelector(sparse_threshold=0.5)

        config = selector.select_kernel(
            layer_type='Conv2d',
            in_channels=16,
            out_channels=32,
            bit_width=8,
            sparsity_ratio=0.7
        )
        assert config.kernel_type == KernelType.SPARSE_CONV2D
        assert config.is_sparse is True
        print(f"  ✓ Sparse Conv2d: {config.kernel_type.value}")

    def test_select_kernel_int4(self):
        """测试INT4核函数选择"""
        selector = KernelSelector()

        config = selector.select_kernel(
            layer_type='Linear',
            in_channels=256,
            out_channels=128,
            bit_width=4
        )
        assert config.kernel_type == KernelType.LINEAR_INT4
        print(f"  ✓ INT4 Linear: {config.kernel_type.value}")

    def test_get_kernel_recommendations(self):
        """测试获取核函数推荐"""
        selector = KernelSelector()

        layer_configs = [
            {'name': 'conv1', 'type': 'Conv2d', 'in_channels': 3, 'out_channels': 16, 'bit_width': 8},
            {'name': 'conv2', 'type': 'Conv2d', 'in_channels': 16, 'out_channels': 32, 'bit_width': 4},
            {'name': 'fc1', 'type': 'Linear', 'in_channels': 128, 'out_channels': 64, 'bit_width': 8},
        ]

        recommendations = selector.get_kernel_recommendations(layer_configs)

        assert len(recommendations) == 3
        assert recommendations[0]['kernel_type'] == 'conv2d_int8'
        assert recommendations[1]['kernel_type'] == 'conv2d_int4'
        assert recommendations[2]['kernel_type'] == 'linear_int8'
        print(f"  ✓ 推荐: {[r['kernel_type'] for r in recommendations]}")

    def test_generate_kernel_code_hint(self):
        """测试生成核函数代码提示"""
        selector = KernelSelector()

        config = KernelConfig(
            kernel_type=KernelType.CONV2D_INT8,
            in_channels=16,
            out_channels=32,
            bit_width=8
        )

        hint = selector.generate_kernel_code_hint(config)
        assert 'INT8' in hint
        print(f"  ✓ 代码提示:\n{hint}")

    def test_estimate_speedup(self):
        """测试估计加速比"""
        selector = KernelSelector()

        baseline = KernelConfig(
            kernel_type=KernelType.CONV2D_FP32,
            in_channels=16,
            out_channels=32,
            bit_width=32
        )

        int8_config = KernelConfig(
            kernel_type=KernelType.CONV2D_INT8,
            in_channels=16,
            out_channels=32,
            bit_width=8
        )

        speedup = selector.estimate_speedup(int8_config, baseline)
        assert speedup > 1.0
        print(f"  ✓ INT8 vs FP32 加速比: {speedup:.1f}x")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_export_pipeline(self):
        """测试完整导出流程"""
        # 创建模型
        model = SimpleCNN()

        # 创建导出器
        config = DeploymentConfig(
            target=DeploymentTarget.PYTORCH,
            export_float16=False,
            input_shape=[3, 8, 8]
        )
        exporter = ModelExporter(model, config)

        # 设置量化参数
        for name, info in exporter.layer_info.items():
            if info.type in ['Conv2d', 'Linear']:
                exporter.set_quantization_params(
                    name,
                    scale=0.1,
                    zero_point=0,
                    min_val=-1.0,
                    max_val=1.0,
                    num_bits=8
                )
                exporter.set_layer_bit_width(name, 8)

        # 导出
        output_dir = '/tmp/test_full_export'
        os.makedirs(output_dir, exist_ok=True)

        files = exporter.export(output_dir, 'test_model')
        kernel_files = exporter.export_for_kernel(output_dir, 'test_model')

        # 验证
        assert 'model' in files
        assert 'metadata' in files
        assert 'kernel_config' in kernel_files

        # 打印摘要
        summary = exporter.get_layer_summary()
        print(f"\n{summary}")

        print(f"\n  ✓ 完整导出流程成功")
        print(f"    导出文件: {list(files.keys())}")
        print(f"    核函数配置: {list(kernel_files.keys())}")

    def test_kernel_selection_pipeline(self):
        """测试核函数选择流程"""
        # 创建模型
        model = SimpleCNN()
        exporter = ModelExporter(model)

        # 创建核函数选择器
        selector = KernelSelector(hardware=HardwareTarget.NVIDIA_GPU)

        # 为每层选择核函数
        layer_configs = []
        for name, info in exporter.layer_info.items():
            if info.type in ['Conv2d', 'Linear']:
                layer_configs.append({
                    'name': name,
                    'type': info.type,
                    'in_channels': info.in_channels,
                    'out_channels': info.out_channels,
                    'bit_width': 8,
                    'groups': info.groups if info.type == 'Conv2d' else 1,
                    'kernel_size': info.kernel_size,
                })

        recommendations = selector.get_kernel_recommendations(layer_configs)

        print("\n  核函数推荐:")
        for rec in recommendations:
            print(f"    {rec['layer_name']:20s} -> {rec['kernel_type']}")

        print(f"\n  ✓ 核函数选择流程成功")


if __name__ == "__main__":
    print("=" * 70)
    print("部署模块测试")
    print("=" * 70)

    test_classes = [
        TestModelExporter,
        TestKernelSelector,
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
