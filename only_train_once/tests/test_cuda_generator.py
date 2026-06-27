"""
CUDA核函数生成器测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_cuda_generator.py -v -s
"""

import sys
import os
import json
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.deployment import (
    CUDAKernelGenerator,
    generate_kernels_from_config,
    extract_model_info,
    export_kernel_config,
)
from only_train_once.deployment.cuda_generator import KernelSpec


# ============================================================================
# 测试模型
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
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

class TestKernelSpec:
    """测试核函数规格"""

    def test_creation(self):
        """测试创建"""
        spec = KernelSpec(
            name='conv1',
            layer_type='Conv2d',
            in_channels=3,
            out_channels=16,
            weight_bit=8,
            activation_bit=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        assert spec.name == 'conv1'
        assert spec.layer_type == 'Conv2d'
        assert spec.weight_bit == 8
        print("  ✓ KernelSpec创建正确")


class TestCUDAKernelGenerator:
    """测试CUDA核函数生成器"""

    def test_initialization(self):
        """测试初始化"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda", prefix="test")
        assert generator.output_dir == "/tmp/test_cuda"
        assert generator.prefix == "test"
        print("  ✓ 初始化正确")

    def test_add_kernel(self):
        """测试添加核函数"""
        generator = CUDAKernelGenerator()

        spec = KernelSpec(
            name='conv1',
            layer_type='Conv2d',
            in_channels=3,
            out_channels=16,
            weight_bit=8,
            activation_bit=8,
            kernel_size=3,
        )

        generator.add_kernel(spec)
        assert len(generator.kernels) == 1
        print("  ✓ 添加核函数正确")

    def test_add_from_config(self):
        """测试从配置添加"""
        generator = CUDAKernelGenerator()

        config = {
            'layers': [
                {
                    'name': 'conv1',
                    'type': 'Conv2d',
                    'in_channels': 3,
                    'out_channels': 16,
                    'is_quantized': True,
                    'quantization': {'weight_bit': 8, 'activation_bit': 8},
                    'kernel_size': 3,
                },
                {
                    'name': 'fc1',
                    'type': 'Linear',
                    'in_channels': 128,
                    'out_channels': 64,
                    'is_quantized': True,
                    'quantization': {'weight_bit': 4, 'activation_bit': 8},
                },
                {
                    'name': 'relu1',
                    'type': 'ReLU',
                    'is_quantized': False,
                },
            ]
        }

        generator.add_from_config(config)
        assert len(generator.kernels) == 2  # 只添加量化的层
        print(f"  ✓ 从配置添加: {len(generator.kernels)} 个核函数")

    def test_generate_header(self):
        """测试生成头文件"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_header")

        spec = KernelSpec(
            name='conv1',
            layer_type='Conv2d',
            in_channels=3,
            out_channels=16,
            weight_bit=8,
            activation_bit=8,
            kernel_size=3,
        )
        generator.add_kernel(spec)

        path = generator._generate_header()
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        assert 'conv1' in content
        assert 'conv2d' in content  # 函数名使用小写
        print(f"  ✓ 头文件生成: {path}")

    def test_generate_quant_utils(self):
        """测试生成量化工具"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_quant")

        path = generator._generate_quant_utils()
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        assert 'quantize_int8' in content
        assert 'dequantize_int8' in content
        assert 'quantize_int4' in content
        print(f"  ✓ 量化工具生成: {path}")

    def test_generate_conv2d_kernel(self):
        """测试生成Conv2d核函数"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_conv")

        spec = KernelSpec(
            name='conv1',
            layer_type='Conv2d',
            in_channels=3,
            out_channels=16,
            weight_bit=8,
            activation_bit=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        path = generator._generate_kernel(spec)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        assert 'conv1' in content
        assert '__global__' in content
        assert 'dequantize' in content
        print(f"  ✓ Conv2d核函数生成: {path}")

    def test_generate_linear_kernel(self):
        """测试生成Linear核函数"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_linear")

        spec = KernelSpec(
            name='fc1',
            layer_type='Linear',
            in_channels=128,
            out_channels=64,
            weight_bit=4,
            activation_bit=8,
        )

        path = generator._generate_kernel(spec)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        assert 'fc1' in content
        assert '__global__' in content
        print(f"  ✓ Linear核函数生成: {path}")

    def test_generate_build_script(self):
        """测试生成编译脚本"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_build")

        spec = KernelSpec(
            name='conv1',
            layer_type='Conv2d',
            in_channels=3,
            out_channels=16,
            weight_bit=8,
            activation_bit=8,
            kernel_size=3,
        )
        generator.add_kernel(spec)

        path = generator._generate_build_script()
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        assert 'nvcc' in content
        assert '.so' in content
        print(f"  ✓ 编译脚本生成: {path}")

    def test_generate_python_binding(self):
        """测试生成Python绑定"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_binding")

        spec = KernelSpec(
            name='conv1',
            layer_type='Conv2d',
            in_channels=3,
            out_channels=16,
            weight_bit=8,
            activation_bit=8,
            kernel_size=3,
        )
        generator.add_kernel(spec)

        path = generator._generate_python_binding()
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()

        assert 'conv1' in content
        assert 'torch' in content
        print(f"  ✓ Python绑定生成: {path}")

    def test_generate_all(self):
        """测试生成所有文件"""
        generator = CUDAKernelGenerator(output_dir="/tmp/test_cuda_all")

        # 添加多个核函数
        generator.add_kernel(KernelSpec(
            name='conv1', layer_type='Conv2d',
            in_channels=3, out_channels=16,
            weight_bit=8, activation_bit=8,
            kernel_size=3,
        ))
        generator.add_kernel(KernelSpec(
            name='fc1', layer_type='Linear',
            in_channels=128, out_channels=64,
            weight_bit=4, activation_bit=8,
        ))

        files = generator.generate()

        assert 'header' in files
        assert 'quant_utils' in files
        assert 'build_script' in files
        assert 'python_binding' in files
        assert 'conv1' in files
        assert 'fc1' in files

        print(f"  ✓ 生成所有文件: {list(files.keys())}")


class TestGenerateFromConfig:
    """测试从配置生成"""

    def test_from_json_config(self):
        """测试从JSON配置生成"""
        # 创建配置
        config = {
            'model_name': 'test_model',
            'layers': [
                {
                    'name': 'features.0',
                    'type': 'Conv2d',
                    'in_channels': 3,
                    'out_channels': 16,
                    'is_quantized': True,
                    'quantization': {'weight_bit': 8, 'activation_bit': 8},
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                },
                {
                    'name': 'classifier.0',
                    'type': 'Linear',
                    'in_channels': 512,
                    'out_channels': 64,
                    'is_quantized': True,
                    'quantization': {'weight_bit': 4, 'activation_bit': 8},
                },
            ]
        }

        # 保存配置
        config_path = '/tmp/test_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # 生成核函数
        output_dir = '/tmp/test_cuda_from_config'
        files = generate_kernels_from_config(config_path, output_dir)

        assert len(files) > 0
        print(f"  ✓ 从JSON配置生成: {list(files.keys())}")


class TestIntegration:
    """集成测试"""

    def test_full_pipeline(self):
        """测试完整流程"""
        # 1. 创建模型
        model = SimpleCNN()

        # 2. 提取模型信息
        info = extract_model_info(model, 'test_model')

        # 3. 导出配置
        config_path = '/tmp/test_pipeline/kernel_config.json'
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        export_kernel_config(info, config_path)

        # 4. 生成核函数
        output_dir = '/tmp/test_pipeline/cuda_kernels'
        files = generate_kernels_from_config(config_path, output_dir)

        # 5. 验证
        assert os.path.exists(config_path)
        assert len(files) > 0

        print("\n  完整流程:")
        print(f"    1. 模型: {info.model_name}")
        print(f"    2. 层数: {len(info.layers)}")
        print(f"    3. 配置: {config_path}")
        print(f"    4. 生成文件: {list(files.keys())}")

        # 打印部分生成的代码
        for name, path in files.items():
            if name in ['header', 'quant_utils']:
                continue
            if os.path.exists(path):
                with open(path) as f:
                    content = f.read()
                lines = content.split('\n')[:10]
                print(f"\n    {name}.cu (前10行):")
                for line in lines:
                    print(f"      {line}")

        print("\n  ✓ 完整流程成功")


if __name__ == "__main__":
    print("=" * 70)
    print("CUDA核函数生成器测试")
    print("=" * 70)

    test_classes = [
        TestKernelSpec,
        TestCUDAKernelGenerator,
        TestGenerateFromConfig,
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
