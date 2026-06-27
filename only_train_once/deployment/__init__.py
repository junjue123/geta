"""
部署模块
处理剪枝+量化模型的导出和部署。

核心组件：
1. ModelInfo - 模型信息提取
2. KernelSelector - 核函数选择器
3. CUDAKernelGenerator - CUDA核函数生成器
"""

from .model_info import (
    ModelInfo,
    LayerStructure,
    QuantizationInfo,
    extract_model_info,
    get_kernel_config_for_layer,
    export_kernel_config,
)

from .kernel_selector import KernelSelector

from .cuda_generator import CUDAKernelGenerator, generate_kernels_from_config

__all__ = [
    'ModelInfo',
    'LayerStructure',
    'QuantizationInfo',
    'extract_model_info',
    'get_kernel_config_for_layer',
    'export_kernel_config',
    'KernelSelector',
    'CUDAKernelGenerator',
    'generate_kernels_from_config',
]
