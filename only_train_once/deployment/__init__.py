"""
部署模块
处理剪枝+量化模型的导出和部署。

核心挑战：
1. 不同层可能有不同的位宽（4-bit, 8-bit等）
2. 剪枝后通道数不规则
3. 需要兼容不同推理引擎（TensorRT, ONNX, 自定义CUDA）

解决方案：
1. ModelExporter - 统一导出接口
2. DeploymentConfig - 部署配置
3. KernelSelector - 核函数选择器
"""

from .exporter import ModelExporter, DeploymentConfig
from .kernel_selector import KernelSelector

__all__ = [
    'ModelExporter',
    'DeploymentConfig',
    'KernelSelector',
]
