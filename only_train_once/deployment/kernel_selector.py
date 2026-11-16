"""
核函数选择器
根据模型配置选择合适的推理核函数。

不同场景需要不同的核函数：
1. 位宽：INT4, INT8, FP16, BF16, FP32
2. 剪枝：结构化剪枝 vs 非结构化剪枝
3. 硬件：NVIDIA GPU, AMD GPU, CPU, 移动端
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareTarget(Enum):
    """硬件目标"""
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    CPU = "cpu"
    MOBILE = "mobile"
    EDGE = "edge"


class KernelType(Enum):
    """核函数类型"""
    # 标准卷积
    CONV2D_FP32 = "conv2d_fp32"
    CONV2D_FP16 = "conv2d_fp16"
    CONV2D_INT8 = "conv2d_int8"
    CONV2D_INT4 = "conv2d_int4"

    # 深度可分离卷积
    DEPTHWISE_FP32 = "depthwise_fp32"
    DEPTHWISE_FP16 = "depthwise_fp16"
    DEPTHWISE_INT8 = "depthwise_int8"

    # 线性层
    LINEAR_FP32 = "linear_fp32"
    LINEAR_FP16 = "linear_fp16"
    LINEAR_INT8 = "linear_int8"
    LINEAR_INT4 = "linear_int4"

    # 剪枝后稀疏卷积
    SPARSE_CONV2D = "sparse_conv2d"
    SPARSE_LINEAR = "sparse_linear"

    # 混合精度
    MIXED_PRECISION = "mixed_precision"


@dataclass
class KernelConfig:
    """核函数配置"""
    kernel_type: KernelType
    in_channels: int
    out_channels: int
    bit_width: int
    is_sparse: bool = False
    sparsity_ratio: float = 0.0
    groups: int = 1
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None


class KernelSelector:
    """核函数选择器

    根据层配置选择最优的核函数。

    Args:
        hardware: 目标硬件
        prefer_mixed_precision: 是否优先使用混合精度
        sparse_threshold: 稀疏度阈值（超过此值使用稀疏核）
    """

    def __init__(
        self,
        hardware: HardwareTarget = HardwareTarget.NVIDIA_GPU,
        prefer_mixed_precision: bool = True,
        sparse_threshold: float = 0.5
    ):
        self.hardware = hardware
        self.prefer_mixed_precision = prefer_mixed_precision
        self.sparse_threshold = sparse_threshold

        # 核函数映射表
        self._kernel_map = self._build_kernel_map()

    def _build_kernel_map(self) -> Dict[str, KernelType]:
        """构建核函数映射表"""
        # 根据硬件选择可用的核函数
        if self.hardware == HardwareTarget.NVIDIA_GPU:
            return {
                'conv2d_fp32': KernelType.CONV2D_FP32,
                'conv2d_fp16': KernelType.CONV2D_FP16,
                'conv2d_int8': KernelType.CONV2D_INT8,
                'conv2d_int4': KernelType.CONV2D_INT4,
                'depthwise_fp32': KernelType.DEPTHWISE_FP32,
                'depthwise_fp16': KernelType.DEPTHWISE_FP16,
                'depthwise_int8': KernelType.DEPTHWISE_INT8,
                'linear_fp32': KernelType.LINEAR_FP32,
                'linear_fp16': KernelType.LINEAR_FP16,
                'linear_int8': KernelType.LINEAR_INT8,
                'linear_int4': KernelType.LINEAR_INT4,
                'sparse_conv2d': KernelType.SPARSE_CONV2D,
                'sparse_linear': KernelType.SPARSE_LINEAR,
            }
        elif self.hardware == HardwareTarget.CPU:
            # CPU只支持部分核函数
            return {
                'conv2d_fp32': KernelType.CONV2D_FP32,
                'conv2d_int8': KernelType.CONV2D_INT8,
                'linear_fp32': KernelType.LINEAR_FP32,
                'linear_int8': KernelType.LINEAR_INT8,
            }
        else:
            # 默认
            return {
                'conv2d_fp32': KernelType.CONV2D_FP32,
                'linear_fp32': KernelType.LINEAR_FP32,
            }

    def select_kernel(
        self,
        layer_type: str,
        in_channels: int,
        out_channels: int,
        bit_width: int = 8,
        sparsity_ratio: float = 0.0,
        groups: int = 1,
        kernel_size: Optional[int] = None
    ) -> KernelConfig:
        """选择核函数

        Args:
            layer_type: 层类型 (Conv2d, Linear, etc.)
            in_channels: 输入通道数
            out_channels: 输出通道数
            bit_width: 位宽
            sparsity_ratio: 稀疏度
            groups: 分组数
            kernel_size: 卷积核大小

        Returns:
            核函数配置
        """
        # 判断是否使用稀疏核
        is_sparse = sparsity_ratio > self.sparse_threshold

        # 判断是否是深度可分离卷积
        is_depthwise = (layer_type == 'Conv2d' and groups > 1 and groups == in_channels)

        # 选择核函数类型
        kernel_type = self._select_kernel_type(
            layer_type, bit_width, is_depthwise, is_sparse
        )

        return KernelConfig(
            kernel_type=kernel_type,
            in_channels=in_channels,
            out_channels=out_channels,
            bit_width=bit_width,
            is_sparse=is_sparse,
            sparsity_ratio=sparsity_ratio,
            groups=groups,
            kernel_size=kernel_size,
        )

    def _select_kernel_type(
        self,
        layer_type: str,
        bit_width: int,
        is_depthwise: bool,
        is_sparse: bool
    ) -> KernelType:
        """选择核函数类型"""
        # 稀疏核优先
        if is_sparse:
            if layer_type == 'Conv2d':
                return KernelType.SPARSE_CONV2D
            elif layer_type == 'Linear':
                return KernelType.SPARSE_LINEAR

        # 深度可分离卷积
        if is_depthwise:
            if bit_width <= 8:
                return KernelType.DEPTHWISE_INT8
            elif bit_width <= 16:
                return KernelType.DEPTHWISE_FP16
            else:
                return KernelType.DEPTHWISE_FP32

        # 标准卷积/线性
        if layer_type == 'Conv2d':
            if bit_width <= 4:
                return KernelType.CONV2D_INT4
            elif bit_width <= 8:
                return KernelType.CONV2D_INT8
            elif bit_width <= 16:
                return KernelType.CONV2D_FP16
            else:
                return KernelType.CONV2D_FP32

        elif layer_type == 'Linear':
            if bit_width <= 4:
                return KernelType.LINEAR_INT4
            elif bit_width <= 8:
                return KernelType.LINEAR_INT8
            elif bit_width <= 16:
                return KernelType.LINEAR_FP16
            else:
                return KernelType.LINEAR_FP32

        # 默认
        return KernelType.CONV2D_FP32

    def get_kernel_recommendations(
        self,
        layer_configs: List[Dict]
    ) -> List[Dict]:
        """获取核函数推荐

        Args:
            layer_configs: 层配置列表

        Returns:
            推荐的核函数配置列表
        """
        recommendations = []

        for config in layer_configs:
            kernel_config = self.select_kernel(
                layer_type=config.get('type', 'Conv2d'),
                in_channels=config.get('in_channels', 0),
                out_channels=config.get('out_channels', 0),
                bit_width=config.get('bit_width', 8),
                sparsity_ratio=config.get('sparsity_ratio', 0.0),
                groups=config.get('groups', 1),
                kernel_size=config.get('kernel_size'),
            )

            recommendations.append({
                'layer_name': config.get('name', 'unknown'),
                'kernel_type': kernel_config.kernel_type.value,
                'bit_width': kernel_config.bit_width,
                'is_sparse': kernel_config.is_sparse,
                'sparsity_ratio': kernel_config.sparsity_ratio,
            })

        return recommendations

    def generate_kernel_code_hint(
        self,
        kernel_config: KernelConfig
    ) -> str:
        """生成核函数代码提示

        Args:
            kernel_config: 核函数配置

        Returns:
            代码提示字符串
        """
        hints = []

        if kernel_config.kernel_type in [KernelType.CONV2D_INT8, KernelType.LINEAR_INT8]:
            hints.append("需要INT8量化核函数")
            hints.append("使用Tensor Core加速（如果支持）")
            hints.append(f"输入: {kernel_config.in_channels} 通道")
            hints.append(f"输出: {kernel_config.out_channels} 通道")

        elif kernel_config.kernel_type in [KernelType.CONV2D_INT4, KernelType.LINEAR_INT4]:
            hints.append("需要INT4量化核函数")
            hints.append("可能需要自定义CUDA kernel")
            hints.append("注意: INT4精度较低，需要仔细验证")

        elif kernel_config.kernel_type == KernelType.SPARSE_CONV2D:
            hints.append("需要稀疏卷积核函数")
            hints.append(f"稀疏度: {kernel_config.sparsity_ratio:.1%}")
            hints.append("考虑使用structured sparsity或block sparsity")

        elif kernel_config.kernel_type == KernelType.DEPTHWISE_INT8:
            hints.append("需要INT8深度可分离卷积核")
            hints.append("通道数较少，可能不适合Tensor Core")

        if kernel_config.bit_width != 8:
            hints.append(f"混合精度: {kernel_config.bit_width}-bit")

        return "\n".join(hints) if hints else "使用标准核函数"

    def estimate_speedup(
        self,
        kernel_config: KernelConfig,
        baseline: KernelConfig
    ) -> float:
        """估计加速比

        Args:
            kernel_config: 当前配置
            baseline: 基线配置

        Returns:
            估计加速比
        """
        # 简单估计：基于位宽和稀疏度
        bit_speedup = baseline.bit_width / kernel_config.bit_width
        sparsity_speedup = 1.0 / (1.0 - kernel_config.sparsity_ratio) if kernel_config.sparsity_ratio < 1.0 else 1.0

        return bit_speedup * sparsity_speedup
