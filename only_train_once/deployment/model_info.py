"""
模型信息提取模块
从剪枝+量化后的模型中提取部署所需的所有信息。

关键点：
1. 每层有独立的量化参数 (d_quant, q_m, t_quant)
2. 位宽是从量化参数推导出来的，不是固定的
3. 剪枝后通道数不规则
4. 需要为每层单独配置核函数
"""

import math
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QuantizationInfo:
    """量化信息"""
    # 权重量化参数
    d_quant_wt: float = 1.0
    q_m_wt: float = 1.0
    t_quant_wt: float = 1.0

    # 激活量化参数
    d_quant_act: float = 1.0
    q_m_act: float = 1.0
    t_quant_act: float = 1.0

    # 推导出的位宽
    weight_bit: int = 8
    activation_bit: int = 8

    # 量化类型
    quant_type: str = "symmetric+nonlinear"
    quant_mode: str = "weight_only"


@dataclass
class LayerStructure:
    """层结构信息"""
    name: str
    type: str  # Conv2d, Linear, BatchNorm2d, etc.

    # 原始维度
    original_in_channels: int = 0
    original_out_channels: int = 0

    # 剪枝后维度
    pruned_in_channels: int = 0
    pruned_out_channels: int = 0

    # 卷积特有参数
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    groups: int = 1

    # 量化信息
    quant_info: Optional[QuantizationInfo] = None

    # 剪枝信息
    pruned_in_indices: List[int] = field(default_factory=list)
    pruned_out_indices: List[int] = field(default_factory=list)

    # 是否是量化的层
    is_quantized: bool = False


@dataclass
class ModelInfo:
    """模型完整信息"""
    # 基本信息
    model_name: str = ""
    total_params: int = 0
    trainable_params: int = 0

    # 层信息
    layers: Dict[str, LayerStructure] = field(default_factory=dict)

    # 全局配置
    has_mixed_precision: bool = False
    has_pruning: bool = False
    min_bit_width: int = 32
    max_bit_width: int = 0

    # 量化统计
    bit_width_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'layers': {name: asdict(layer) for name, layer in self.layers.items()},
            'has_mixed_precision': self.has_mixed_precision,
            'has_pruning': self.has_pruning,
            'min_bit_width': self.min_bit_width,
            'max_bit_width': self.max_bit_width,
            'bit_width_distribution': self.bit_width_distribution,
        }

    def to_json(self, path: str):
        """导出为JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"模型信息已导出: {path}")


def extract_model_info(model: nn.Module, model_name: str = "") -> ModelInfo:
    """从模型中提取完整信息

    Args:
        model: 剪枝+量化后的模型
        model_name: 模型名称

    Returns:
        模型信息
    """
    info = ModelInfo(
        model_name=model_name or type(model).__name__,
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # 遍历所有模块
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # 跳过非叶子模块

        layer = _extract_layer_info(name, module)
        if layer is not None:
            info.layers[name] = layer

            # 更新统计
            if layer.is_quantized:
                bit = layer.quant_info.weight_bit
                info.bit_width_distribution[bit] = info.bit_width_distribution.get(bit, 0) + 1
                info.min_bit_width = min(info.min_bit_width, bit)
                info.max_bit_width = max(info.max_bit_width, bit)

    # 检查是否有混合精度
    if info.min_bit_width != info.max_bit_width:
        info.has_mixed_precision = True

    # 检查是否有剪枝
    for layer in info.layers.values():
        if layer.pruned_in_channels < layer.original_in_channels or \
           layer.pruned_out_channels < layer.original_out_channels:
            info.has_pruning = True
            break

    return info


def _extract_layer_info(name: str, module: nn.Module) -> Optional[LayerStructure]:
    """提取单层信息"""
    layer_type = type(module).__name__

    # Conv2d层
    if isinstance(module, nn.Conv2d):
        layer = LayerStructure(
            name=name,
            type='Conv2d',
            original_in_channels=module.in_channels,
            original_out_channels=module.out_channels,
            pruned_in_channels=module.in_channels,
            pruned_out_channels=module.out_channels,
            kernel_size=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
            stride=module.stride[0] if isinstance(module.stride, tuple) else module.stride,
            padding=module.padding[0] if isinstance(module.padding, tuple) else module.padding,
            groups=module.groups,
        )

        # 检查是否量化
        if hasattr(module, 'd_quant_wt'):
            layer.is_quantized = True
            layer.quant_info = _extract_quantization_info(module)

        return layer

    # Linear层
    elif isinstance(module, nn.Linear):
        layer = LayerStructure(
            name=name,
            type='Linear',
            original_in_channels=module.in_features,
            original_out_channels=module.out_features,
            pruned_in_channels=module.in_features,
            pruned_out_channels=module.out_features,
        )

        # 检查是否量化
        if hasattr(module, 'd_quant_wt'):
            layer.is_quantized = True
            layer.quant_info = _extract_quantization_info(module)

        return layer

    # BatchNorm层
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        num_features = module.num_features if hasattr(module, 'num_features') else 0
        return LayerStructure(
            name=name,
            type=layer_type,
            original_in_channels=num_features,
            original_out_channels=num_features,
            pruned_in_channels=num_features,
            pruned_out_channels=num_features,
        )

    # 其他层（ReLU, MaxPool等）
    else:
        return LayerStructure(
            name=name,
            type=layer_type,
        )


def _extract_quantization_info(module: nn.Module) -> QuantizationInfo:
    """提取量化信息"""
    info = QuantizationInfo()

    # 权重量化参数
    if hasattr(module, 'd_quant_wt'):
        info.d_quant_wt = module.d_quant_wt.item()
    if hasattr(module, 'q_m_wt'):
        info.q_m_wt = module.q_m_wt.item()
    if hasattr(module, 't_quant_wt'):
        info.t_quant_wt = module.t_quant_wt.item()

    # 激活量化参数
    if hasattr(module, 'd_quant_act'):
        info.d_quant_act = module.d_quant_act.item()
    if hasattr(module, 'q_m_act'):
        info.q_m_act = module.q_m_act.item()
    if hasattr(module, 't_quant_act'):
        info.t_quant_act = module.t_quant_act.item()

    # 计算位宽
    info.weight_bit = _compute_bit_width(
        info.d_quant_wt, info.q_m_wt, info.t_quant_wt
    )
    info.activation_bit = _compute_bit_width(
        info.d_quant_act, info.q_m_act, info.t_quant_act
    )

    # 量化类型
    if hasattr(module, 'quant_type'):
        info.quant_type = module.quant_type.value
    if hasattr(module, 'quant_mode'):
        info.quant_mode = module.quant_mode.value

    return info


def _compute_bit_width(d_quant: float, q_m: float, t_quant: float) -> int:
    """计算位宽

    公式: bit_width = round(log2(exp(t * log(qmax)) / abs(d) + 1) + 1)

    Args:
        d_quant: 量化步长
        q_m: 量化范围
        t_quant: 非线性参数

    Returns:
        位宽
    """
    if abs(d_quant) < 1e-10:
        return 32  # 默认

    qmax = abs(q_m)
    if qmax < 1e-10:
        return 32

    try:
        bit_width = round(math.log2(math.exp(t_quant * math.log(qmax)) / abs(d_quant) + 1) + 1)
        return max(1, min(32, bit_width))
    except (ValueError, OverflowError):
        return 32


def get_kernel_config_for_layer(layer: LayerStructure) -> Dict[str, Any]:
    """为单层生成核函数配置

    Args:
        layer: 层结构信息

    Returns:
        核函数配置
    """
    config = {
        'name': layer.name,
        'type': layer.type,
        'in_channels': layer.pruned_in_channels,
        'out_channels': layer.pruned_out_channels,
        'is_quantized': layer.is_quantized,
    }

    if layer.is_quantized and layer.quant_info:
        config['quantization'] = {
            'weight_bit': layer.quant_info.weight_bit,
            'activation_bit': layer.quant_info.activation_bit,
            'd_quant_wt': layer.quant_info.d_quant_wt,
            'q_m_wt': layer.quant_info.q_m_wt,
            't_quant_wt': layer.quant_info.t_quant_wt,
            'd_quant_act': layer.quant_info.d_quant_act,
            'q_m_act': layer.quant_info.q_m_act,
            't_quant_act': layer.quant_info.t_quant_act,
            'quant_type': layer.quant_info.quant_type,
            'quant_mode': layer.quant_info.quant_mode,
        }

    if layer.type == 'Conv2d':
        config['kernel_size'] = layer.kernel_size
        config['stride'] = layer.stride
        config['padding'] = layer.padding
        config['groups'] = layer.groups

    if layer.pruned_in_indices:
        config['pruned_in_indices'] = layer.pruned_in_indices
    if layer.pruned_out_indices:
        config['pruned_out_indices'] = layer.pruned_out_indices

    return config


def export_kernel_config(model_info: ModelInfo, output_path: str):
    """导出核函数配置

    Args:
        model_info: 模型信息
        output_path: 输出路径
    """
    config = {
        'model_name': model_info.model_name,
        'total_params': model_info.total_params,
        'has_mixed_precision': model_info.has_mixed_precision,
        'has_pruning': model_info.has_pruning,
        'min_bit_width': model_info.min_bit_width,
        'max_bit_width': model_info.max_bit_width,
        'bit_width_distribution': model_info.bit_width_distribution,
        'layers': [],
    }

    for name, layer in model_info.layers.items():
        if layer.type in ['Conv2d', 'Linear']:
            layer_config = get_kernel_config_for_layer(layer)
            config['layers'].append(layer_config)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"核函数配置已导出: {output_path}")
    logger.info(f"  总层数: {len(config['layers'])}")
    logger.info(f"  位宽分布: {config['bit_width_distribution']}")
