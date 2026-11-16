"""
模型导出模块
处理剪枝+量化模型的导出。

导出内容：
1. 模型权重（已剪枝+量化）
2. 元数据（位宽、剪枝掩码、量化参数）
3. 部署配置（目标平台、精度要求）
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeploymentTarget(Enum):
    """部署目标平台"""
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    CUSTOM_CUDA = "custom_cuda"
    OPENVINO = "openvino"
    TFLITE = "tflite"
    PYTORCH = "pytorch"


class QuantizationType(Enum):
    """量化类型"""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"  # 混合精度


@dataclass
class LayerInfo:
    """层信息"""
    name: str
    type: str  # Conv2d, Linear, etc.
    in_channels: int
    out_channels: int
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    groups: int = 1
    bit_width: int = 8
    is_pruned: bool = False
    pruned_in_channels: int = 0  # 被剪掉的输入通道数
    pruned_out_channels: int = 0  # 被剪掉的输出通道数


@dataclass
class QuantizationParams:
    """量化参数"""
    scale: float
    zero_point: int
    min_val: float
    max_val: float
    num_bits: int
    symmetric: bool = True


@dataclass
class DeploymentConfig:
    """部署配置"""
    target: DeploymentTarget = DeploymentTarget.PYTORCH
    quant_type: QuantizationType = QuantizationType.INT8
    export_float16: bool = False
    export_int8: bool = False
    optimize_for_inference: bool = True
    include_quantization_params: bool = True
    include_pruning_mask: bool = False
    batch_size: int = 1
    input_shape: Optional[List[int]] = None


class ModelExporter:
    """模型导出器

    将剪枝+量化后的模型导出为不同格式。

    Args:
        model: 剪枝+量化后的模型
        config: 部署配置
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DeploymentConfig] = None
    ):
        self.model = model
        self.config = config or DeploymentConfig()
        self.layer_info: Dict[str, LayerInfo] = {}
        self.quant_params: Dict[str, QuantizationParams] = {}

        # 分析模型结构
        self._analyze_model()

    def _analyze_model(self):
        """分析模型结构，收集层信息"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.layer_info[name] = LayerInfo(
                    name=name,
                    type='Conv2d',
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                    stride=module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                    padding=module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                    groups=module.groups,
                )
            elif isinstance(module, nn.Linear):
                self.layer_info[name] = LayerInfo(
                    name=name,
                    type='Linear',
                    in_channels=module.in_features,
                    out_channels=module.out_features,
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                # 记录BN/GN的通道数
                if hasattr(module, 'num_features'):
                    self.layer_info[name] = LayerInfo(
                        name=name,
                        type=type(module).__name__,
                        in_channels=module.num_features,
                        out_channels=module.num_features,
                    )

    def export(
        self,
        output_dir: str,
        model_name: str = "model"
    ) -> Dict[str, str]:
        """导出模型

        Args:
            output_dir: 输出目录
            model_name: 模型名称

        Returns:
            导出文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}

        # 根据目标平台选择导出方式
        if self.config.target == DeploymentTarget.PYTORCH:
            exported_files.update(self._export_pytorch(output_dir, model_name))
        elif self.config.target == DeploymentTarget.ONNX:
            exported_files.update(self._export_onnx(output_dir, model_name))
        elif self.config.target == DeploymentTarget.TENSORRT:
            exported_files.update(self._export_tensorrt(output_dir, model_name))
        else:
            # 默认导出PyTorch格式
            exported_files.update(self._export_pytorch(output_dir, model_name))

        # 导出元数据
        if self.config.include_quantization_params:
            meta_path = os.path.join(output_dir, f"{model_name}_meta.json")
            self._export_metadata(meta_path)
            exported_files['metadata'] = meta_path

        logger.info(f"导出完成: {exported_files}")
        return exported_files

    def _export_pytorch(
        self,
        output_dir: str,
        model_name: str
    ) -> Dict[str, str]:
        """导出PyTorch格式"""
        files = {}

        # 1. 保存完整模型
        model_path = os.path.join(output_dir, f"{model_name}.pt")
        if self.config.export_float16:
            self.model.half()
        torch.save(self.model, model_path)
        files['model'] = model_path

        # 2. 保存state_dict
        state_dict_path = os.path.join(output_dir, f"{model_name}_state_dict.pt")
        torch.save(self.model.state_dict(), state_dict_path)
        files['state_dict'] = state_dict_path

        # 3. TorchScript导出（如果需要优化推理）
        if self.config.optimize_for_inference:
            try:
                scripted_path = os.path.join(output_dir, f"{model_name}_scripted.pt")
                if self.config.input_shape:
                    dummy_input = torch.randn(self.config.batch_size, *self.config.input_shape)
                    if self.config.export_float16:
                        dummy_input = dummy_input.half()
                    scripted = torch.jit.trace(self.model, dummy_input)
                    scripted.save(scripted_path)
                    files['scripted'] = scripted_path
            except Exception as e:
                logger.warning(f"TorchScript导出失败: {e}")

        return files

    def _export_onnx(
        self,
        output_dir: str,
        model_name: str
    ) -> Dict[str, str]:
        """导出ONNX格式"""
        files = {}

        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

        if self.config.input_shape:
            dummy_input = torch.randn(self.config.batch_size, *self.config.input_shape)
            if self.config.export_float16:
                dummy_input = dummy_input.half()

            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            files['onnx'] = onnx_path

        return files

    def _export_tensorrt(
        self,
        output_dir: str,
        model_name: str
    ) -> Dict[str, str]:
        """导出TensorRT格式（需要tensorrt库）"""
        files = {}

        # 先导出ONNX
        onnx_files = self._export_onnx(output_dir, model_name)
        if 'onnx' in onnx_files:
            files['onnx'] = onnx_files['onnx']

            # 尝试转换为TensorRT
            try:
                import tensorrt as trt
                trt_path = os.path.join(output_dir, f"{model_name}.trt")

                logger.info("TensorRT转换需要trtexec工具或TensorRT Python API")
                logger.info(f"建议命令: trtexec --onnx={files['onnx']} --saveEngine={trt_path}")

                if self.config.export_int8:
                    logger.info("INT8量化需要校准数据集")

                files['trt_path_hint'] = trt_path

            except ImportError:
                logger.warning("TensorRT未安装，仅导出ONNX格式")

        return files

    def _export_metadata(self, output_path: str):
        """导出模型元数据"""
        # 自定义JSON编码器处理Enum类型
        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.value
                return super().default(obj)

        metadata = {
            'layer_info': {name: asdict(info) for name, info in self.layer_info.items()},
            'quant_params': {name: asdict(params) for name, params in self.quant_params.items()},
            'config': asdict(self.config),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=EnumEncoder)

    def get_layer_summary(self) -> str:
        """获取层摘要信息"""
        lines = ["=" * 80]
        lines.append(f"{'Layer Name':30s} | {'Type':10s} | {'In':>6s} | {'Out':>6s} | {'Bit':>3s} | {'Pruned':>6s}")
        lines.append("-" * 80)

        for name, info in self.layer_info.items():
            lines.append(
                f"{name:30s} | {info.type:10s} | {info.in_channels:6d} | "
                f"{info.out_channels:6d} | {info.bit_width:3d} | "
                f"{'Yes' if info.is_pruned else 'No':>6s}"
            )

        lines.append("=" * 80)
        total_params = sum(p.numel() for p in self.model.parameters())
        lines.append(f"Total parameters: {total_params:,}")

        return "\n".join(lines)

    def set_layer_bit_width(self, layer_name: str, bit_width: int):
        """设置层的位宽

        Args:
            layer_name: 层名称
            bit_width: 位宽（4, 8, 16等）
        """
        if layer_name in self.layer_info:
            self.layer_info[layer_name].bit_width = bit_width

    def set_quantization_params(
        self,
        layer_name: str,
        scale: float,
        zero_point: int,
        min_val: float,
        max_val: float,
        num_bits: int = 8,
        symmetric: bool = True
    ):
        """设置量化参数

        Args:
            layer_name: 层名称
            scale: 缩放因子
            zero_point: 零点
            min_val: 最小值
            max_val: 最大值
            num_bits: 位数
            symmetric: 是否对称量化
        """
        self.quant_params[layer_name] = QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            min_val=min_val,
            max_val=max_val,
            num_bits=num_bits,
            symmetric=symmetric
        )

    def export_for_kernel(
        self,
        output_dir: str,
        model_name: str = "model"
    ) -> Dict[str, str]:
        """导出核函数所需的配置

        为自定义CUDA核函数导出配置信息。

        Args:
            output_dir: 输出目录
            model_name: 模型名称

        Returns:
            配置文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # 生成核函数配置
        kernel_config = {
            'layers': [],
            'global_config': {
                'total_layers': len(self.layer_info),
                'has_mixed_precision': any(
                    info.bit_width != 8 for info in self.layer_info.values()
                ),
                'has_pruning': any(
                    info.is_pruned for info in self.layer_info.values()
                ),
            }
        }

        for name, info in self.layer_info.items():
            layer_config = {
                'name': name,
                'type': info.type,
                'in_channels': info.in_channels,
                'out_channels': info.out_channels,
                'bit_width': info.bit_width,
                'is_pruned': info.is_pruned,
            }

            # 添加量化参数
            if name in self.quant_params:
                qp = self.quant_params[name]
                layer_config['quant_params'] = {
                    'scale': qp.scale,
                    'zero_point': qp.zero_point,
                    'num_bits': qp.num_bits,
                    'symmetric': qp.symmetric,
                }

            # 添加卷积特有参数
            if info.type == 'Conv2d':
                layer_config['kernel_size'] = info.kernel_size
                layer_config['stride'] = info.stride
                layer_config['padding'] = info.padding
                layer_config['groups'] = info.groups

            kernel_config['layers'].append(layer_config)

        # 保存配置
        config_path = os.path.join(output_dir, f"{model_name}_kernel_config.json")
        with open(config_path, 'w') as f:
            json.dump(kernel_config, f, indent=2)

        logger.info(f"核函数配置已导出: {config_path}")
        return {'kernel_config': config_path}
