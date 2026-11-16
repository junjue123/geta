"""
CUDA核函数自动生成器
根据模型配置自动生成CUDA核函数代码。

支持：
1. 不同位宽 (INT4, INT8, FP16, FP32)
2. 不同层类型 (Conv2d, Linear)
3. 混合精度
4. 结构化剪枝
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KernelSpec:
    """核函数规格"""
    name: str
    layer_type: str  # Conv2d, Linear
    in_channels: int
    out_channels: int
    weight_bit: int
    activation_bit: int
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    groups: int = 1


class CUDAKernelGenerator:
    """CUDA核函数生成器

    根据模型配置生成CUDA核函数代码。

    Args:
        output_dir: 输出目录
        prefix: 代码前缀
    """

    def __init__(self, output_dir: str = "./cuda_kernels", prefix: str = "geta"):
        self.output_dir = output_dir
        self.prefix = prefix
        self.kernels: List[KernelSpec] = []

    def add_kernel(self, spec: KernelSpec):
        """添加核函数规格"""
        self.kernels.append(spec)

    def add_from_config(self, config: Dict[str, Any]):
        """从配置添加核函数

        Args:
            config: 模型配置字典
        """
        for layer in config.get('layers', []):
            if layer.get('is_quantized', False):
                quant = layer.get('quantization', {})
                spec = KernelSpec(
                    name=layer['name'].replace('.', '_'),
                    layer_type=layer['type'],
                    in_channels=layer.get('in_channels', 0),
                    out_channels=layer.get('out_channels', 0),
                    weight_bit=quant.get('weight_bit', 8),
                    activation_bit=quant.get('activation_bit', 8),
                    kernel_size=layer.get('kernel_size'),
                    stride=layer.get('stride'),
                    padding=layer.get('padding'),
                    groups=layer.get('groups', 1),
                )
                self.add_kernel(spec)

    def generate(self) -> Dict[str, str]:
        """生成所有核函数代码

        Returns:
            生成的文件路径字典
        """
        os.makedirs(self.output_dir, exist_ok=True)

        files = {}

        # 1. 生成通用头文件
        header_path = self._generate_header()
        files['header'] = header_path

        # 2. 生成量化工具函数
        quant_utils_path = self._generate_quant_utils()
        files['quant_utils'] = quant_utils_path

        # 3. 为每个核函数生成代码
        for spec in self.kernels:
            kernel_path = self._generate_kernel(spec)
            files[spec.name] = kernel_path

        # 4. 生成编译脚本
        build_path = self._generate_build_script()
        files['build_script'] = build_path

        # 5. 生成Python绑定
        binding_path = self._generate_python_binding()
        files['python_binding'] = binding_path

        logger.info(f"生成了 {len(self.kernels)} 个核函数")
        return files

    def _generate_header(self) -> str:
        """生成通用头文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        content = f"""/**
 * {self.prefix} - 自动生成的CUDA核函数
 *
 * 支持混合精度量化推理
 * 生成时间: 自动生成
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// 量化类型定义
typedef struct {{
    float scale;
    int32_t zero_point;
    int32_t num_bits;
    bool symmetric;
}} QuantParams;

// 核函数声明
"""

        # 为每个核函数添加声明
        for spec in self.kernels:
            func_name = self._get_function_name(spec)
            content += f"void {func_name}(\n"
            content += f"    const float* input,\n"
            content += f"    const int8_t* weight,\n"
            content += f"    const float* bias,\n"
            content += f"    float* output,\n"
            content += f"    int batch_size,\n"
            content += f"    int in_channels,\n"
            content += f"    int out_channels"
            if spec.layer_type == 'Conv2d':
                content += f",\n    int height,\n    int width,\n    int out_height,\n    int out_width"
            content += f"\n);\n\n"

        path = os.path.join(self.output_dir, f"{self.prefix}_kernels.h")
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _generate_quant_utils(self) -> str:
        """生成量化工具函数"""
        os.makedirs(self.output_dir, exist_ok=True)
        content = f"""/**
 * 量化工具函数
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// INT8量化
__device__ __forceinline__ int8_t quantize_int8(float value, float scale, int32_t zero_point) {{
    int32_t quantized = __float2int_rn(value / scale) + zero_point;
    quantized = max(-128, min(127, quantized));
    return (int8_t)quantized;
}}

// INT8反量化
__device__ __forceinline__ float dequantize_int8(int8_t value, float scale, int32_t zero_point) {{
    return (float)(value - zero_point) * scale;
}}

// INT4量化 (打包到int8)
__device__ __forceinline__ int8_t quantize_int4(float value, float scale, int32_t zero_point) {{
    int32_t quantized = __float2int_rn(value / scale) + zero_point;
    quantized = max(-8, min(7, quantized));
    return (int8_t)(quantized & 0x0F);
}}

// INT4反量化
__device__ __forceinline__ float dequantize_int4(int8_t value, float scale, int32_t zero_point) {{
    // 符号扩展
    int32_t signed_value = (value & 0x08) ? (value | 0xFFFFFFF0) : (value & 0x0F);
    return (float)(signed_value - zero_point) * scale;
}}

// FP16转换
__device__ __forceinline__ half float_to_half(float value) {{
    return __float2half(value);
}}

__device__ __forceinline__ float half_to_float(half value) {{
    return __half2float(value);
}}

// 打包INT4到INT8 (两个INT4打包成一个INT8)
__device__ __forceinline__ int8_t pack_int4(int8_t low, int8_t high) {{
    return (low & 0x0F) | ((high & 0x0F) << 4);
}}

// 解包INT8到两个INT4
__device__ __forceinline__ void unpack_int4(int8_t packed, int8_t& low, int8_t& high) {{
    low = packed & 0x0F;
    high = (packed >> 4) & 0x0F;
    // 符号扩展
    if (low & 0x08) low |= 0xF0;
    if (high & 0x08) high |= 0xF0;
}}
"""

        path = os.path.join(self.output_dir, f"{self.prefix}_quant_utils.cuh")
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _generate_kernel(self, spec: KernelSpec) -> str:
        """生成单个核函数"""
        os.makedirs(self.output_dir, exist_ok=True)
        func_name = self._get_function_name(spec)

        if spec.layer_type == 'Conv2d':
            content = self._generate_conv2d_kernel(spec, func_name)
        else:  # Linear
            content = self._generate_linear_kernel(spec, func_name)

        path = os.path.join(self.output_dir, f"{func_name}.cu")
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _generate_conv2d_kernel(self, spec: KernelSpec, func_name: str) -> str:
        """生成Conv2d核函数"""
        # 根据位宽选择数据类型
        weight_type = self._get_weight_type(spec.weight_bit)
        input_type = self._get_input_type(spec.activation_bit)

        content = f"""/**
 * Conv2d核函数
 * 层名: {spec.name}
 * 输入: {spec.in_channels} channels
 * 输出: {spec.out_channels} channels
 * 权重位宽: {spec.weight_bit}-bit
 * 激活位宽: {spec.activation_bit}-bit
 * 卷积核: {spec.kernel_size}x{spec.kernel_size}
 */

#include <cuda_runtime.h>
#include "{self.prefix}_quant_utils.cuh"

// 基础Conv2d核函数
__global__ void {func_name}_kernel(
    const {input_type}* __restrict__ input,
    const {weight_type}* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int out_height,
    int out_width,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point
) {{
    // 计算输出位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_height * out_width;

    if (idx >= total) return;

    // 解析索引
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);

    float sum = 0.0f;

    // 卷积计算
    for (int ic = 0; ic < in_channels; ic++) {{
        for (int kh = 0; kh < {spec.kernel_size}; kh++) {{
            for (int kw = 0; kw < {spec.kernel_size}; kw++) {{
                int h_in = h_out * {spec.stride or 1} - {spec.padding or 0} + kh;
                int w_in = w_out * {spec.stride or 1} - {spec.padding or 0} + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {{
                    // 获取输入值
                    int input_idx = ((b * in_channels + ic) * height + h_in) * width + w_in;
                    float in_val = dequantize_{self._get_quant_func_name(spec.activation_bit)}(
                        input[input_idx], input_scale, input_zero_point
                    );

                    // 获取权重值
                    int weight_idx = ((oc * in_channels + ic) * {spec.kernel_size} + kh) * {spec.kernel_size} + kw;
                    float w_val = dequantize_{self._get_quant_func_name(spec.weight_bit)}(
                        weight[weight_idx], weight_scale, weight_zero_point
                    );

                    sum += in_val * w_val;
                }}
            }}
        }}
    }}

    // 加偏置
    if (bias != nullptr) {{
        sum += bias[oc];
    }}

    // 写入输出
    output[idx] = sum;
}}

// 主机端接口
extern "C" void {func_name}(
    const float* input,
    const {weight_type}* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int out_height,
    int out_width,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point
) {{
    int total = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    {func_name}_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        height, width, out_height, out_width,
        input_scale, input_zero_point,
        weight_scale, weight_zero_point
    );
}}
"""
        return content

    def _generate_linear_kernel(self, spec: KernelSpec, func_name: str) -> str:
        """生成Linear核函数"""
        weight_type = self._get_weight_type(spec.weight_bit)
        input_type = self._get_input_type(spec.activation_bit)

        content = f"""/**
 * Linear核函数
 * 层名: {spec.name}
 * 输入: {spec.in_channels} features
 * 输出: {spec.out_channels} features
 * 权重位宽: {spec.weight_bit}-bit
 * 激活位宽: {spec.activation_bit}-bit
 */

#include <cuda_runtime.h>
#include "{self.prefix}_quant_utils.cuh"

// 基础Linear核函数
__global__ void {func_name}_kernel(
    const {input_type}* __restrict__ input,
    const {weight_type}* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point
) {{
    // 计算输出位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;

    if (idx >= total) return;

    // 解析索引
    int of = idx % out_features;
    int b = idx / out_features;

    float sum = 0.0f;

    // 矩阵乘法
    for (int if_idx = 0; if_idx < in_features; if_idx++) {{
        // 获取输入值
        int input_idx = b * in_features + if_idx;
        float in_val = dequantize_{self._get_quant_func_name(spec.activation_bit)}(
            input[input_idx], input_scale, input_zero_point
        );

        // 获取权重值
        int weight_idx = of * in_features + if_idx;
        float w_val = dequantize_{self._get_quant_func_name(spec.weight_bit)}(
            weight[weight_idx], weight_scale, weight_zero_point
        );

        sum += in_val * w_val;
    }}

    // 加偏置
    if (bias != nullptr) {{
        sum += bias[of];
    }}

    // 写入输出
    output[idx] = sum;
}}

// 主机端接口
extern "C" void {func_name}(
    const float* input,
    const {weight_type}* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point
) {{
    int total = batch_size * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    {func_name}_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        batch_size, in_features, out_features,
        input_scale, input_zero_point,
        weight_scale, weight_zero_point
    );
}}
"""
        return content

    def _generate_build_script(self) -> str:
        """生成编译脚本"""
        os.makedirs(self.output_dir, exist_ok=True)
        content = f"""#!/bin/bash
# {self.prefix} CUDA核函数编译脚本

NVCC=nvcc
NVCC_FLAGS="-O3 -shared -Xcompiler -fPIC"

# 编译所有核函数
"""

        # 收集所有.cu文件
        cu_files = []
        for spec in self.kernels:
            func_name = self._get_function_name(spec)
            cu_files.append(f"{func_name}.cu")

        # 生成编译命令
        content += f"$NVCC $NVCC_FLAGS -o lib{self.prefix}_kernels.so \\\n"
        for i, f in enumerate(cu_files):
            if i < len(cu_files) - 1:
                content += f"    {f} \\\n"
            else:
                content += f"    {f}\n"

        content += f"""
echo "编译完成: lib{self.prefix}_kernels.so"
"""

        path = os.path.join(self.output_dir, "build.sh")
        with open(path, 'w') as f:
            f.write(content)
        os.chmod(path, 0o755)
        return path

    def _generate_python_binding(self) -> str:
        """生成Python绑定"""
        os.makedirs(self.output_dir, exist_ok=True)
        content = f"""'''
{self.prefix} Python绑定
自动生成的CUDA核函数Python接口
'''

import torch
import ctypes
import os

# 加载CUDA库
_lib = None

def load_library():
    global _lib
    lib_path = os.path.join(os.path.dirname(__file__), 'lib{self.prefix}_kernels.so')
    if os.path.exists(lib_path):
        _lib = ctypes.CDLL(lib_path)
    else:
        raise RuntimeError(f"找不到CUDA库: {{lib_path}}")

def _check_lib():
    if _lib is None:
        load_library()

"""

        # 为每个核函数生成Python接口
        for spec in self.kernels:
            func_name = self._get_function_name(spec)
            py_func_name = spec.name.replace('.', '_')

            content += f"""def {py_func_name}(input_tensor, weight_tensor, bias_tensor=None,
                  input_scale=1.0, input_zero_point=0,
                  weight_scale=1.0, weight_zero_point=0):
    \"\"\"
    {spec.name} ({spec.layer_type})
    权重: {spec.weight_bit}-bit, 激活: {spec.activation_bit}-bit
    \"\"\"
    _check_lib()

    # 准备输出
    batch_size = input_tensor.shape[0]
"""

            if spec.layer_type == 'Conv2d':
                content += f"""    height, width = input_tensor.shape[2], input_tensor.shape[3]
    out_height = (height + 2 * {spec.padding} - {spec.kernel_size}) // {spec.stride} + 1
    out_width = (width + 2 * {spec.padding} - {spec.kernel_size}) // {spec.stride} + 1
    output = torch.zeros(batch_size, {spec.out_channels}, out_height, out_width,
                         device=input_tensor.device, dtype=torch.float32)
"""
            else:
                content += f"""    output = torch.zeros(batch_size, {spec.out_channels},
                         device=input_tensor.device, dtype=torch.float32)
"""

            content += f"""
    # 调用CUDA核函数
    c_func = _lib.{func_name}

    # 设置参数类型
    c_func.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # bias
        ctypes.c_void_p,  # output
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # in_channels/in_features
        ctypes.c_int,     # out_channels/out_features
"""

            if spec.layer_type == 'Conv2d':
                content += """        ctypes.c_int,     # height
        ctypes.c_int,     # width
        ctypes.c_int,     # out_height
        ctypes.c_int,     # out_width
"""

            content += """        ctypes.c_float,   # input_scale
        ctypes.c_int,     # input_zero_point
        ctypes.c_float,   # weight_scale
        ctypes.c_int,     # weight_zero_point
    ]
    c_func.restype = None

    # 调用
    bias_ptr = bias_tensor.data_ptr() if bias_tensor is not None else ctypes.c_void_p(0)
"""

            if spec.layer_type == 'Conv2d':
                content += f"""    c_func(
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        batch_size,
        {spec.in_channels},
        {spec.out_channels},
        height, width, out_height, out_width,
        input_scale, input_zero_point,
        weight_scale, weight_zero_point
    )
"""
            else:
                content += f"""    c_func(
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        batch_size,
        {spec.in_channels},
        {spec.out_channels},
        input_scale, input_zero_point,
        weight_scale, weight_zero_point
    )
"""

            content += "\n    return output\n\n"

        path = os.path.join(self.output_dir, f"{self.prefix}_bindings.py")
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _get_function_name(self, spec: KernelSpec) -> str:
        """获取核函数名称"""
        return f"{self.prefix}_{spec.name}_{spec.layer_type.lower()}_w{spec.weight_bit}a{spec.activation_bit}"

    def _get_weight_type(self, bit_width: int) -> str:
        """获取权重数据类型"""
        if bit_width <= 8:
            return "int8_t"  # INT4/INT8都用int8_t
        elif bit_width <= 16:
            return "half"
        else:
            return "float"

    def _get_input_type(self, bit_width: int) -> str:
        """获取输入数据类型"""
        if bit_width <= 8:
            return "int8_t"
        elif bit_width <= 16:
            return "half"
        else:
            return "float"

    def _get_quant_func_name(self, bit_width: int) -> str:
        """获取量化函数名称

        将位宽映射到支持的量化函数:
        - 1-4 bit -> int4
        - 5-8 bit -> int8
        - 9-16 bit -> half
        - 17-32 bit -> float
        """
        if bit_width <= 4:
            return "int4"
        elif bit_width <= 8:
            return "int8"
        elif bit_width <= 16:
            return "half"
        else:
            return "float"


def generate_kernels_from_config(config_path: str, output_dir: str = "./cuda_kernels"):
    """从配置文件生成核函数

    Args:
        config_path: 配置文件路径
        output_dir: 输出目录

    Returns:
        生成的文件路径字典
    """
    import json

    with open(config_path) as f:
        config = json.load(f)

    generator = CUDAKernelGenerator(output_dir, prefix="geta")
    generator.add_from_config(config)

    return generator.generate()
