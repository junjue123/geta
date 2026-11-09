'''
geta Python绑定
自动生成的CUDA核函数Python接口
'''

import torch
import ctypes
import os

# 加载CUDA库
_lib = None

def load_library():
    global _lib
    lib_path = os.path.join(os.path.dirname(__file__), 'libgeta_kernels.so')
    if os.path.exists(lib_path):
        _lib = ctypes.CDLL(lib_path)
    else:
        raise RuntimeError(f"找不到CUDA库: {lib_path}")

def _check_lib():
    if _lib is None:
        load_library()

def conv1(input_tensor, weight_tensor, bias_tensor=None,
                  input_scale=1.0, input_zero_point=0,
                  weight_scale=1.0, weight_zero_point=0):
    """
    conv1 (Conv2d)
    权重: 8-bit, 激活: 8-bit
    """
    _check_lib()

    # 准备输出
    batch_size = input_tensor.shape[0]
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    out_height = (height + 2 * 1 - 3) // 1 + 1
    out_width = (width + 2 * 1 - 3) // 1 + 1
    output = torch.zeros(batch_size, 64, out_height, out_width,
                         device=input_tensor.device, dtype=torch.float32)

    # 调用CUDA核函数
    c_func = _lib.geta_conv1_conv2d_w8a8

    # 设置参数类型
    c_func.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # bias
        ctypes.c_void_p,  # output
        ctypes.c_int,     # batch_size
        ctypes.c_int,     # in_channels/in_features
        ctypes.c_int,     # out_channels/out_features
        ctypes.c_int,     # height
        ctypes.c_int,     # width
        ctypes.c_int,     # out_height
        ctypes.c_int,     # out_width
        ctypes.c_float,   # input_scale
        ctypes.c_int,     # input_zero_point
        ctypes.c_float,   # weight_scale
        ctypes.c_int,     # weight_zero_point
    ]
    c_func.restype = None

    # 调用
    bias_ptr = bias_tensor.data_ptr() if bias_tensor is not None else ctypes.c_void_p(0)
    c_func(
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        batch_size,
        3,
        64,
        height, width, out_height, out_width,
        input_scale, input_zero_point,
        weight_scale, weight_zero_point
    )

    return output

