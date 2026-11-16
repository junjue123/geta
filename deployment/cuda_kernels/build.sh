#!/bin/bash
# geta CUDA核函数编译脚本

NVCC=nvcc
NVCC_FLAGS="-O3 -shared -Xcompiler -fPIC"

# 编译所有核函数
$NVCC $NVCC_FLAGS -o libgeta_kernels.so \
    geta_conv1_conv2d_w8a8.cu

echo "编译完成: libgeta_kernels.so"
