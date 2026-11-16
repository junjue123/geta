/**
 * geta - 自动生成的CUDA核函数
 *
 * 支持混合精度量化推理
 * 生成时间: 自动生成
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// 量化类型定义
typedef struct {
    float scale;
    int32_t zero_point;
    int32_t num_bits;
    bool symmetric;
} QuantParams;

// 核函数声明
void geta_conv1_conv2d_w8a8(
    const float* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int out_height,
    int out_width
);

