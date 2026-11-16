/**
 * Conv2d核函数
 * 层名: conv1
 * 输入: 3 channels
 * 输出: 64 channels
 * 权重位宽: 8-bit
 * 激活位宽: 8-bit
 * 卷积核: 3x3
 */

#include <cuda_runtime.h>
#include "geta_quant_utils.cuh"

// 基础Conv2d核函数
__global__ void geta_conv1_conv2d_w8a8_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
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
) {
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
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int h_in = h_out * 1 - 1 + kh;
                int w_in = w_out * 1 - 1 + kw;

                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    // 获取输入值
                    int input_idx = ((b * in_channels + ic) * height + h_in) * width + w_in;
                    float in_val = dequantize_int8(
                        input[input_idx], input_scale, input_zero_point
                    );

                    // 获取权重值
                    int weight_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;
                    float w_val = dequantize_int8(
                        weight[weight_idx], weight_scale, weight_zero_point
                    );

                    sum += in_val * w_val;
                }
            }
        }
    }

    // 加偏置
    if (bias != nullptr) {
        sum += bias[oc];
    }

    // 写入输出
    output[idx] = sum;
}

// 主机端接口
extern "C" void geta_conv1_conv2d_w8a8(
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
    int out_width,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point
) {
    int total = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    geta_conv1_conv2d_w8a8_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        height, width, out_height, out_width,
        input_scale, input_zero_point,
        weight_scale, weight_zero_point
    );
}
