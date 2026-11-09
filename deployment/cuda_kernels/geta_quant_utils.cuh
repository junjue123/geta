/**
 * 量化工具函数
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// INT8量化
__device__ __forceinline__ int8_t quantize_int8(float value, float scale, int32_t zero_point) {
    int32_t quantized = __float2int_rn(value / scale) + zero_point;
    quantized = max(-128, min(127, quantized));
    return (int8_t)quantized;
}

// INT8反量化
__device__ __forceinline__ float dequantize_int8(int8_t value, float scale, int32_t zero_point) {
    return (float)(value - zero_point) * scale;
}

// INT4量化 (打包到int8)
__device__ __forceinline__ int8_t quantize_int4(float value, float scale, int32_t zero_point) {
    int32_t quantized = __float2int_rn(value / scale) + zero_point;
    quantized = max(-8, min(7, quantized));
    return (int8_t)(quantized & 0x0F);
}

// INT4反量化
__device__ __forceinline__ float dequantize_int4(int8_t value, float scale, int32_t zero_point) {
    // 符号扩展
    int32_t signed_value = (value & 0x08) ? (value | 0xFFFFFFF0) : (value & 0x0F);
    return (float)(signed_value - zero_point) * scale;
}

// FP16转换
__device__ __forceinline__ half float_to_half(float value) {
    return __float2half(value);
}

__device__ __forceinline__ float half_to_float(half value) {
    return __half2float(value);
}

// 打包INT4到INT8 (两个INT4打包成一个INT8)
__device__ __forceinline__ int8_t pack_int4(int8_t low, int8_t high) {
    return (low & 0x0F) | ((high & 0x0F) << 4);
}

// 解包INT8到两个INT4
__device__ __forceinline__ void unpack_int4(int8_t packed, int8_t& low, int8_t& high) {
    low = packed & 0x0F;
    high = (packed >> 4) & 0x0F;
    // 符号扩展
    if (low & 0x08) low |= 0xF0;
    if (high & 0x08) high |= 0xF0;
}
