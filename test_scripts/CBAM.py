import numpy as np
import torch
from torch import nn
from torch.nn import init
import time


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    # 空间注意力模块，使用较小的卷积核保证尺寸一致
    def __init__(self, kernel_size=7):
        super().__init__()
        # 确保padding设置正确，使输入输出尺寸一致
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):  # 修正默认kernel_size
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)  # 空间注意力使用小卷积核

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)  # 通道注意力
        out = out * self.sa(out)  # 空间注意力
        return out + residual


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_cbam_flops(input_shape, reduction=16, kernel_size=7):
    batch_size, channel, height, width = input_shape
    mid_channel = channel // reduction

    # 通道注意力部分
    pool_flops = 2 * batch_size * channel * height * width
    conv1_flops = 2 * batch_size * mid_channel * channel * 1 * 1
    relu_flops = 2 * batch_size * mid_channel * 1 * 1
    conv2_flops = 2 * batch_size * channel * mid_channel * 1 * 1
    add_flops = batch_size * channel * 1 * 1
    sigmoid_ca_flops = batch_size * channel * 1 * 1
    ca_total = pool_flops + conv1_flops + relu_flops + conv2_flops + add_flops + sigmoid_ca_flops

    # 空间注意力部分
    channel_op_flops = 2 * batch_size * height * width
    conv_sa_flops = batch_size * 1 * 2 * kernel_size * kernel_size * height * width
    sigmoid_sa_flops = batch_size * 1 * height * width
    sa_total = channel_op_flops + conv_sa_flops + sigmoid_sa_flops

    # 整体操作
    multiply_flops = 2 * batch_size * channel * height * width
    add_residual_flops = batch_size * channel * height * width
    total_flops = ca_total + sa_total + multiply_flops + add_residual_flops
    return total_flops


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (16, 16, 128, 128)
    input_tensor = torch.randn(input_shape).to(device)

    # 关键修复：使用小卷积核（如7）而非输入特征图的尺寸
    cbam = CBAMBlock(channel=16, reduction=16, kernel_size=7).to(device)

    # 参数量
    params = count_parameters(cbam)
    print(f"参数量: {params} 个 ({params / 1e6:.6f} M)")

    # FLOPs
    flops = calculate_cbam_flops(input_shape, reduction=16, kernel_size=7)
    print(f"FLOPs: {flops} 次 ({flops / 1e6:.2f} M)")

    # 处理速度
    for _ in range(10):
        output = cbam(input_tensor)

    start_time = time.time()
    iterations = 1000
    for _ in range(iterations):
        output = cbam(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_time = time.time() - start_time
    print(f"总时间: {total_time:.4f}秒")
    print(f"单次时间: {total_time / iterations:.6f}秒")
    print(f"每秒处理次数: {iterations / total_time:.2f}次")
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
