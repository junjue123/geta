import numpy as np
import torch
from torch import nn
from torch.nn import init
import time


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

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
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


# 1. 手动计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2. 手动计算FLOPs
def calculate_eca_flops(input_shape, kernel_size=3):
    """
    手动计算ECAAttention的FLOPs
    input_shape: (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = input_shape

    # 全局平均池化的FLOPs: 每个通道每个空间位置求和取平均
    gap_flops = batch_size * channels * height * width

    # 1D卷积的FLOPs: 每个输出元素需要 kernel_size 次乘法和加法
    conv_flops = batch_size * channels * kernel_size

    # 特征图相乘的FLOPs: 每个元素一次乘法
    multiply_flops = batch_size * channels * height * width

    total_flops = gap_flops + conv_flops + multiply_flops
    return total_flops


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和输入
    input_shape = (16, 16, 128, 128)  # (batch_size, channels, height, width)
    input_tensor = torch.randn(input_shape).to(device)
    eca = ECAAttention(kernel_size=3).to(device)

    # 1. 计算模型参数量
    params = count_parameters(eca)
    print(f"\n模型参数量: {params} 个")
    print(f"模型参数量: {params / 1e6:.6f} M")

    # 2. 计算FLOPs
    flops = calculate_eca_flops(input_shape)
    print(f"FLOPs: {flops} 次")
    print(f"FLOPs: {flops / 1e6:.2f} M")

    # 3. 计算处理速度
    # 预热运行
    for _ in range(10):
        output = eca(input_tensor)

    # 正式测量
    start_time = time.time()
    iterations = 1000  # 增加迭代次数以获得更稳定的测量结果
    for _ in range(iterations):
        output = eca(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()  # 确保CUDA操作完成

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n处理{iterations}次的总时间: {total_time:.4f}秒")
    print(f"单次处理时间: {total_time / iterations:.6f}秒")
    print(f"每秒处理次数: {iterations / total_time:.2f}次")

    # 验证输出形状
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
