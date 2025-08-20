import numpy as np
import torch
from torch import nn
from torch.nn import init
import time


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 1. 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2. 手动计算SEAttention的FLOPs
def calculate_seatt_flops(input_shape, reduction=16):
    """
    计算SEAttention的FLOPs
    input_shape: (batch_size, channels, height, width)
    """
    batch_size, channel, height, width = input_shape
    mid_channel = channel // reduction

    # 1. 全局平均池化的FLOPs
    gap_flops = batch_size * channel * height * width

    # 2. 第一个全连接层的FLOPs
    # 每个输出元素需要 channel 次乘法和 (channel-1) 次加法
    fc1_flops = batch_size * mid_channel * (channel + (channel - 1))

    # 3. ReLU激活函数的FLOPs (每个元素一次比较操作)
    relu_flops = batch_size * mid_channel

    # 4. 第二个全连接层的FLOPs
    fc2_flops = batch_size * channel * (mid_channel + (mid_channel - 1))

    # 5. Sigmoid激活函数的FLOPs
    sigmoid_flops = batch_size * channel

    # 6. 特征图相乘的FLOPs
    multiply_flops = batch_size * channel * height * width

    # 7. 维度变换(view操作)的FLOPs
    view_flops = 2 * batch_size * channel  # 两次view操作

    # 总FLOPs
    total_flops = gap_flops + fc1_flops + relu_flops + fc2_flops + \
                  sigmoid_flops + multiply_flops + view_flops
    return total_flops


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和输入
    input_shape = (16, 16, 128, 128)  # (batch_size, channels, height, width)
    input_tensor = torch.randn(input_shape).to(device)
    se = SEAttention(channel=16, reduction=8).to(device)

    # 1. 计算模型参数量
    params = count_parameters(se)
    print(f"\n模型参数量: {params} 个")
    print(f"模型参数量: {params / 1e6:.6f} M")

    # 2. 计算FLOPs
    flops = calculate_seatt_flops(input_shape, reduction=8)
    print(f"FLOPs: {flops} 次")
    print(f"FLOPs: {flops / 1e6:.2f} M")

    # 3. 计算处理速度
    # 预热运行
    for _ in range(10):
        output = se(input_tensor)

    # 正式测量
    start_time = time.time()
    iterations = 1000  # 增加迭代次数以获得更稳定的测量结果
    for _ in range(iterations):
        output = se(input_tensor)
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
