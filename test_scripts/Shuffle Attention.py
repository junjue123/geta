import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import time


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
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

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


# 1. 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2. 手动计算ShuffleAttention的FLOPs
def calculate_shuffleatt_flops(input_shape, G=8):
    """
    计算ShuffleAttention的FLOPs
    input_shape: (batch_size, channels, height, width)
    """
    batch_size, channel, h, w = input_shape

    # 计算中间通道数
    c_div_2g = channel // (2 * G)

    # 1. 分组和通道分割操作不涉及FLOPs

    # 2. 通道注意力FLOPs
    # 全局平均池化
    gap_flops = batch_size * G * c_div_2g * h * w
    # 权重和偏置操作
    cweight_flops = batch_size * G * c_div_2g * 1 * 1  # 乘法
    cbias_flops = batch_size * G * c_div_2g * 1 * 1  # 加法
    # Sigmoid激活
    sigmoid_c_flops = batch_size * G * c_div_2g * 1 * 1
    # 特征相乘
    multiply_c_flops = batch_size * G * c_div_2g * h * w
    channel_att_flops = gap_flops + cweight_flops + cbias_flops + sigmoid_c_flops + multiply_c_flops

    # 3. 空间注意力FLOPs
    # GroupNorm操作
    gn_flops = 2 * batch_size * G * c_div_2g * h * w  # 均值和方差计算
    # 权重和偏置操作
    sweight_flops = batch_size * G * c_div_2g * h * w  # 乘法
    sbias_flops = batch_size * G * c_div_2g * h * w  # 加法
    # Sigmoid激活
    sigmoid_s_flops = batch_size * G * c_div_2g * h * w
    # 特征相乘
    multiply_s_flops = batch_size * G * c_div_2g * h * w
    spatial_att_flops = gn_flops + sweight_flops + sbias_flops + sigmoid_s_flops + multiply_s_flops

    # 4. 通道 shuffle 操作不涉及FLOPs

    # 总FLOPs
    total_flops = channel_att_flops + spatial_att_flops
    return total_flops


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和输入
    input_shape = (16, 16, 128, 128)  # (batch_size, channels, height, width)
    input_tensor = torch.randn(input_shape).to(device)
    sa = ShuffleAttention(channel=16, G=8).to(device)

    # 1. 计算模型参数量
    params = count_parameters(sa)
    print(f"\n模型参数量: {params} 个")
    print(f"模型参数量: {params / 1e6:.6f} M")

    # 2. 计算FLOPs
    flops = calculate_shuffleatt_flops(input_shape)
    print(f"FLOPs: {flops} 次")
    print(f"FLOPs: {flops / 1e6:.2f} M")

    # 3. 计算处理速度
    # 预热运行
    for _ in range(10):
        output = sa(input_tensor)

    # 正式测量
    start_time = time.time()
    iterations = 1000  # 增加迭代次数以获得更稳定的测量结果
    for _ in range(iterations):
        output = sa(input_tensor)
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
