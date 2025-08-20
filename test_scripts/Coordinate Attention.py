import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# 1. 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2. 手动计算CoordAtt的FLOPs
def calculate_coordatt_flops(input_shape, reduction=32):
    """
    计算CoordAtt的FLOPs
    input_shape: (batch_size, channels, height, width)
    """
    batch_size, inp, h, w = input_shape

    # 计算中间通道数
    mip = max(8, inp // reduction)

    # 1. 池化操作的FLOPs
    # 高度方向池化
    pool_h_flops = batch_size * inp * h * w
    # 宽度方向池化
    pool_w_flops = batch_size * inp * h * w
    pool_total = pool_h_flops + pool_w_flops

    # 2. 卷积层1的FLOPs (1x1卷积)
    # 每个输出元素需要 inp 次乘法和加法
    conv1_flops = batch_size * mip * (h + w) * 1 * inp

    # 3. 批归一化层的FLOPs
    bn_flops = batch_size * mip * (h + w) * 2  # 均值和方差计算

    # 4. h_swish激活函数的FLOPs
    # 每个元素需要几次操作
    act_flops = batch_size * mip * (h + w) * 3  # 简化计算

    # 5. 两个卷积层的FLOPs (conv_h和conv_w)
    conv_h_flops = batch_size * inp * h * 1 * mip
    conv_w_flops = batch_size * inp * w * 1 * mip
    conv_total = conv_h_flops + conv_w_flops

    # 6. Sigmoid激活函数的FLOPs
    sigmoid_flops = batch_size * inp * (h + w)  # 简化计算

    # 7. 特征图相乘的FLOPs
    multiply_flops = 2 * batch_size * inp * h * w  # 两次乘法

    # 总FLOPs
    total_flops = pool_total + conv1_flops + bn_flops + act_flops + \
                  conv_total + sigmoid_flops + multiply_flops

    return total_flops


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和输入
    input_shape = (16, 16, 128, 128)  # (batch_size, channels, height, width)
    input_tensor = torch.randn(input_shape).to(device)
    ca = CoordAtt(inp=16, oup=16, reduction=32).to(device)

    # 1. 计算模型参数量
    params = count_parameters(ca)
    print(f"\n模型参数量: {params} 个")
    print(f"模型参数量: {params / 1e6:.6f} M")

    # 2. 计算FLOPs
    flops = calculate_coordatt_flops(input_shape)
    print(f"FLOPs: {flops} 次")
    print(f"FLOPs: {flops / 1e6:.2f} M")

    # 3. 计算处理速度
    # 预热运行
    for _ in range(10):
        output = ca(input_tensor)

    # 正式测量
    start_time = time.time()
    iterations = 1000  # 增加迭代次数以获得更稳定的测量结果
    for _ in range(iterations):
        output = ca(input_tensor)
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
