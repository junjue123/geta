import numpy as np
import torch
from torch import nn
from torch.nn import init
import time


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

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

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


# 1. 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2. 手动计算ExternalAttention的FLOPs（包含维度变换）
def calculate_externalatt_flops(input_shape, S=8):
    """
    计算ExternalAttention的FLOPs（包含维度变换）
    input_shape: (batch_size, d_model, height, width)
    """
    batch_size, d_model, height, width = input_shape
    n = height * width  # 展平后的序列长度

    # 1. 维度变换（view操作）的FLOPs：主要是内存重组，按1次操作/元素估算
    flatten_flops = batch_size * d_model * height * width  # 展平操作
    reshape_flops = batch_size * d_model * height * width  # 恢复形状操作
    dim_transform_flops = flatten_flops + reshape_flops

    # 2. mk线性层的FLOPs：每个输出元素需要d_model次乘法和(d_model-1)次加法
    mk_flops = batch_size * n * S * (d_model + (d_model - 1))

    # 3. Softmax的FLOPs：每个元素需要指数运算+求和+除法
    softmax_flops = batch_size * n * S * 3  # 简化估算

    # 4. 归一化操作的FLOPs：求和+除法
    norm_flops = batch_size * n * (S + 1)  # sum操作S次加法，除法1次/元素

    # 5. mv线性层的FLOPs：每个输出元素需要S次乘法和(S-1)次加法
    mv_flops = batch_size * n * d_model * (S + (S - 1))

    # 总FLOPs
    total_flops = dim_transform_flops + mk_flops + softmax_flops + norm_flops + mv_flops
    return total_flops


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和输入
    input_shape = (16, 16, 128, 128)  # (batch_size, d_model, height, width)
    img_input = torch.randn(input_shape).to(device)
    batch_size, d_model, height, width = img_input.shape
    ea = ExternalAttention(d_model=d_model, S=8).to(device)

    # 1. 计算模型参数量
    params = count_parameters(ea)
    print(f"\n模型参数量: {params} 个")
    print(f"模型参数量: {params / 1e6:.6f} M")

    # 2. 计算FLOPs
    flops = calculate_externalatt_flops(input_shape, S=8)
    print(f"FLOPs: {flops} 次")
    print(f"FLOPs: {flops / 1e6:.2f} M")

    # 3. 计算处理速度（包含维度变换）
    # 预热运行
    for _ in range(10):
        # 包含维度变换的完整流程
        img_flattened = img_input.view(batch_size, height * width, d_model)
        img_output = ea(img_flattened)
        img_output = img_output.view(batch_size, d_model, height, width)

    # 正式测量
    start_time = time.time()
    iterations = 1000  # 增加迭代次数以获得更稳定的测量结果
    for _ in range(iterations):
        # 包含维度变换的完整流程
        img_flattened = img_input.view(batch_size, height * width, d_model)
        img_output = ea(img_flattened)
        img_output = img_output.view(batch_size, d_model, height, width)

        if device.type == "cuda":
            torch.cuda.synchronize()  # 确保CUDA操作完成

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n处理{iterations}次的总时间: {total_time:.4f}秒")
    print(f"单次处理时间: {total_time / iterations:.6f}秒")
    print(f"每秒处理次数: {iterations / total_time:.2f}次")

    # 验证输出形状
    print(f"\n输入形状: {img_input.shape}")
    print(f"输出形状: {img_output.shape}")