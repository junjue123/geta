import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
import cv2
from torchvision.models import resnet34, resnet50
# from network.attention import Attention0
from sccov import GroupCBAMEnhancer as GCE


class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSC, self).__init__()
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子，用于缩放注意力分数

        # 定义Q、K、V的线性变换
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 查询向量Q的线性投影
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 键K和值V的线性投影（合并）

        # 定义dropout层
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力分数的dropout
        self.proj = nn.Linear(dim, dim)  # 输出的线性投影
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 输出的dropout

        # 用于控制top-k稀疏度的参数
        self.k1 = k1  # 第一个注意力图的稀疏度控制参数
        self.k2 = k2  # 第二个注意力图的稀疏度控制参数

        # 注意力图融合的可学习权重
        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)  # 第一个注意力图的权重
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)  # 第二个注意力图的权重

        # 多尺度平均池化层，用于提取不同尺度的特征
        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        # 层归一化
        self.layer_norm = nn.LayerNorm(dim)

        self.topk = topk  # 是否使用top-k稀疏注意力机制

    def forward(self, x, y):
        # x: 通常是高层特征图，作为查询Q的来源 (b, c, h, w)
        # y: 通常是低层特征图，作为键K和值V的来源 (b, c, h, w)

        # 1. 对输入特征y进行多尺度平均池化
        y1 = self.avgpool1(y)  # 第一个尺度的池化
        y2 = self.avgpool2(y)  # 第二个尺度的池化
        y3 = self.avgpool3(y)  # 第三个尺度的池化

        # 2. 多尺度特征融合（加法融合）
        y = y1 + y2 + y3  # 不同尺度特征相加融合

        # 3. 特征图展平：将空间维度(h,w)展平为一个维度
        y = y.flatten(-2, -1)  # 形状变为 (b, c, n1)，n1 = h1*w1

        # 4. 维度转换和归一化
        y = y.transpose(1, 2)  # 转换为 (b, n1, c)，便于后续线性变换
        y = self.layer_norm(y)  # 对y进行层归一化

        # 5. 对x进行形状转换：将特征图转换为序列形式
        x = rearrange(x, 'b c h w -> b (h w) c')  # 形状变为 (b, n, c)，n = h*w

        # 6. 提取键K和值V（来自y）
        B, N1, C = y.shape  # B:批次大小, N1:y的序列长度, C:通道数
        # 对y进行线性变换并拆分出k和v
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # k: (b, num_heads, n1, head_dim), v: (b, num_heads, n1, head_dim)

        # 7. 提取查询Q（来自x）
        B, N, C = x.shape  # N: x的序列长度
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q的形状: (b, num_heads, n, head_dim)

        # 8. 计算注意力分数
        # 矩阵乘法计算Q和K的相似度，再乘以缩放因子
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 形状: (b, num_heads, n, n1)

        # 9. 生成第一个稀疏注意力图（使用top-k机制）
        # 创建掩码矩阵，初始值全为0
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        # 选取top-(N1/k1)个注意力分数最高的位置
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]
        # 在选中的位置设置掩码为1
        mask1.scatter_(-1, index, 1.)
        # 仅保留掩码为1的位置的注意力分数，其他位置设为负无穷（softmax后接近0）
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)  # 计算softmax得到注意力权重
        attn1 = self.attn_drop(attn1)  # 应用dropout
        out1 = (attn1 @ v)  # 注意力加权求和得到输出1，形状: (b, num_heads, n, head_dim)

        # 10. 生成第二个稀疏注意力图（不同稀疏度）
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        # 选取top-(N1/k2)个注意力分数最高的位置（k2 > k1，所以稀疏度更低）
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)  # 输出2，形状: (b, num_heads, n, head_dim)

        # 11. 融合两个注意力图的输出
        out = out1 * self.attn1 + out2 * self.attn2  # 加权融合，使用可学习权重

        # 12. 维度转换和线性投影
        x = out.transpose(1, 2).reshape(B, N, C)  # 形状变回 (b, n, c)
        x = self.proj(x)  # 线性投影，调整通道维度
        x = self.proj_drop(x)  # 应用dropout

        # 13. 形状转换：将序列形式转换回特征图形式
        hw = int(sqrt(N))  # 计算空间维度大小（假设h=w）
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)  # 形状变回 (b, c, h, w)

        return x


class MSCN(nn.Module):
    def __init__(self, num_classes=45, res=50, k1=2, k2=3, g=8):
        super(MSCN, self).__init__()
        # print(f'att = {att}')
        if res == 34:
            self.resnet = resnet34()
            dim = [64, 128, 256, 512]
            self.dim = dim
        elif res == 50:
            self.resnet = resnet50()
            dim = [256, 512, 1024, 2048]
            self.dim = dim
        dim_b = self.dim[1]
        dim_fc = self.dim[-1]

        self.msc1 = MSC(dim=dim_b, kernel=[3, 5, 7], pad=[1, 2, 3], k1=k1, k2=k2)
        self.msc2 = MSC(dim=dim_b, kernel=[3, 5, 7], pad=[1, 2, 3], k1=k1, k2=k2)
        self.msc3 = MSC(dim=dim_b, kernel=[3, 5, 7], pad=[1, 2, 3], k1=k1, k2=k2)

        self.gce = GCE(dim_fc, g, 0, 0)

        self.cov1 = nn.Conv2d(dim[0], dim_b, 3, 2, 1)
        self.cov2 = nn.Conv2d(dim[1], dim_b, 3, 2, 1)
        self.cov3 = nn.Conv2d(dim[2], dim_b, 3, 2, 1)
        self.cov4 = nn.Conv2d(dim[3], dim_b, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim_fc, num_classes)

    def reset_head(self, num_classes):
        self.fc = nn.Linear(self.dim[-1], num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x1 = self.cov1(x)
        x = self.resnet.layer2(x)
        x2 = self.cov2(x)
        x = self.resnet.layer3(x)
        x3 = self.cov3(x)
        x = self.resnet.layer4(x)
        x4 = self.cov4(x)

        x1 = self.msc1(x4, x1)
        x2 = self.msc2(x4, x2)
        x3 = self.msc3(x4, x3)

        x = torch.cat([x4, x1, x2, x3], dim=1)
        x = self.gce(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    print('net test')
    a = torch.randn(1, 3, 224, 224)
    model = MSCN(res=34)
    # model = resnet50()
    # print(model.resnet.fc)
    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(model, a)
    # print("FLOPs: %.2fG" % (flops.total()/1e9))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    b = model(a)
    print(b.shape)