import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # for weight initialization.
from einops import rearrange
import math


class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=None, pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.k1 = k1
        self.k2 = k2

        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        # 自动调整步长参数s
        self.num_scales = len(kernel)

        # 如果未提供s或s的长度与kernel不匹配，则自动生成s
        if s is None or len(s) != self.num_scales:
            # 默认步长为1，与大多数池化操作保持一致
            s = [1] * self.num_scales
            print(f"Warning: Automatically setting stride s to {s} to match kernel length")

        # 校验参数长度一致性
        if not (len(kernel) == len(s) == len(pad)):
            raise ValueError("kernel, s, pad must have the same length")

        # 动态创建多尺度平均池化层
        self.avg_pools = nn.ModuleList()
        for k, st, p in zip(kernel, s, pad):
            self.avg_pools.append(nn.AvgPool2d(kernel_size=k, stride=st, padding=p))

        self.layer_norm = nn.LayerNorm(dim)
        self.topk = topk

    def forward(self, x, y):
        # x: (b, c, h, w)，y: (b, c, h, w)

        # 对y进行多尺度池化
        multi_scale_ys = []
        for pool in self.avg_pools:
            y_pooled = pool(y)
            multi_scale_ys.append(y_pooled)

        # 多尺度特征融合
        y = torch.stack(multi_scale_ys, dim=0).sum(dim=0)

        y = y.flatten(-2, -1)  # (b, c, n1)
        y = y.transpose(1, 2)  # (b, n1, c)
        y = self.layer_norm(y)

        x = rearrange(x, 'b c h w -> b (h w) c')  # (b, n, c)

        # 提取K、V（来自y）
        B, N1, C = y.shape
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (b, num_heads, n1, head_dim)

        # 提取Q（来自x）
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                 3)  # (b, num_heads, n, head_dim)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (b, num_heads, n, n1)

        # 生成第一个稀疏注意力图
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        # 生成第二个稀疏注意力图
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        # 融合输出
        out = out1 * self.attn1 + out2 * self.attn2

        # 输出处理
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 恢复特征图形状
        hw = int(math.sqrt(N))
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)

        return x



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
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        x = x * attention
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class GroupCBAMEnhancer(nn.Module):
    def __init__(self, channel, group=8, cov1=1, cov2=1):
        super().__init__()
        self.cov1 = None
        self.cov2 = None
        if cov1 != 0:
            self.cov1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.group = group
        cbam = []
        for i in range(self.group):
            cbam_ = CBAM(channel // group)
            cbam.append(cbam_)

        self.cbam = nn.ModuleList(cbam)
        self.sigomid = nn.Sigmoid()
        if cov2 != 0:
            self.cov2 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        x0 = x
        if self.cov1 != None:
            x = self.cov1(x)
        y = torch.split(x, x.size(1) // self.group, dim=1)
        mask = []
        for y_, cbam in zip(y, self.cbam):
            y_ = cbam(y_)
            y_ = self.sigomid(y_)

            mk = y_

            mean = torch.mean(y_, [1, 2, 3])
            mean = mean.view(-1, 1, 1, 1)

            # mean = torch.mean(y_,[2,3])
            # mean = mean.view(mean.size(0),mean.size(1),1,1)

            gate = torch.ones_like(y_) * mean
            mk = torch.where(y_ > gate, 1, y_)

            mask.append(mk)
        mask = torch.cat(mask, dim=1)
        # print(mask.shape)
        x = x * mask
        if self.cov2 != None:
            x = self.cov2(x)
        x = x + x0
        return x


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModifiedResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, k1=2, k2=3, g=8):
        super(ModifiedResNet20, self).__init__()
        self.in_channels = 16
        dim = [16, 32, 64, 96]
        self.dim = dim
        dim_b = self.dim[1]
        # dim_fc = self.dim[-1] #原来是针对四个通道相加的，这里应该修改一下
        dim_fc = 3*dim_b

        self.msc1 = MSC(dim=dim_b, kernel=[3, 5], pad=[1, 2], k1=k1, k2=k2)
        self.msc2 = MSC(dim=dim_b, kernel=[3, 5], pad=[1, 2], k1=k1, k2=k2)

        self.gce = GroupCBAMEnhancer(dim_fc, g, 0, 0)

        self.cov1 = nn.Conv2d(dim[0], dim_b, 3, 2, 1)
        self.cov2 = nn.Conv2d(dim[1], dim_b, 3, 2, 1)
        self.cov3 = nn.Conv2d(dim[2], dim_b, 1)

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(dim_fc, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv(x)))

        out = self.layer1(out)
        feature1 = self.cov1(out) #32

        out = self.layer2(out)
        feature2 = self.cov2(out) #32

        out = self.layer3(out)
        feature3 = self.cov3(out) #32

        feature1 = self.msc1(feature3, feature1)
        feature2 = self.msc2(feature3, feature2)

        out = torch.cat([feature3, feature1, feature2], dim=1) #96
        out = self.gce(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20_cifar10_MSCN():
    """ return a ResNet 20 object
    """
    return ModifiedResNet20(BasicBlock, [3, 3, 3])

if __name__ == '__main__':
    print('net test')
    a = torch.randn(1, 3, 32, 32)
    model = resnet20_cifar10_MSCN()

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    b = model(a)
    print(b.shape)