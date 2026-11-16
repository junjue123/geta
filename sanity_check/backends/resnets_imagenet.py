import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  # for weight initialization.


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    """基础残差块，用于ResNet18/34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 维度不匹配时，通过1x1卷积调整通道数和步长
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """瓶颈残差块，用于ResNet50/101/152"""
    expansion = 4  # 输出通道数是输入的4倍

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        # 维度不匹配时，通过1x1卷积调整
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """通用ResNet框架，适配ImageNet"""

    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # ImageNet输入为3x224x224，第一层卷积核7x7、步长2，适配大尺寸输入
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样

        # 构建4个残差层（ResNet20是3层，ImageNet版是4层）
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化，适配任意尺寸特征图（ImageNet最终特征图是7x7）
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层：512*expansion -> 1000（ImageNet类别数）
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_need=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)

        if feature_need:
            feature = out
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out, feature
        else:
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


# ------------------- 实例化函数 -------------------
def resnet18_imagenet():
    """返回适配ImageNet的ResNet18实例
    ResNet18: BasicBlock + [2,2,2,2]
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet50_imagenet():
    """返回适配ImageNet的ResNet50实例
    ResNet50: Bottleneck + [3,4,6,3]
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])


"""
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
[3] https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""