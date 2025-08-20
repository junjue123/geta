
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Bottleneck(nn.Module):
    expansion = 4  # ResNet50使用的bottleneck结构expansion为4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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


class SharedFirstLayer(nn.Module):
    def __init__(self):
        super(SharedFirstLayer, self).__init__()
        # 与ResNet20原第一层完全一致：3x3卷积，输出16通道
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.apply(_weights_init)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))


class ModifiedResNet18(nn.Module):
    def __init__(self, block, num_blocks, shared_first_layer, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        self.shared_first = shared_first_layer  # 共享第一层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 新增1x1卷积层：将16通道转换为64通道（ResNet18原初始通道数）
        self.channel_up = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.in_channels = 64  # 转换后的通道数

        # 后续残差块保持ResNet18原有结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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
        # 共享第一层
        out = self.shared_first(x)  # 输出16通道

        # 通道数转换：16→64
        out = self.channel_up(out)

        # 保留原ResNet18的maxpool
        out = self.maxpool(out)

        # 后续残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        if feature_need:
            feature = out
            out = self.linear(out)
            return out, feature
        else:
            out = self.linear(out)
            return out


class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avg_pool2d = nn.AvgPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.linear = nn.Linear(1024, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, feature_need=False):
        out = F.relu(self.bn1(self.conv1(x)))  # [2, 3, 72, 72] -> [2, 16, 72, 72]
        out = self.layer1(out)  # 72 -> 72
        out = self.layer2(out)  # 72 -> 36
        out = self.layer3(out)  # 36 -> 18
        out = self.avg_pool2d(out)  # 18 -> 1
        out = out.view(out.size(0), -1)  # [64, 1024]
        if feature_need:
            feature = out
            out = self.linear(out)  # [64, 10]
            return out,feature
        else:
            out = self.linear(out)
            return out


class ModifiedResNet50(nn.Module):
    def __init__(self, block, num_blocks, shared_first_layer, num_classes=10):
        super(ModifiedResNet50, self).__init__()
        self.shared_first = shared_first_layer  # 共享第一层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 1x1卷积层：将16通道转换为64通道（ResNet50原初始通道数）
        self.channel_up = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.in_channels = 64  # 转换后的通道数

        # 四个残差层，保持ResNet50原有结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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
        # 共享第一层，输出16通道
        out = self.shared_first(x)

        # 通道数转换：16→64
        out = self.channel_up(out)

        # 保留原ResNet50的maxpool
        out = self.maxpool(out)

        # 后续残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        if feature_need:
            feature = out
            out = self.linear(out)
            return out, feature
        else:
            out = self.linear(out)
            return out


def resnet18_MSTAR():
    shared_first = SharedFirstLayer()
    return ModifiedResNet18(BasicBlock, [2,2,2,2], shared_first_layer=shared_first)
def resnet34_MSTAR():
    shared_first = SharedFirstLayer()
    return ModifiedResNet18(BasicBlock, [3, 4, 6, 3], shared_first_layer=shared_first)
def resnet50_MSTAR():
    """返回适用于MSTAR的ResNet50，与ResNet18共享第一层"""
    shared_first = SharedFirstLayer()
    return ModifiedResNet50(Bottleneck, [3, 4, 6, 3], shared_first_layer=shared_first)
