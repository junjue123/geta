"""MobileNetV1 in PyTorch
[1] Andrew G. Howard, Menglong Zhu, Bo Chen, et al.
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
Referenced ResNet implementation style: modular block + configurable structure
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution Block (MobileNetV1 core)
    Consists of: Depthwise Conv (grouped conv) + BatchNorm + ReLU + Pointwise Conv + BatchNorm + ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1, dw_config=None):
        super().__init__()

        if dw_config is None:
            # Default MobileNetV1 config
            self.dw_conv = nn.Sequential(
                # Depthwise: group=in_channels, each group is convolved with 1 kernel
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            # Pointwise: 1x1 conv to combine channels
            self.pw_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Support custom config (matching ResNet's cfg style)
            dw_ch = dw_config[0]
            pw_ch = dw_config[1]
            self.dw_conv = nn.Sequential(
                nn.Conv2d(in_channels, dw_ch, kernel_size=3, stride=stride,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(dw_ch),
                nn.ReLU(inplace=True)
            )
            self.pw_conv = nn.Sequential(
                nn.Conv2d(dw_ch, pw_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(pw_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNetV1(nn.Module):
    """MobileNetV1 implementation (adapted for CIFAR dataset)
    Structure: Conv1 (3x3) + 13 DepthwiseSeparable blocks + AvgPool + FC
    """

    def __init__(self, num_classes=10, cfg=None):
        super().__init__()
        self.config = cfg

        # Default MobileNetV1 channel/config setting (adapted for CIFAR)
        default_channels = [
            (32, 64),  # Block 1: dw=32, pw=64, stride=1
            (64, 128),  # Block 2: dw=64, pw=128, stride=2
            (128, 128),  # Block 3: dw=128, pw=128, stride=1
            (128, 256),  # Block 4: dw=128, pw=256, stride=2
            (256, 256),  # Block 5: dw=256, pw=256, stride=1
            (256, 512),  # Block 6: dw=256, pw=512, stride=2
            (512, 512),  # Block 7: dw=512, pw=512, stride=1
            (512, 512),  # Block 8: dw=512, pw=512, stride=1
            (512, 512),  # Block 9: dw=512, pw=512, stride=1
            (512, 512),  # Block 10: dw=512, pw=512, stride=1
            (512, 1024),  # Block 11: dw=512, pw=1024, stride=2
            (1024, 1024),  # Block 12: dw=1024, pw=1024, stride=1
        ]
        default_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1]

        if self.config is None:
            # Use default config
            self.in_channels = 3
            # Initial Conv layer (3x3, stride=1 for CIFAR, original MobileNetV1 uses stride=2 for ImageNet)
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.in_channels = 32
            # Build depthwise separable blocks
            self.dw_blocks = self._make_layers(default_channels, default_strides)
            # Final layers
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, num_classes)
        else:
            # Use custom config (format: [init_ch, [(dw_ch1,pw_ch1), (dw_ch2,pw_ch2), ...], final_ch])
            assert len(self.config) == 3, "cfg format: [init_ch, block_configs, final_ch]"
            init_ch = self.config[0]
            block_configs = self.config[1]
            final_ch = self.config[2]
            assert len(block_configs) == len(default_strides), "block_configs length must match default strides (12)"

            self.in_channels = 3
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, init_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(init_ch),
                nn.ReLU(inplace=True)
            )
            self.in_channels = init_ch
            self.dw_blocks = self._make_layers(block_configs, default_strides, use_cfg=True)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(final_ch, num_classes)

    def _make_layers(self, channels, strides, use_cfg=False):
        """Build depthwise separable layers
        Args:
            channels: list of (dw_ch, pw_ch) tuples
            strides: list of stride for each block
            use_cfg: whether using custom config
        """
        layers = []
        for idx, (ch, stride) in enumerate(zip(channels, strides)):
            if use_cfg:
                layers.append(DepthwiseSeparableConv(self.in_channels, None, stride, dw_config=ch))
                self.in_channels = ch[1]  # pw_ch is output channel
            else:
                dw_ch, pw_ch = ch
                layers.append(DepthwiseSeparableConv(self.in_channels, pw_ch, stride))
                self.in_channels = pw_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_blocks(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenetv1_cifar10(cfg=None):
    """Return MobileNetV1 object adapted for CIFAR-10 dataset"""
    return MobileNetV1(num_classes=10, cfg=cfg)

def mobilenetv1_cifar100(cfg=None):
    """Return MobileNetV1 object adapted for CIFAR-10 dataset"""
    return MobileNetV1(num_classes=100, cfg=cfg)


# Test code (verify model structure)
if __name__ == "__main__":
    model = mobilenetv1_cifar10()
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 input shape: (batch, 3, 32, 32)
    output = model(x)
    print(f"Model output shape: {output.shape}")  # Should be (2, 10)
    print(f"Model structure:\n{model}")