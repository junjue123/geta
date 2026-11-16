"""MobileNetV2 in PyTorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, et al.
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
Adapted to CIFAR dataset and matching previous code style
"""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2 core)
    Structure:
    Input -> 1x1 Conv (expand) -> BatchNorm -> ReLU6 ->
    3x3 DW Conv (depthwise) -> BatchNorm -> ReLU6 ->
    1x1 Conv (project) -> BatchNorm (no activation) + Residual Connection
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, inv_config=None):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        if inv_config is None:
            # Default expansion: t * in_channels (t is expand_ratio)
            expand_channels = in_channels * expand_ratio

            # 1x1 pointwise conv (expand)
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True)
            ) if expand_ratio != 1 else nn.Identity()  # No expansion if t=1

            # 3x3 depthwise conv (groups=expand_channels)
            self.depthwise = nn.Sequential(
                nn.Conv2d(expand_channels, expand_channels, kernel_size=3,
                          stride=stride, padding=1, groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True)
            )

            # 1x1 pointwise conv (project) with linear activation
            self.project = nn.Sequential(
                nn.Conv2d(expand_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
                # No activation here (linear bottleneck)
            )
        else:
            # Custom config: [expand_channels, depthwise_channels, project_channels]
            expand_ch, dw_ch, proj_ch = inv_config
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expand_ch, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_ch),
                nn.ReLU6(inplace=True)
            ) if expand_ratio != 1 else nn.Identity()

            self.depthwise = nn.Sequential(
                nn.Conv2d(expand_ch, dw_ch, kernel_size=3,
                          stride=stride, padding=1, groups=dw_ch, bias=False),
                nn.BatchNorm2d(dw_ch),
                nn.ReLU6(inplace=True)
            )

            self.project = nn.Sequential(
                nn.Conv2d(dw_ch, proj_ch, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(proj_ch)
            )
            self.use_residual = (stride == 1) and (in_channels == proj_ch)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x += residual
        return x


class MobileNetV2(nn.Module):
    """MobileNetV2 implementation (adapted for CIFAR)
    Architecture reference: Table 2 in original paper + CIFAR input adaptation
    """

    def __init__(self, num_classes=10, cfg=None):
        super().__init__()
        self.config = cfg

        # Default MobileNetV2 settings (adapted for CIFAR)
        # Each block: (in_channels, out_channels, stride, expand_ratio, repeat)
        default_blocks = [
            (3, 16, 1, 1, 1),  # Initial conv
            (16, 24, 1, 6, 2),  # Block 1 (stride 1 for CIFAR)
            (24, 32, 2, 6, 3),  # Block 2
            (32, 64, 2, 6, 4),  # Block 3
            (64, 96, 1, 6, 3),  # Block 4
            (96, 160, 2, 6, 3),  # Block 5
            (160, 320, 1, 6, 1),  # Block 6
            (320, 1280, 1, 6, 1)  # Final expansion
        ]

        if self.config is None:
            # Build with default config
            layers = []
            # Initial conv (first block in default_blocks)
            in_ch, out_ch, stride, exp_ratio, _ = default_blocks[0]
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True)
            ))

            # Build inverted residual blocks
            in_channels = out_ch
            for idx in range(1, len(default_blocks)):
                in_ch, out_ch, stride, exp_ratio, repeat = default_blocks[idx]
                for i in range(repeat):
                    # First block in repeat uses stride, others use 1
                    block_stride = stride if i == 0 else 1
                    layers.append(InvertedResidual(
                        in_channels, out_ch, block_stride, exp_ratio
                    ))
                    in_channels = out_ch

            # Final layers (修复：conv_last输入通道改为in_channels)
            self.features = nn.Sequential(*layers)
            self.conv_last = nn.Sequential(
                nn.Conv2d(in_channels, 1280, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True)
            )
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1280, num_classes)
        else:
            # Custom config (matching previous style)
            # Format: [init_conv_out, [block_configs...], final_conv_out, num_classes]
            init_out = self.config[0]
            block_configs = self.config[1]
            final_out = self.config[2]

            layers = []
            # Initial conv
            layers.append(nn.Sequential(
                nn.Conv2d(3, init_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(init_out),
                nn.ReLU6(inplace=True)
            ))

            # Build inverted residual blocks from config
            in_channels = init_out
            for block in block_configs:
                # block: (out_ch, stride, exp_ratio, repeat, [inv_config])
                out_ch, stride, exp_ratio, repeat, inv_cfg = block
                for i in range(repeat):
                    block_stride = stride if i == 0 else 1
                    layers.append(InvertedResidual(
                        in_channels, out_ch, block_stride, exp_ratio, inv_config=inv_cfg[i]
                    ))
                    in_channels = out_ch

            self.features = nn.Sequential(*layers)
            self.conv_last = nn.Sequential(
                nn.Conv2d(in_channels, final_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(final_out),
                nn.ReLU6(inplace=True)
            )
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(final_out, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenetv2_cifar10(cfg=None):
    """Return MobileNetV2 object adapted for CIFAR-10 dataset"""
    return MobileNetV2(num_classes=10, cfg=cfg)


def mobilenetv2_cifar100(cfg=None):
    """Return MobileNetV2 object adapted for CIFAR-100 dataset"""  # 修复注释错误
    return MobileNetV2(num_classes=100, cfg=cfg)


# Test code
if __name__ == "__main__":
    # 测试CIFAR-10模型
    model_cifar10 = mobilenetv2_cifar10()
    x_cifar10 = torch.randn(2, 3, 32, 32)  # CIFAR-10输入形状
    output_cifar10 = model_cifar10(x_cifar10)
    print(f"CIFAR-10 Model Output shape: {output_cifar10.shape}")  # 预期输出 (2, 10)

    # 测试CIFAR-100模型
    model_cifar100 = mobilenetv2_cifar100()
    x_cifar100 = torch.randn(2, 3, 32, 32)  # CIFAR-100输入形状
    output_cifar100 = model_cifar100(x_cifar100)
    print(f"CIFAR-100 Model Output shape: {output_cifar100.shape}")  # 预期输出 (2, 100)

    print("Model test passed!")