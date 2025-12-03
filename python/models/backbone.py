"""
Tesla FSD Backbone Networks
- RegNet: Primary backbone used by Tesla
- EfficientNet: Alternative backbone for comparison

The backbone extracts multi-scale features from camera images.
Tesla uses a modified RegNet architecture optimized for their custom HW3/HW4 chips.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RegNetBlock(nn.Module):
    """
    RegNet X/Y block with optional SE attention
    Tesla uses RegNet-Y style with SE blocks for better feature extraction
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        group_width: int = 1,
        se_ratio: float = 0.25,
        use_se: bool = True
    ):
        super().__init__()

        groups = out_channels // group_width

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, int(1 / se_ratio)) if use_se else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out


class RegNetBackbone(nn.Module):
    """
    RegNet Backbone - Tesla's primary vision backbone

    Produces multi-scale feature maps at 1/4, 1/8, 1/16, 1/32 resolution
    for use by detection heads and BEV transformer.

    Architecture based on "Designing Network Design Spaces" (FAIR)
    Modified for autonomous driving applications.
    """

    # RegNet-Y configurations (width, depth, group_width)
    CONFIGS = {
        'regnet_y_400mf': {'widths': [32, 64, 160, 384], 'depths': [1, 2, 7, 12], 'group_width': 8},
        'regnet_y_800mf': {'widths': [64, 128, 320, 768], 'depths': [1, 2, 8, 14], 'group_width': 16},
        'regnet_y_1.6gf': {'widths': [48, 120, 336, 888], 'depths': [2, 4, 10, 18], 'group_width': 24},
        'regnet_y_3.2gf': {'widths': [72, 216, 576, 1512], 'depths': [2, 5, 13, 25], 'group_width': 24},
        'regnet_y_8gf': {'widths': [96, 192, 432, 1088], 'depths': [2, 6, 17, 32], 'group_width': 16},
        # Tesla likely uses a custom variant around 4-8GF
        'tesla_custom': {'widths': [80, 192, 480, 1024], 'depths': [2, 5, 14, 24], 'group_width': 32},
    }

    def __init__(
        self,
        config_name: str = 'tesla_custom',
        in_channels: int = 3,
        use_se: bool = True,
        pretrained: bool = False
    ):
        super().__init__()

        config = self.CONFIGS[config_name]
        widths = config['widths']
        depths = config['depths']
        group_width = config['group_width']

        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Build stages
        self.stage1 = self._make_stage(32, widths[0], depths[0], stride=2, group_width=group_width, use_se=use_se)
        self.stage2 = self._make_stage(widths[0], widths[1], depths[1], stride=2, group_width=group_width, use_se=use_se)
        self.stage3 = self._make_stage(widths[1], widths[2], depths[2], stride=2, group_width=group_width, use_se=use_se)
        self.stage4 = self._make_stage(widths[2], widths[3], depths[3], stride=2, group_width=group_width, use_se=use_se)

        # Output channels for each stage (for FPN/detection heads)
        self.out_channels = widths

        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        group_width: int,
        use_se: bool
    ) -> nn.Sequential:
        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            c_in = in_channels if i == 0 else out_channels
            layers.append(RegNetBlock(c_in, out_channels, s, group_width, use_se=use_se))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Dictionary of feature maps at different scales
            - 'p2': 1/4 resolution
            - 'p3': 1/8 resolution
            - 'p4': 1/16 resolution
            - 'p5': 1/32 resolution
        """
        x = self.stem(x)  # 1/2

        c2 = self.stage1(x)   # 1/4
        c3 = self.stage2(c2)  # 1/8
        c4 = self.stage3(c3)  # 1/16
        c5 = self.stage4(c4)  # 1/32

        return {
            'p2': c2,
            'p3': c3,
            'p4': c4,
            'p5': c5
        }


class EfficientNetBlock(nn.Module):
    """MBConv block for EfficientNet"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25
    ):
        super().__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # SE block
        layers.append(SEBlock(hidden_dim, int(1 / se_ratio)))

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet Backbone - Alternative to RegNet

    Based on "EfficientNet: Rethinking Model Scaling" (Google)
    Provides a good balance of accuracy and efficiency.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0
    ):
        super().__init__()

        def scale_width(w): return int(w * width_mult)
        def scale_depth(d): return int(math.ceil(d * depth_mult))

        # EfficientNet-B0 base configuration
        # [expand_ratio, channels, repeats, stride, kernel]
        config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, scale_width(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(scale_width(32)),
            nn.SiLU(inplace=True)
        )

        # Build stages
        stages = []
        in_ch = scale_width(32)

        for expand, out_ch, repeats, stride, kernel in config:
            out_ch = scale_width(out_ch)
            repeats = scale_depth(repeats)

            blocks = []
            for i in range(repeats):
                s = stride if i == 0 else 1
                blocks.append(EfficientNetBlock(in_ch, out_ch, kernel, s, expand))
                in_ch = out_ch

            stages.append(nn.Sequential(*blocks))

        self.stage1 = stages[0]  # 1/2
        self.stage2 = nn.Sequential(*stages[1:3])  # 1/4
        self.stage3 = nn.Sequential(*stages[3:5])  # 1/8 -> 1/16
        self.stage4 = nn.Sequential(*stages[5:])   # 1/32

        self.out_channels = [
            scale_width(16),
            scale_width(40),
            scale_width(112),
            scale_width(320)
        ]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)

        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        return {
            'p2': c1,
            'p3': c2,
            'p4': c3,
            'p5': c4
        }


# Feature Pyramid Network for multi-scale feature fusion
class FPN(nn.Module):
    """
    Feature Pyramid Network
    Fuses multi-scale features from backbone for detection heads
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )

        self.out_channels = out_channels

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature_list = [features['p2'], features['p3'], features['p4'], features['p5']]

        # Build FPN top-down
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feature_list)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest'
            )

        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return {
            'p2': outputs[0],
            'p3': outputs[1],
            'p4': outputs[2],
            'p5': outputs[3]
        }
