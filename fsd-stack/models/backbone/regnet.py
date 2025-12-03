"""
RegNet Backbone for Tesla FSD

RegNet is the backbone architecture used in Tesla's FSD neural network.
It provides efficient feature extraction with good accuracy/compute tradeoff.

Reference: "Designing Network Design Spaces" (Radosavovic et al., 2020)
Tesla AI Day 2021/2022 showed RegNet-based backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm


class RegNetBackbone(nn.Module):
    """
    RegNet backbone for multi-camera feature extraction.

    Tesla uses a shared RegNet backbone across all 8 cameras for efficiency.
    Features are extracted at multiple scales for different downstream tasks.

    Args:
        model_name: RegNet variant ('regnetx_032', 'regnety_040', 'regnety_080', etc.)
        pretrained: Whether to use ImageNet pretrained weights
        out_indices: Which stages to output features from (0-indexed)
        freeze_stages: Number of stages to freeze for fine-tuning
        norm_eval: Whether to keep BN in eval mode during training
    """

    # Feature dimensions for different RegNet variants
    FEATURE_DIMS = {
        'regnetx_002': [24, 56, 152, 368],
        'regnetx_004': [32, 64, 160, 384],
        'regnetx_006': [48, 96, 240, 528],
        'regnetx_008': [64, 128, 288, 672],
        'regnetx_016': [72, 168, 408, 912],
        'regnetx_032': [96, 192, 432, 1008],
        'regnetx_040': [80, 240, 560, 1360],
        'regnetx_064': [168, 392, 784, 1624],
        'regnetx_080': [80, 240, 720, 1920],
        'regnetx_120': [224, 448, 896, 2240],
        'regnetx_160': [256, 512, 896, 2048],
        'regnety_002': [24, 56, 152, 368],
        'regnety_004': [48, 104, 208, 440],
        'regnety_006': [48, 112, 256, 608],
        'regnety_008': [64, 128, 320, 768],
        'regnety_016': [48, 120, 336, 888],
        'regnety_032': [72, 216, 576, 1512],
        'regnety_040': [128, 192, 512, 1088],  # Tesla-like
        'regnety_064': [144, 288, 576, 1296],
        'regnety_080': [168, 448, 896, 2016],
        'regnety_120': [224, 448, 896, 2240],
        'regnety_160': [224, 448, 1232, 3024],
        'regnety_320': [232, 696, 1392, 3712],
    }

    def __init__(
        self,
        model_name: str = 'regnety_040',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        freeze_stages: int = -1,
        norm_eval: bool = False,
        with_cp: bool = False,  # Gradient checkpointing
    ):
        super().__init__()

        self.model_name = model_name
        self.out_indices = out_indices
        self.freeze_stages = freeze_stages
        self.norm_eval = norm_eval

        # Load pretrained RegNet from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # Get feature dimensions
        if model_name in self.FEATURE_DIMS:
            self.feature_dims = [self.FEATURE_DIMS[model_name][i] for i in out_indices]
        else:
            # Infer from model
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                features = self.backbone(dummy)
                self.feature_dims = [f.shape[1] for f in features]

        # Optional: Freeze early stages for fine-tuning
        if freeze_stages >= 0:
            self._freeze_stages(freeze_stages)

        # Feature Pyramid Network for multi-scale features (optional)
        self.fpn = None

    def _freeze_stages(self, num_stages: int):
        """Freeze the first num_stages stages."""
        # Freeze stem
        if hasattr(self.backbone, 'stem'):
            for param in self.backbone.stem.parameters():
                param.requires_grad = False

        # Freeze stages
        for i in range(num_stages):
            stage_name = f's{i+1}'
            if hasattr(self.backbone, stage_name):
                stage = getattr(self.backbone, stage_name)
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, N_cams, C, H, W)

        Returns:
            List of feature tensors at different scales
        """
        # Handle multi-camera input
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            features = self.backbone(x)
            # Reshape back to (B, N, C, H, W) for each scale
            features = [
                f.view(B, N, f.shape[1], f.shape[2], f.shape[3])
                for f in features
            ]
        else:
            features = self.backbone(x)

        return features

    def train(self, mode: bool = True):
        """Set training mode with optional frozen BN."""
        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        return self

    @property
    def out_channels(self) -> List[int]:
        """Return output channel dimensions."""
        return self.feature_dims


class RegNetWithNeck(nn.Module):
    """
    RegNet backbone with FPN neck for multi-scale features.

    This is similar to what Tesla uses - a shared backbone with
    a Feature Pyramid Network to produce multi-scale features.
    """

    def __init__(
        self,
        backbone_name: str = 'regnety_040',
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        num_fpn_levels: int = 4,
    ):
        super().__init__()

        self.backbone = RegNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3),
        )

        # FPN lateral connections
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in self.backbone.out_channels:
            lateral_conv = nn.Conv2d(in_channels, fpn_out_channels, 1)
            fpn_conv = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        self.out_channels = [fpn_out_channels] * num_fpn_levels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward with FPN."""
        # Backbone features
        features = self.backbone(x)

        # FPN
        laterals = [
            lateral_conv(f)
            for f, lateral_conv in zip(features, self.lateral_convs)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Output convolutions
        outputs = [
            fpn_conv(lateral)
            for lateral, fpn_conv in zip(laterals, self.fpn_convs)
        ]

        return outputs


# Convenience function
def build_regnet_backbone(
    variant: str = 'regnety_040',
    pretrained: bool = True,
    with_fpn: bool = True,
    fpn_channels: int = 256,
) -> nn.Module:
    """
    Build a RegNet backbone.

    Args:
        variant: RegNet variant name
        pretrained: Use ImageNet pretrained weights
        with_fpn: Include Feature Pyramid Network
        fpn_channels: FPN output channels

    Returns:
        RegNet backbone module
    """
    if with_fpn:
        return RegNetWithNeck(
            backbone_name=variant,
            pretrained=pretrained,
            fpn_out_channels=fpn_channels,
        )
    else:
        return RegNetBackbone(
            model_name=variant,
            pretrained=pretrained,
        )


if __name__ == '__main__':
    # Test the backbone
    model = RegNetBackbone('regnety_040', pretrained=False)
    print(f"Feature dimensions: {model.out_channels}")

    # Test with single image
    x = torch.randn(2, 3, 480, 640)
    features = model(x)
    print("\nSingle image features:")
    for i, f in enumerate(features):
        print(f"  Stage {i}: {f.shape}")

    # Test with multi-camera input
    x = torch.randn(2, 8, 3, 480, 640)  # 8 cameras
    features = model(x)
    print("\nMulti-camera features:")
    for i, f in enumerate(features):
        print(f"  Stage {i}: {f.shape}")

    # Test with FPN
    model_fpn = RegNetWithNeck('regnety_040', pretrained=False)
    x = torch.randn(2, 3, 480, 640)
    features = model_fpn(x)
    print("\nFPN features:")
    for i, f in enumerate(features):
        print(f"  Level {i}: {f.shape}")
