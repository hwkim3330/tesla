"""
EfficientNet Backbone for Tesla FSD

EfficientNet-V2 provides excellent accuracy/efficiency tradeoff
and can be used as an alternative backbone to RegNet.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import timm


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone for image feature extraction.

    Args:
        model_name: EfficientNet variant
        pretrained: Use ImageNet pretrained weights
        out_indices: Stages to extract features from
    """

    VARIANTS = {
        'efficientnet_b0': [16, 24, 40, 112, 320],
        'efficientnet_b1': [16, 24, 40, 112, 320],
        'efficientnet_b2': [16, 24, 48, 120, 352],
        'efficientnet_b3': [24, 32, 48, 136, 384],
        'efficientnet_b4': [24, 32, 56, 160, 448],
        'efficientnet_b5': [24, 40, 64, 176, 512],
        'efficientnetv2_s': [24, 48, 64, 160, 256],
        'efficientnetv2_m': [24, 48, 80, 176, 512],
        'efficientnetv2_l': [32, 64, 96, 224, 640],
    }

    def __init__(
        self,
        model_name: str = 'efficientnetv2_m',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
    ):
        super().__init__()

        self.model_name = model_name
        self.out_indices = out_indices

        # Load from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dims = [f.shape[1] for f in features]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            features = self.backbone(x)
            features = [
                f.view(B, N, f.shape[1], f.shape[2], f.shape[3])
                for f in features
            ]
        else:
            features = self.backbone(x)
        return features

    @property
    def out_channels(self) -> List[int]:
        return self.feature_dims


if __name__ == '__main__':
    model = EfficientNetBackbone('efficientnetv2_m', pretrained=False)
    x = torch.randn(2, 3, 480, 640)
    features = model(x)
    print("EfficientNet-V2-M features:")
    for i, f in enumerate(features):
        print(f"  Stage {i}: {f.shape}")
