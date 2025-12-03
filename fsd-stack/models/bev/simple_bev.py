"""SimpleBEV - Lightweight BEV transformation."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class SimpleBEV(nn.Module):
    """Simple IPM-based BEV transformation for lightweight inference."""

    def __init__(
        self,
        in_channels: int = 256,
        bev_channels: int = 128,
        bev_size: Tuple[int, int] = (100, 100),
    ):
        super().__init__()
        self.bev_size = bev_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(bev_size),
        )

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor = None,
        extrinsics: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, C, H, W = features.shape
        features = features.mean(dim=1)  # Average across cameras
        bev = self.encoder(features)
        return {'bev': bev}
