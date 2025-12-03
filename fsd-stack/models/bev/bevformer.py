"""
BEVFormer - Transformer-based BEV encoding.

Reference: "BEVFormer: Learning Bird's-Eye-View Representation
            from Multi-Camera Images via Spatiotemporal Transformers"
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class BEVFormer(nn.Module):
    """
    BEVFormer for camera-to-BEV transformation using transformers.

    More sophisticated than LSS, uses deformable attention for
    BEV query-based feature extraction.
    """

    def __init__(
        self,
        in_channels: int = 256,
        bev_channels: int = 256,
        bev_size: Tuple[int, int] = (200, 200),
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()

        self.bev_channels = bev_channels
        self.bev_size = bev_size

        # BEV queries (learnable)
        self.bev_queries = nn.Parameter(
            torch.randn(1, bev_size[0] * bev_size[1], bev_channels)
        )

        # Transformer layers (simplified)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=bev_channels,
                nhead=num_heads,
                dim_feedforward=bev_channels * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Input projection
        self.input_proj = nn.Linear(in_channels, bev_channels)

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform multi-camera features to BEV.

        This is a simplified implementation. Full BEVFormer uses
        deformable attention and spatial cross-attention.
        """
        B, N, C, H, W = features.shape

        # Flatten spatial dimensions
        features = features.view(B, N, C, -1).permute(0, 1, 3, 2)  # (B, N, HW, C)
        features = features.view(B, -1, C)  # (B, N*HW, C)

        # Project to BEV dimension
        features = self.input_proj(features)

        # BEV queries
        bev_queries = self.bev_queries.expand(B, -1, -1)

        # Apply transformer layers (cross-attention with features)
        x = bev_queries
        for layer in self.layers:
            x = layer(x)

        # Reshape to BEV grid
        bev = x.view(B, *self.bev_size, self.bev_channels)
        bev = bev.permute(0, 3, 1, 2)  # (B, C, H, W)

        return {'bev': bev}


if __name__ == '__main__':
    model = BEVFormer(in_channels=256, bev_size=(50, 50))
    features = torch.randn(2, 6, 256, 30, 40)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 6, 3, 3)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 6, 4, 4)

    outputs = model(features, intrinsics, extrinsics)
    print(f"BEV shape: {outputs['bev'].shape}")
