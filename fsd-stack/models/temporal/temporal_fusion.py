"""Temporal fusion for video understanding."""

import torch
import torch.nn as nn
from typing import Optional


class TemporalFusion(nn.Module):
    """GRU-based temporal fusion."""

    def __init__(self, channels: int = 256, num_frames: int = 4):
        super().__init__()
        self.gru = nn.GRU(channels, channels, batch_first=True)
        self.num_frames = num_frames
        self.memory = None

    def forward(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.mean(dim=[2, 3])

        if reset or self.memory is None:
            self.memory = feat.unsqueeze(1)
        else:
            self.memory = torch.cat([self.memory, feat.unsqueeze(1)], dim=1)
            if self.memory.shape[1] > self.num_frames:
                self.memory = self.memory[:, -self.num_frames:]

        _, hidden = self.gru(self.memory)
        return x + hidden.squeeze(0).view(B, C, 1, 1).expand_as(x)


class VideoTransformer(nn.Module):
    """Transformer-based temporal fusion."""

    def __init__(self, channels: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(channels, num_heads, channels * 4, batch_first=True),
            num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)
        x = self.layers(x)
        return x.view(B, T, C, H, W)
