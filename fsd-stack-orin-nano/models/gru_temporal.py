"""
GRU Temporal Module for Orin Nano

Transformer 대비 3배 빠른 시계열 처리.
과거 상황을 기억하여 판단에 활용.

왜 GRU인가?
- LSTM보다 게이트가 적음 (3개 → 2개)
- 파라미터 25% 감소
- 대부분 태스크에서 비슷한 성능
- Orin Nano에서 훨씬 빠른 추론
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRUTemporal(nn.Module):
    """
    GRU 기반 시계열 모듈.

    과거 프레임의 정보를 hidden state에 저장하여
    현재 판단에 활용.

    Transformer vs GRU 비교:
    ┌─────────────┬─────────────┬─────────────┬─────────────┐
    │    Model    │ 복잡도      │ 병렬화      │ Orin Nano  │
    ├─────────────┼─────────────┼─────────────┼─────────────┤
    │ Transformer │ O(N²)       │ 가능        │ 느림        │
    │ LSTM        │ O(N)        │ 불가        │ 보통        │
    │ GRU         │ O(N)        │ 불가        │ 빠름 ✓     │
    └─────────────┴─────────────┴─────────────┴─────────────┘
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU 레이어
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 출력 프로젝션 (bidirectional인 경우 필요)
        if bidirectional:
            self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.proj = nn.Identity()

        # Hidden state 저장
        self.register_buffer('hidden', None)

    def reset_hidden(self, batch_size: int = 1, device: torch.device = None):
        """Hidden state 초기화."""
        if device is None:
            device = next(self.parameters()).device

        self.hidden = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        reset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (B, C) 또는 (B, T, C)
            hidden: Optional hidden state
            reset: Reset hidden state

        Returns:
            output: (B, hidden_dim) 또는 (B, T, hidden_dim)
            hidden: Updated hidden state
        """
        # Handle single frame input
        single_frame = x.dim() == 2
        if single_frame:
            x = x.unsqueeze(1)  # (B, 1, C)

        B = x.shape[0]

        # Initialize or reset hidden
        if reset or hidden is None:
            if self.hidden is None or self.hidden.shape[1] != B:
                self.reset_hidden(B, x.device)
            hidden = self.hidden

        # GRU forward
        output, hidden = self.gru(x, hidden)

        # Store hidden for next call
        self.hidden = hidden.detach()

        # Project if bidirectional
        output = self.proj(output)

        # Return single frame output
        if single_frame:
            output = output.squeeze(1)

        return output, hidden


class TemporalFusion(nn.Module):
    """
    공간적 특징과 시계열 특징을 융합.

    카메라/LiDAR에서 추출한 공간적 특징을
    GRU로 시간적으로 처리하여 컨텍스트 추가.
    """

    def __init__(
        self,
        spatial_dim: int = 512,  # Concatenated camera + lidar
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_frames_memory: int = 10,  # 과거 몇 프레임 기억
    ):
        super().__init__()

        self.num_frames_memory = num_frames_memory

        # 입력 프로젝션
        self.input_proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # GRU for temporal reasoning
        self.gru = GRUTemporal(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
        )

        # 출력 프로젝션
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        reset: bool = False,
    ) -> torch.Tensor:
        """
        시공간 융합.

        Args:
            spatial_features: 공간적 특징 (B, spatial_dim)
            reset: 시계열 메모리 리셋

        Returns:
            Temporally fused features (B, output_dim)
        """
        # Project spatial features
        x = self.input_proj(spatial_features)

        # Temporal processing with GRU
        x, _ = self.gru(x, reset=reset)

        # Output projection
        x = self.output_proj(x)

        return x


class ConvGRU(nn.Module):
    """
    Convolutional GRU for spatial-temporal processing.

    일반 GRU는 1D 시퀀스용.
    ConvGRU는 이미지 시퀀스 처리에 적합.
    BEV 특징맵의 시계열 처리에 사용.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Reset gate
        self.conv_reset = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
        )

        # Update gate
        self.conv_update = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
        )

        # New gate
        self.conv_new = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
        )

        self.register_buffer('hidden', None)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        reset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input (B, C, H, W)
            hidden: Previous hidden state (B, hidden_channels, H, W)
            reset: Reset hidden state

        Returns:
            output, hidden
        """
        B, C, H, W = x.shape

        # Initialize hidden if needed
        if reset or hidden is None:
            hidden = torch.zeros(B, self.hidden_channels, H, W, device=x.device)

        # Concatenate input and hidden
        combined = torch.cat([x, hidden], dim=1)

        # Gates
        reset_gate = torch.sigmoid(self.conv_reset(combined))
        update_gate = torch.sigmoid(self.conv_update(combined))

        # New candidate
        combined_new = torch.cat([x, reset_gate * hidden], dim=1)
        new_hidden = torch.tanh(self.conv_new(combined_new))

        # Update hidden
        hidden = (1 - update_gate) * hidden + update_gate * new_hidden

        self.hidden = hidden.detach()

        return hidden, hidden


# 시계열 모델 비교
TEMPORAL_MODEL_COMPARISON = """
시계열 모델 비교 (Orin Nano 기준):

┌─────────────────┬────────────┬────────────┬────────────┬─────────────┐
│     Model       │  Params    │  Memory    │  Latency   │   추천      │
├─────────────────┼────────────┼────────────┼────────────┼─────────────┤
│ Transformer     │    많음    │    많음    │   높음     │     ✗       │
│ LSTM (2 layer)  │    보통    │    보통    │   보통     │     △       │
│ GRU (1 layer)   │    적음    │    적음    │   낮음     │     ✓       │
│ ConvGRU         │    적음    │    보통    │   보통     │     ✓       │
└─────────────────┴────────────┴────────────┴────────────┴─────────────┘

GRU vs LSTM 상세 비교:

LSTM:
  - 게이트: Forget, Input, Output (3개)
  - State: Cell state + Hidden state (2개)
  - 파라미터: 4 * (input_dim * hidden_dim + hidden_dim²)

GRU:
  - 게이트: Reset, Update (2개)
  - State: Hidden state만 (1개)
  - 파라미터: 3 * (input_dim * hidden_dim + hidden_dim²)

→ GRU가 ~25% 파라미터 감소, 비슷한 성능
"""


if __name__ == '__main__':
    print("Testing GRU Temporal Module...")

    # Test GRUTemporal
    gru = GRUTemporal(input_dim=256, hidden_dim=256)

    # Simulate sequential processing
    for i in range(5):
        x = torch.randn(2, 256)
        out, hidden = gru(x, reset=(i == 0))
        print(f"Frame {i}: input {x.shape} -> output {out.shape}")

    print("\nGRU Parameters:", sum(p.numel() for p in gru.parameters()))

    # Test TemporalFusion
    fusion = TemporalFusion(spatial_dim=512, hidden_dim=256, output_dim=256)
    x = torch.randn(2, 512)
    out = fusion(x, reset=True)
    print(f"\nTemporalFusion: {x.shape} -> {out.shape}")

    # Test ConvGRU
    conv_gru = ConvGRU(input_channels=256, hidden_channels=256)
    x = torch.randn(2, 256, 50, 50)
    out, hidden = conv_gru(x, reset=True)
    print(f"\nConvGRU: {x.shape} -> {out.shape}")

    print(TEMPORAL_MODEL_COMPARISON)
