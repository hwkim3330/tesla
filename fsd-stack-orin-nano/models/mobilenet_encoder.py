"""
MobileNetV3-Small Encoder for Orin Nano

ResNet/ViT 대비 5배 빠른 경량 이미지 인코더.
모바일/엣지 디바이스를 위해 설계된 최적의 CNN.

특징:
- 파라미터: 1.5M (ResNet-50의 6%)
- FLOPs: 0.06B (ResNet-50의 1.5%)
- Inverted Residual + Squeeze-Excitation
- h-swish Activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class HardSwish(nn.Module):
    """h-swish: x * ReLU6(x + 3) / 6"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6


class HardSigmoid(nn.Module):
    """h-sigmoid: ReLU6(x + 3) / 6"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3, inplace=True) / 6


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation Block.

    채널별 중요도를 학습하여 feature recalibration.
    연산량은 적지만 정확도 향상에 효과적.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2/V3의 핵심).

    일반 Residual: 넓은 → 좁은 → 넓은
    Inverted:      좁은 → 넓은 → 좁은 (메모리 효율적)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        use_se: bool = True,
        activation: str = 'hswish',
    ):
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)

        # Activation function
        act = HardSwish() if activation == 'hswish' else nn.ReLU(inplace=True)

        layers = []

        # Expansion (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act,
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size,
                stride=stride, padding=kernel_size // 2,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            act,
        ])

        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcite(hidden_dim))

        # Projection (1x1 conv)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small - Orin Nano 최적화 버전.

    원본 대비 변경사항:
    - 마지막 분류 헤드 제거 (feature extraction만)
    - 다중 스케일 출력 (FPN 호환)

    파라미터: ~1.5M
    입력: (B, 3, H, W)
    출력: (B, 576, H/32, W/32) 또는 multi-scale
    """

    def __init__(
        self,
        num_classes: int = 0,  # 0이면 feature extractor
        width_mult: float = 1.0,
        output_stride: int = 32,
    ):
        super().__init__()

        self.num_classes = num_classes

        def _make_divisible(v, divisor=8):
            return int((v + divisor / 2) // divisor * divisor)

        # MobileNetV3-Small 설정
        # [kernel, exp_ratio, out_ch, SE, activation, stride]
        config = [
            [3, 1, 16, True, 'relu', 2],
            [3, 4.5, 24, False, 'relu', 2],
            [3, 3.67, 24, False, 'relu', 1],
            [5, 4, 40, True, 'hswish', 2],
            [5, 6, 40, True, 'hswish', 1],
            [5, 6, 40, True, 'hswish', 1],
            [5, 3, 48, True, 'hswish', 1],
            [5, 3, 48, True, 'hswish', 1],
            [5, 6, 96, True, 'hswish', 2],
            [5, 6, 96, True, 'hswish', 1],
            [5, 6, 96, True, 'hswish', 1],
        ]

        # First layer
        in_channels = _make_divisible(16 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            HardSwish(),
        )

        # Inverted Residual blocks
        self.blocks = nn.ModuleList()
        for k, exp, out, se, act, s in config:
            out_channels = _make_divisible(out * width_mult)
            self.blocks.append(
                InvertedResidual(in_channels, out_channels, k, s, exp, se, act)
            )
            in_channels = out_channels

        # Final expansion layer
        last_channels = _make_divisible(576 * width_mult)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            HardSwish(),
        )

        self.out_channels = last_channels

        # Classification head (optional)
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(last_channels, 1024),
                HardSwish(),
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Features (B, 576, H/32, W/32) or class logits
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)

        if self.num_classes > 0:
            x = self.classifier(x)

        return x

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features for FPN.

        Returns features at 1/4, 1/8, 1/16, 1/32 resolution.
        """
        features = []

        x = self.stem(x)

        # Block indices for multi-scale output
        scale_indices = [1, 2, 6, 10]  # After stride-2 layers

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in scale_indices:
                features.append(x)

        x = self.final_conv(x)
        features.append(x)

        return features


class CameraEncoder(nn.Module):
    """
    Camera encoder wrapper for FSD.

    여러 카메라의 이미지를 MobileNetV3로 인코딩.
    가중치 공유로 메모리 효율적.
    """

    def __init__(
        self,
        num_cameras: int = 1,
        pretrained: bool = True,
        output_dim: int = 256,
    ):
        super().__init__()

        self.num_cameras = num_cameras

        # MobileNetV3-Small backbone
        self.backbone = MobileNetV3Small()

        # Load pretrained weights if available
        if pretrained:
            self._load_pretrained()

        # Project to desired dimension
        self.proj = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        self.output_dim = output_dim

    def _load_pretrained(self):
        """Load ImageNet pretrained weights."""
        try:
            import torchvision
            pretrained = torchvision.models.mobilenet_v3_small(pretrained=True)
            # Copy matching weights
            state_dict = pretrained.state_dict()
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained MobileNetV3-Small weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode camera images.

        Args:
            images: (B, C, H, W) or (B, N_cams, C, H, W)

        Returns:
            Features (B, output_dim, H', W') or (B, N_cams, output_dim, H', W')
        """
        if images.dim() == 5:
            # Multi-camera input
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            features = self.backbone(images)
            features = self.proj(features)
            _, c, h, w = features.shape
            features = features.view(B, N, c, h, w)
        else:
            features = self.backbone(images)
            features = self.proj(features)

        return features


# Model comparison for reference
MODEL_COMPARISON = """
┌─────────────────┬────────────┬──────────┬─────────────────────┐
│     Model       │  Params    │  FLOPs   │ ImageNet Acc (Top1) │
├─────────────────┼────────────┼──────────┼─────────────────────┤
│ ResNet-50       │   25.6M    │  4.1B    │      76.1%          │
│ ResNet-18       │   11.7M    │  1.8B    │      69.8%          │
│ EfficientNet-B0 │    5.3M    │  0.4B    │      77.1%          │
│ MobileNetV2     │    3.4M    │  0.3B    │      72.0%          │
│ MobileNetV3-L   │    5.4M    │  0.22B   │      75.2%          │
│ MobileNetV3-S   │    1.5M    │  0.06B   │      67.4%          │ ← 우리가 사용
└─────────────────┴────────────┴──────────┴─────────────────────┘

MobileNetV3-Small 선택 이유:
1. Orin Nano의 제한된 GPU 코어 (1024개 CUDA cores)
2. 메모리 대역폭 제한 (68 GB/s)
3. 실시간 요구사항 (20+ FPS)
4. 정확도 vs 속도 최적점
"""


if __name__ == '__main__':
    # Test MobileNetV3-Small
    model = MobileNetV3Small()
    x = torch.randn(2, 3, 480, 640)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV3-Small Parameters: {params / 1e6:.2f}M")

    # Forward pass
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    # Multi-scale features
    features = model.extract_features(x)
    print("\nMulti-scale features:")
    for i, f in enumerate(features):
        print(f"  Scale {i}: {f.shape}")

    # Camera encoder
    encoder = CameraEncoder(output_dim=256)
    y = encoder(x)
    print(f"\nCamera Encoder output: {y.shape}")

    print(MODEL_COMPARISON)
