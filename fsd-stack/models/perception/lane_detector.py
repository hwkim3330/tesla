"""
Lane Detection for Autonomous Driving

Detects lane markings and road boundaries.
Outputs polyline representations for path planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LaneDetector(nn.Module):
    """
    Lane detection using segmentation + polyline extraction.

    Predicts:
    - Lane segmentation mask
    - Lane instance embeddings
    - Lane polyline points
    """

    LANE_TYPES = [
        'solid_white', 'dashed_white', 'solid_yellow',
        'dashed_yellow', 'double_yellow', 'road_edge', 'curb'
    ]

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 7,
        num_points: int = 72,  # Points per lane
        max_lanes: int = 8,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_points = num_points
        self.max_lanes = max_lanes

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes + 1, 1),  # +1 for background
        )

        # Instance embedding head
        self.embed_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 1),  # 16-dim embedding
        )

        # Polyline prediction (row-based anchor approach)
        self.row_anchor = nn.Parameter(
            torch.linspace(0.3, 0.95, num_points),
            requires_grad=False
        )

        self.poly_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((num_points, 1)),
            nn.Flatten(start_dim=2),
            nn.Linear(256, max_lanes * 2),  # x position + existence for each lane
        )

        # Lane existence classifier
        self.exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, max_lanes),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect lanes.

        Args:
            features: Image features (B, C, H, W)

        Returns:
            Dictionary with:
                - 'seg': Segmentation logits (B, C+1, H, W)
                - 'embed': Instance embeddings (B, 16, H, W)
                - 'lanes': Polyline predictions (B, max_lanes, num_points)
                - 'exist': Lane existence (B, max_lanes)
        """
        B = features.shape[0]

        # Segmentation
        seg = self.seg_head(features)

        # Instance embeddings
        embed = self.embed_head(features)

        # Polyline predictions
        poly_feat = self.poly_head(features)
        lanes = poly_feat.view(B, self.num_points, self.max_lanes, 2)
        lanes = lanes.permute(0, 2, 1, 3)  # (B, max_lanes, num_points, 2)

        # Lane existence
        exist = self.exist_head(features)
        exist = torch.sigmoid(exist)

        return {
            'seg': seg,
            'embed': embed,
            'lanes': lanes[..., 0],  # x positions
            'lanes_conf': torch.sigmoid(lanes[..., 1]),  # confidence per point
            'exist': exist,
        }

    def decode_lanes(
        self,
        outputs: Dict[str, torch.Tensor],
        img_size: Tuple[int, int],
        threshold: float = 0.5,
    ) -> List[List[Tuple[float, float]]]:
        """
        Decode lane predictions to polyline coordinates.

        Args:
            outputs: Network outputs
            img_size: Image size (H, W)
            threshold: Lane existence threshold

        Returns:
            List of lanes, each lane is a list of (x, y) points
        """
        lanes_x = outputs['lanes']  # (B, max_lanes, num_points)
        exist = outputs['exist']     # (B, max_lanes)

        B = lanes_x.shape[0]
        H, W = img_size

        all_lanes = []

        for b in range(B):
            batch_lanes = []
            for lane_idx in range(self.max_lanes):
                if exist[b, lane_idx] > threshold:
                    # Get x positions (normalized)
                    xs = lanes_x[b, lane_idx].cpu().numpy()
                    # Get y positions from row anchors
                    ys = self.row_anchor.cpu().numpy()

                    # Convert to pixel coordinates
                    points = [(x * W, y * H) for x, y in zip(xs, ys) if 0 <= x <= 1]
                    if len(points) > 2:
                        batch_lanes.append(points)

            all_lanes.append(batch_lanes)

        return all_lanes


class LaneLoss(nn.Module):
    """Loss functions for lane detection."""

    def __init__(
        self,
        seg_weight: float = 1.0,
        exist_weight: float = 0.1,
        poly_weight: float = 1.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.exist_weight = exist_weight
        self.poly_weight = poly_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute lane detection losses."""
        losses = {}

        # Segmentation loss (cross-entropy)
        if 'seg' in targets:
            losses['seg'] = F.cross_entropy(
                predictions['seg'],
                targets['seg'],
            ) * self.seg_weight

        # Lane existence loss (BCE)
        if 'exist' in targets:
            losses['exist'] = F.binary_cross_entropy(
                predictions['exist'],
                targets['exist'].float(),
            ) * self.exist_weight

        # Polyline regression loss (smooth L1)
        if 'lanes' in targets:
            mask = targets['exist'].unsqueeze(-1).expand_as(predictions['lanes'])
            losses['poly'] = F.smooth_l1_loss(
                predictions['lanes'] * mask,
                targets['lanes'] * mask,
            ) * self.poly_weight

        losses['total'] = sum(losses.values())
        return losses


if __name__ == '__main__':
    detector = LaneDetector(in_channels=256)
    features = torch.randn(2, 256, 60, 80)
    outputs = detector(features)

    print("Lane Detector outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
