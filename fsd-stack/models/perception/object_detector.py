"""
Object Detection for Autonomous Driving

3D object detection from camera images.
Detects vehicles, pedestrians, cyclists, and other road users.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection3D:
    """3D object detection result."""
    bbox_2d: torch.Tensor       # 2D bbox (x1, y1, x2, y2)
    bbox_3d: torch.Tensor       # 3D bbox (x, y, z, w, l, h, yaw)
    class_id: int               # Object class
    class_name: str             # Class name
    confidence: float           # Detection confidence
    velocity: Optional[torch.Tensor] = None  # (vx, vy) if available


class DetectionHead(nn.Module):
    """
    Detection head for anchor-based object detection.

    Predicts:
    - 2D bounding boxes
    - 3D bounding boxes (center, size, orientation)
    - Class probabilities
    - Velocity (optional)
    """

    CLASSES = [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle',
        'pedestrian', 'traffic_cone', 'barrier', 'other'
    ]

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 9,
        num_anchors: int = 9,
        predict_velocity: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.predict_velocity = predict_velocity

        # Shared convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 2D box head (4 values per anchor)
        self.bbox_2d_head = nn.Conv2d(256, num_anchors * 4, 1)

        # 3D box head (7 values: x, y, z, w, l, h, yaw)
        self.bbox_3d_head = nn.Conv2d(256, num_anchors * 7, 1)

        # Classification head
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, 1)

        # Velocity head (optional)
        if predict_velocity:
            self.vel_head = nn.Conv2d(256, num_anchors * 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.bbox_2d_head, self.bbox_3d_head, self.cls_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.bias, bias_value)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, _, H, W = x.shape

        x = self.conv(x)

        bbox_2d = self.bbox_2d_head(x)
        bbox_3d = self.bbox_3d_head(x)
        cls = self.cls_head(x)

        # Reshape outputs
        bbox_2d = bbox_2d.view(B, self.num_anchors, 4, H, W)
        bbox_3d = bbox_3d.view(B, self.num_anchors, 7, H, W)
        cls = cls.view(B, self.num_anchors, self.num_classes, H, W)

        outputs = {
            'bbox_2d': bbox_2d,
            'bbox_3d': bbox_3d,
            'cls': cls,
        }

        if self.predict_velocity:
            vel = self.vel_head(x)
            vel = vel.view(B, self.num_anchors, 2, H, W)
            outputs['velocity'] = vel

        return outputs


class ObjectDetector(nn.Module):
    """
    Multi-scale object detector using FPN features.

    Args:
        in_channels: FPN feature channels
        num_classes: Number of object classes
        num_levels: Number of FPN levels
        score_threshold: Detection threshold
        nms_threshold: NMS IoU threshold
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 9,
        num_levels: int = 5,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Detection heads (shared across levels)
        self.head = DetectionHead(
            in_channels=in_channels,
            num_classes=num_classes,
        )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Detect objects in multi-scale features.

        Args:
            features: List of FPN features

        Returns:
            Dictionary with per-level predictions
        """
        outputs = {
            'bbox_2d': [],
            'bbox_3d': [],
            'cls': [],
            'velocity': [],
        }

        for feat in features[:self.num_levels]:
            preds = self.head(feat)
            for key in outputs:
                if key in preds:
                    outputs[key].append(preds[key])

        return outputs


class ObjectDetector3D(nn.Module):
    """
    3D object detector from BEV features.

    Detects objects directly in BEV space for more accurate 3D localization.
    """

    CLASSES = [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle',
        'pedestrian', 'traffic_cone', 'barrier'
    ]

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 8,
        bev_size: Tuple[int, int] = (200, 200),
        bev_range: float = 50.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.resolution = 2 * bev_range / bev_size[0]

        # BEV feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Heatmap head (object centers)
        self.heatmap = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

        # Regression heads
        self.offset = nn.Conv2d(256, 2, 1)      # Sub-pixel offset
        self.size = nn.Conv2d(256, 3, 1)        # w, l, h
        self.rotation = nn.Conv2d(256, 2, 1)    # sin, cos of yaw
        self.velocity = nn.Conv2d(256, 2, 1)    # vx, vy
        self.height = nn.Conv2d(256, 1, 1)      # z offset

    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect 3D objects in BEV.

        Args:
            bev_features: BEV features (B, C, H, W)

        Returns:
            Dictionary with detection outputs
        """
        x = self.encoder(bev_features)

        # Heatmap (object centers)
        heatmap = self.heatmap(x)
        heatmap = torch.sigmoid(heatmap)

        # Regression outputs
        offset = self.offset(x)
        size = self.size(x)
        size = F.relu(size)  # Sizes must be positive
        rotation = self.rotation(x)
        rotation = F.normalize(rotation, dim=1)  # Normalize sin/cos
        velocity = self.velocity(x)
        height = self.height(x)

        return {
            'heatmap': heatmap,
            'offset': offset,
            'size': size,
            'rotation': rotation,
            'velocity': velocity,
            'height': height,
        }

    def decode_detections(
        self,
        outputs: Dict[str, torch.Tensor],
        score_threshold: float = 0.3,
        max_detections: int = 100,
    ) -> List[Detection3D]:
        """
        Decode network outputs to detection objects.

        Args:
            outputs: Network outputs
            score_threshold: Confidence threshold
            max_detections: Maximum number of detections

        Returns:
            List of Detection3D objects
        """
        heatmap = outputs['heatmap']
        B, C, H, W = heatmap.shape

        detections = []

        for b in range(B):
            for c in range(C):
                # Find local maxima
                heat = heatmap[b, c]
                mask = (heat > score_threshold) & (heat == F.max_pool2d(heat.unsqueeze(0), 3, 1, 1).squeeze(0))

                indices = mask.nonzero(as_tuple=False)

                for idx in indices[:max_detections]:
                    y, x = idx[0].item(), idx[1].item()
                    score = heat[y, x].item()

                    # Get regression values
                    offset = outputs['offset'][b, :, y, x]
                    size = outputs['size'][b, :, y, x]
                    rot = outputs['rotation'][b, :, y, x]
                    vel = outputs['velocity'][b, :, y, x]
                    z = outputs['height'][b, 0, y, x]

                    # Convert to world coordinates
                    px = (x + offset[0].item()) * self.resolution - self.bev_range
                    py = (y + offset[1].item()) * self.resolution - self.bev_range
                    yaw = torch.atan2(rot[0], rot[1]).item()

                    bbox_3d = torch.tensor([
                        px, py, z.item(),
                        size[0].item(), size[1].item(), size[2].item(),
                        yaw
                    ])

                    det = Detection3D(
                        bbox_2d=torch.zeros(4),  # Not computed here
                        bbox_3d=bbox_3d,
                        class_id=c,
                        class_name=self.CLASSES[c],
                        confidence=score,
                        velocity=vel,
                    )
                    detections.append(det)

        return detections


class DetectionLoss(nn.Module):
    """
    Loss function for 3D object detection.

    Combines:
    - Focal loss for heatmap
    - L1 loss for regression
    """

    def __init__(
        self,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian focal loss for heatmap."""
        pos_mask = (target == 1).float()
        neg_mask = (target < 1).float()

        pos_loss = -torch.log(pred + 1e-6) * torch.pow(1 - pred, self.focal_alpha) * pos_mask
        neg_loss = -torch.log(1 - pred + 1e-6) * torch.pow(pred, self.focal_alpha) * \
                   torch.pow(1 - target, self.focal_beta) * neg_mask

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos > 0:
            return (pos_loss + neg_loss) / num_pos
        return neg_loss

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses."""
        losses = {}

        # Heatmap loss
        losses['heatmap'] = self.focal_loss(
            predictions['heatmap'],
            targets['heatmap'],
        )

        # Regression losses (only for positive samples)
        pos_mask = (targets['heatmap'] == 1).float()

        for key in ['offset', 'size', 'rotation', 'velocity', 'height']:
            if key in predictions and key in targets:
                loss = F.l1_loss(
                    predictions[key] * pos_mask,
                    targets[key] * pos_mask,
                    reduction='sum'
                ) / (pos_mask.sum() + 1e-6)
                losses[key] = loss

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


if __name__ == '__main__':
    # Test object detector
    detector = ObjectDetector3D(in_channels=256)

    bev = torch.randn(2, 256, 200, 200)
    outputs = detector(bev)

    print("Object Detector 3D outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
