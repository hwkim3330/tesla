"""
Traffic Light Detection with Distance Estimation

This module detects traffic lights, classifies their state (red, yellow, green, arrow),
and estimates the distance using monocular depth estimation.

Key features:
- Multi-scale detection for different traffic light sizes
- State classification with arrow detection
- Camera-based distance estimation
- Relevance scoring (which traffic light applies to our lane)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class TrafficLightDetection:
    """Traffic light detection result."""
    bbox: torch.Tensor          # (x1, y1, x2, y2)
    confidence: float           # Detection confidence
    state: str                  # 'red', 'yellow', 'green', 'off', 'unknown'
    arrow: Optional[str]        # 'left', 'right', 'straight', None
    distance: float             # Estimated distance in meters
    relevance: float            # Relevance score (0-1) for current lane
    camera_id: int              # Which camera detected it


class TrafficLightHead(nn.Module):
    """
    Detection head for traffic lights.

    Outputs:
    - Bounding box regression
    - Objectness score
    - State classification
    - Arrow classification
    - Distance estimation
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 3,
        num_states: int = 5,  # red, yellow, green, off, unknown
        num_arrows: int = 4,  # left, right, straight, none
    ):
        super().__init__()

        self.num_anchors = num_anchors
        self.num_states = num_states
        self.num_arrows = num_arrows

        # Shared convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Per-anchor outputs
        # 4 (bbox) + 1 (objectness) + num_states + num_arrows + 1 (distance)
        out_channels = num_anchors * (4 + 1 + num_states + num_arrows + 1)
        self.pred = nn.Conv2d(256, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Feature map (B, C, H, W)

        Returns:
            Dictionary with predictions
        """
        B, _, H, W = x.shape

        x = self.conv(x)
        pred = self.pred(x)

        # Reshape to (B, num_anchors, num_outputs, H, W)
        pred = pred.view(B, self.num_anchors, -1, H, W)

        # Split outputs
        idx = 0
        bbox = pred[:, :, idx:idx+4]  # (B, A, 4, H, W)
        idx += 4

        objectness = pred[:, :, idx:idx+1]  # (B, A, 1, H, W)
        idx += 1

        state = pred[:, :, idx:idx+self.num_states]  # (B, A, S, H, W)
        idx += self.num_states

        arrow = pred[:, :, idx:idx+self.num_arrows]  # (B, A, Ar, H, W)
        idx += self.num_arrows

        distance = pred[:, :, idx:idx+1]  # (B, A, 1, H, W)

        return {
            'bbox': bbox,
            'objectness': torch.sigmoid(objectness),
            'state': state,  # Will apply softmax during loss/inference
            'arrow': arrow,
            'distance': F.softplus(distance) + 1.0,  # Ensure positive, min 1m
        }


class DistanceEstimator(nn.Module):
    """
    Monocular distance estimation for traffic lights.

    Uses multiple cues:
    - Bounding box size (larger = closer)
    - Vertical position in image (lower = closer for ground-level objects)
    - Learned features from ROI
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()

        self.roi_size = roi_size

        # ROI feature processor
        self.roi_processor = nn.Sequential(
            nn.Linear(in_channels * roi_size[0] * roi_size[1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Geometric features (bbox size, position)
        self.geo_processor = nn.Sequential(
            nn.Linear(6, 32),  # (cx, cy, w, h, aspect_ratio, relative_area)
            nn.ReLU(inplace=True),
        )

        # Combined distance prediction
        self.distance_head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Camera intrinsics-based refinement
        self.known_tl_height = 0.3  # Typical traffic light height in meters

    def forward(
        self,
        roi_features: torch.Tensor,
        bboxes: torch.Tensor,
        img_size: Tuple[int, int],
        focal_length: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Estimate distance to traffic lights.

        Args:
            roi_features: ROI-pooled features (N, C, H, W)
            bboxes: Bounding boxes (N, 4) in (x1, y1, x2, y2) format
            img_size: Image size (H, W)
            focal_length: Camera focal length (if known)

        Returns:
            distances: Estimated distances in meters (N,)
        """
        N = roi_features.shape[0]
        if N == 0:
            return torch.zeros(0, device=roi_features.device)

        # Process ROI features
        roi_flat = roi_features.view(N, -1)
        roi_feat = self.roi_processor(roi_flat)

        # Compute geometric features
        x1, y1, x2, y2 = bboxes.unbind(-1)
        cx = (x1 + x2) / 2 / img_size[1]  # Normalized center x
        cy = (y1 + y2) / 2 / img_size[0]  # Normalized center y
        w = (x2 - x1) / img_size[1]       # Normalized width
        h = (y2 - y1) / img_size[0]       # Normalized height
        aspect = w / (h + 1e-6)
        area = w * h

        geo_features = torch.stack([cx, cy, w, h, aspect, area], dim=-1)
        geo_feat = self.geo_processor(geo_features)

        # Combined prediction
        combined = torch.cat([roi_feat, geo_feat], dim=-1)
        distance = self.distance_head(combined).squeeze(-1)
        distance = F.softplus(distance) + 1.0  # Minimum 1 meter

        # Refine with pinhole camera model if focal length is known
        if focal_length is not None:
            # Distance = (known_height * focal_length) / pixel_height
            pixel_height = (y2 - y1)
            geometric_distance = (self.known_tl_height * focal_length) / (pixel_height + 1e-6)
            # Blend learned and geometric estimates
            distance = 0.7 * distance + 0.3 * geometric_distance

        return distance


class TrafficLightDetector(nn.Module):
    """
    Complete traffic light detection module.

    Detects traffic lights in multi-camera images, classifies their state,
    and estimates distance using monocular cues.

    Args:
        in_channels: Input feature channels (from backbone)
        num_classes: Number of traffic light states
        anchor_sizes: Anchor box sizes for detection
        score_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold
    """

    # Traffic light states
    STATES = ['red', 'yellow', 'green', 'off', 'unknown']
    ARROWS = ['left', 'right', 'straight', 'none']

    def __init__(
        self,
        in_channels: int = 256,
        fpn_channels: int = 256,
        num_levels: int = 4,
        anchor_sizes: List[List[int]] = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        super().__init__()

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.num_levels = num_levels

        # Default anchors optimized for traffic lights
        if anchor_sizes is None:
            anchor_sizes = [
                [(8, 16), (12, 24), (16, 32)],     # Small scale
                [(24, 48), (32, 64), (40, 80)],    # Medium scale
                [(48, 96), (64, 128), (80, 160)],  # Large scale
                [(96, 192), (128, 256), (160, 320)],  # Very large
            ]
        self.anchor_sizes = anchor_sizes

        # Feature pyramid adaptation
        self.fpn_adapt = nn.ModuleList([
            nn.Conv2d(in_channels, fpn_channels, 1)
            for _ in range(num_levels)
        ])

        # Detection heads for each FPN level
        self.heads = nn.ModuleList([
            TrafficLightHead(
                fpn_channels,
                num_anchors=len(anchor_sizes[i]) if i < len(anchor_sizes) else 3,
            )
            for i in range(num_levels)
        ])

        # Distance estimator
        self.distance_estimator = DistanceEstimator(fpn_channels)

        # Relevance scorer (which traffic light applies to our lane)
        self.relevance_head = nn.Sequential(
            nn.Linear(256 + 4, 128),  # Features + bbox
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect traffic lights in multi-scale features.

        Args:
            features: List of FPN features [(B, C, H1, W1), (B, C, H2, W2), ...]
            intrinsics: Camera intrinsics for distance refinement (B, 3, 3)

        Returns:
            Dictionary with multi-level predictions
        """
        all_outputs = {
            'bbox': [],
            'objectness': [],
            'state': [],
            'arrow': [],
            'distance': [],
        }

        for i, (feat, head, adapt) in enumerate(zip(features, self.heads, self.fpn_adapt)):
            # Adapt features
            feat = adapt(feat)

            # Get predictions
            preds = head(feat)

            for key in all_outputs:
                all_outputs[key].append(preds[key])

        return all_outputs

    def decode_predictions(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        img_size: Tuple[int, int],
        focal_length: Optional[float] = None,
    ) -> List[TrafficLightDetection]:
        """
        Decode raw predictions to detection objects.

        Args:
            predictions: Raw model outputs
            img_size: Original image size (H, W)
            focal_length: Camera focal length

        Returns:
            List of TrafficLightDetection objects
        """
        detections = []

        # Process each FPN level
        for level_idx in range(len(predictions['bbox'])):
            bbox = predictions['bbox'][level_idx]
            objectness = predictions['objectness'][level_idx]
            state = predictions['state'][level_idx]
            arrow = predictions['arrow'][level_idx]
            distance = predictions['distance'][level_idx]

            B, A, _, H, W = bbox.shape

            # Create anchor grid
            anchors = self._generate_anchors(level_idx, H, W, bbox.device)

            # Decode boxes
            decoded_bbox = self._decode_boxes(bbox, anchors, img_size)

            # Apply threshold
            scores = objectness.squeeze(2)  # (B, A, H, W)
            mask = scores > self.score_threshold

            # Get detections for each image
            for b in range(B):
                indices = mask[b].nonzero(as_tuple=False)

                for idx in indices:
                    a, h, w = idx[0].item(), idx[1].item(), idx[2].item()

                    det = TrafficLightDetection(
                        bbox=decoded_bbox[b, a, :, h, w],
                        confidence=scores[b, a, h, w].item(),
                        state=self.STATES[state[b, a, :, h, w].argmax().item()],
                        arrow=self.ARROWS[arrow[b, a, :, h, w].argmax().item()],
                        distance=distance[b, a, 0, h, w].item(),
                        relevance=1.0,  # Will be computed separately
                        camera_id=0,
                    )
                    detections.append(det)

        # Apply NMS
        detections = self._apply_nms(detections)

        return detections

    def _generate_anchors(
        self,
        level: int,
        feat_h: int,
        feat_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate anchor boxes for a feature level."""
        stride = 2 ** (level + 2)  # FPN strides: 4, 8, 16, 32

        # Grid
        y = torch.arange(feat_h, device=device) * stride + stride // 2
        x = torch.arange(feat_w, device=device) * stride + stride // 2
        y, x = torch.meshgrid(y, x, indexing='ij')

        # Anchor sizes for this level
        sizes = self.anchor_sizes[min(level, len(self.anchor_sizes) - 1)]
        num_anchors = len(sizes)

        anchors = torch.zeros(num_anchors, 4, feat_h, feat_w, device=device)
        for a, (w, h) in enumerate(sizes):
            anchors[a, 0] = x - w / 2  # x1
            anchors[a, 1] = y - h / 2  # y1
            anchors[a, 2] = x + w / 2  # x2
            anchors[a, 3] = y + h / 2  # y2

        return anchors

    def _decode_boxes(
        self,
        offsets: torch.Tensor,
        anchors: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode box offsets to absolute coordinates."""
        # Simple offset decoding
        decoded = anchors.unsqueeze(0) + offsets * 10  # Scale factor

        # Clip to image bounds
        decoded[:, :, 0].clamp_(min=0, max=img_size[1])
        decoded[:, :, 1].clamp_(min=0, max=img_size[0])
        decoded[:, :, 2].clamp_(min=0, max=img_size[1])
        decoded[:, :, 3].clamp_(min=0, max=img_size[0])

        return decoded

    def _apply_nms(
        self,
        detections: List[TrafficLightDetection],
    ) -> List[TrafficLightDetection]:
        """Apply Non-Maximum Suppression."""
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        # NMS
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._compute_iou(best.bbox, det.bbox)
                if iou < self.nms_threshold:
                    remaining.append(det)
            detections = remaining

        return keep

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0].item(), box2[0].item())
        y1 = max(box1[1].item(), box2[1].item())
        x2 = min(box1[2].item(), box2[2].item())
        y2 = min(box1[3].item(), box2[3].item())

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-6)


class TrafficLightLoss(nn.Module):
    """
    Loss function for traffic light detection.

    Combines:
    - Focal loss for objectness
    - Smooth L1 for bbox regression
    - Cross-entropy for state/arrow classification
    - Smooth L1 for distance estimation
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection losses.

        Args:
            predictions: Model outputs
            targets: Ground truth annotations

        Returns:
            Dictionary of loss components
        """
        losses = {
            'objectness': 0.0,
            'bbox': 0.0,
            'state': 0.0,
            'arrow': 0.0,
            'distance': 0.0,
        }

        # Implementation would match predictions to GT and compute losses
        # Simplified for brevity

        return losses


if __name__ == '__main__':
    # Test traffic light detector
    detector = TrafficLightDetector(in_channels=256)

    # Simulate FPN features
    features = [
        torch.randn(2, 256, 120, 160),
        torch.randn(2, 256, 60, 80),
        torch.randn(2, 256, 30, 40),
        torch.randn(2, 256, 15, 20),
    ]

    outputs = detector(features)

    print("Traffic Light Detector outputs:")
    for level in range(len(outputs['bbox'])):
        print(f"  Level {level}:")
        print(f"    bbox: {outputs['bbox'][level].shape}")
        print(f"    objectness: {outputs['objectness'][level].shape}")
        print(f"    state: {outputs['state'][level].shape}")
        print(f"    distance: {outputs['distance'][level].shape}")
