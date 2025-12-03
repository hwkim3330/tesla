"""
HydraNet - Tesla's Multi-Task Learning Architecture

Named after the mythical Hydra with multiple heads,
HydraNet uses a shared backbone with multiple task-specific heads.

Key design principles:
1. Shared feature extraction (efficient)
2. Task-specific heads (specialized)
3. Multi-scale features (via FPN)
4. Temporal fusion (for video)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .backbone import RegNetBackbone, FPN
from .detection_heads import (
    ObjectDetectionHead,
    TrafficLightHead,
    LaneDetectionHead,
    DepthEstimationHead,
    SemanticSegmentationHead,
    PathPredictionHead
)


class HydraNet(nn.Module):
    """
    HydraNet - Multi-Task Vision Network

    Architecture:
    ┌─────────────┐
    │   Camera    │  (8 cameras in Tesla)
    │   Images    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Backbone  │  (RegNet)
    │   (Shared)  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │     FPN     │  (Multi-scale features)
    └──────┬──────┘
           │
    ┌──────┼──────┬──────┬──────┬──────┐
    │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼
  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐
  │Det│  │TL │  │Lane│  │Dep│  │Seg│  │Path│
  │   │  │   │  │    │  │th │  │   │  │    │
  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘
  Objects Traffic Lanes  Depth Semantic Path
          Lights
    """

    def __init__(
        self,
        backbone_name: str = 'tesla_custom',
        fpn_channels: int = 256,
        num_object_classes: int = 10,
        num_semantic_classes: int = 20,
        pretrained_backbone: bool = False
    ):
        super().__init__()

        # Shared backbone
        self.backbone = RegNetBackbone(
            config_name=backbone_name,
            pretrained=pretrained_backbone
        )

        # Feature Pyramid Network
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels
        )

        # Task-specific heads (the "Hydra" heads)
        self.object_head = ObjectDetectionHead(
            in_channels=fpn_channels,
            num_classes=num_object_classes
        )

        self.traffic_light_head = TrafficLightHead(
            in_channels=fpn_channels
        )

        self.lane_head = LaneDetectionHead(
            in_channels=fpn_channels
        )

        self.depth_head = DepthEstimationHead(
            in_channels=fpn_channels
        )

        self.segmentation_head = SemanticSegmentationHead(
            in_channels=fpn_channels,
            num_classes=num_semantic_classes
        )

        self.path_head = PathPredictionHead(
            in_channels=fpn_channels
        )

        # Head selection (which heads to run)
        self.active_heads = {
            'objects': True,
            'traffic_lights': True,
            'lanes': True,
            'depth': True,
            'segmentation': True,
            'path': True
        }

    def set_active_heads(self, heads: Dict[str, bool]):
        """Enable/disable specific heads for inference"""
        self.active_heads.update(heads)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HydraNet

        Args:
            images: [B, 3, H, W] or [B, N_cams, 3, H, W] input images
            return_features: Whether to return intermediate features

        Returns:
            Dictionary of predictions from all active heads
        """
        # Handle multi-camera input
        if images.dim() == 5:
            B, N_cams, C, H, W = images.shape
            images = images.view(B * N_cams, C, H, W)
            multi_cam = True
        else:
            multi_cam = False

        # Extract backbone features
        backbone_features = self.backbone(images)

        # Apply FPN
        fpn_features = self.fpn(backbone_features)

        # Collect predictions from each head
        outputs = {}

        if self.active_heads.get('objects', True):
            outputs['objects'] = self.object_head(fpn_features)

        if self.active_heads.get('traffic_lights', True):
            outputs['traffic_lights'] = self.traffic_light_head(fpn_features)

        if self.active_heads.get('lanes', True):
            outputs['lanes'] = self.lane_head(fpn_features)

        if self.active_heads.get('depth', True):
            outputs['depth'] = self.depth_head(fpn_features)

        if self.active_heads.get('segmentation', True):
            outputs['segmentation'] = self.segmentation_head(fpn_features)

        if self.active_heads.get('path', True):
            outputs['path'] = self.path_head(fpn_features)

        if return_features:
            outputs['backbone_features'] = backbone_features
            outputs['fpn_features'] = fpn_features

        return outputs

    def inference(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Run inference with post-processing

        Args:
            images: Input images
            conf_threshold: Confidence threshold for detections

        Returns:
            Post-processed predictions ready for visualization
        """
        with torch.no_grad():
            raw_outputs = self.forward(images)

        results = {}

        # Post-process object detections
        if 'objects' in raw_outputs:
            results['objects'] = self._post_process_objects(
                raw_outputs['objects'], conf_threshold
            )

        # Post-process traffic lights
        if 'traffic_lights' in raw_outputs:
            results['traffic_lights'] = self._post_process_traffic_lights(
                raw_outputs['traffic_lights'], conf_threshold
            )

        # Post-process lanes
        if 'lanes' in raw_outputs:
            results['lanes'] = self._post_process_lanes(
                raw_outputs['lanes']
            )

        # Depth is already usable
        if 'depth' in raw_outputs:
            results['depth'] = raw_outputs['depth']['depth']

        # Path prediction
        if 'path' in raw_outputs:
            results['path'] = raw_outputs['path']

        return results

    def _post_process_objects(
        self,
        predictions: Dict,
        conf_threshold: float
    ) -> List[Dict]:
        """Convert raw object predictions to detection results"""
        detections = []

        # Process each feature level
        for level, preds in predictions.items():
            cls_scores = torch.sigmoid(preds['cls_logits'])
            objectness = torch.sigmoid(preds['objectness'])

            # Combine scores
            scores = cls_scores * objectness

            # Find high-confidence detections
            max_scores, labels = scores.max(dim=1)
            mask = max_scores > conf_threshold

            if mask.sum() > 0:
                # Get bounding boxes, 3D attributes, etc.
                # Simplified - actual implementation would decode boxes properly
                pass

        return detections

    def _post_process_traffic_lights(
        self,
        predictions: Dict,
        conf_threshold: float
    ) -> List[Dict]:
        """Convert raw traffic light predictions to results"""
        traffic_lights = []

        for level, preds in predictions.items():
            detection = preds['detection']
            state = preds['state']
            distance = preds['distance']

            # Find detections above threshold
            mask = detection.squeeze(1) > conf_threshold

            if mask.sum() > 0:
                # Get state predictions
                state_probs = F.softmax(state, dim=1)
                state_labels = state_probs.argmax(dim=1)

                # Simplified - would extract actual positions
                pass

        return traffic_lights

    def _post_process_lanes(self, predictions: Dict) -> Dict:
        """Convert lane predictions to polylines"""
        lane_seg = predictions['lane_segmentation']
        lane_embed = predictions['lane_embedding']

        # Instance clustering on embeddings
        # Simplified - actual implementation uses clustering algorithms

        return {
            'segmentation': lane_seg,
            'drivable_area': predictions['drivable_area']
        }


class HydraNetLoss(nn.Module):
    """
    Multi-task loss for HydraNet training

    Handles loss weighting and gradient balancing between tasks.
    """

    def __init__(
        self,
        object_weight: float = 1.0,
        traffic_light_weight: float = 1.0,
        lane_weight: float = 1.0,
        depth_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        path_weight: float = 1.0,
        use_uncertainty_weighting: bool = True
    ):
        super().__init__()

        self.weights = {
            'objects': object_weight,
            'traffic_lights': traffic_light_weight,
            'lanes': lane_weight,
            'depth': depth_weight,
            'segmentation': segmentation_weight,
            'path': path_weight
        }

        # Learnable uncertainty weights (Kendall et al.)
        if use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                key: nn.Parameter(torch.zeros(1))
                for key in self.weights.keys()
            })
        else:
            self.log_vars = None

        # Individual loss functions
        self.focal_loss = FocalLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(
        self,
        predictions: Dict,
        targets: Dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual losses for logging
        """
        losses = {}

        # Object detection loss
        if 'objects' in predictions and 'objects' in targets:
            losses['objects'] = self._compute_object_loss(
                predictions['objects'], targets['objects']
            )

        # Traffic light loss
        if 'traffic_lights' in predictions and 'traffic_lights' in targets:
            losses['traffic_lights'] = self._compute_traffic_light_loss(
                predictions['traffic_lights'], targets['traffic_lights']
            )

        # Lane detection loss
        if 'lanes' in predictions and 'lanes' in targets:
            losses['lanes'] = self._compute_lane_loss(
                predictions['lanes'], targets['lanes']
            )

        # Depth estimation loss
        if 'depth' in predictions and 'depth' in targets:
            losses['depth'] = self._compute_depth_loss(
                predictions['depth'], targets['depth']
            )

        # Segmentation loss
        if 'segmentation' in predictions and 'segmentation' in targets:
            losses['segmentation'] = self.ce_loss(
                predictions['segmentation']['segmentation'],
                targets['segmentation']
            )

        # Path prediction loss
        if 'path' in predictions and 'path' in targets:
            losses['path'] = self._compute_path_loss(
                predictions['path'], targets['path']
            )

        # Combine losses with weighting
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)

        for key, loss in losses.items():
            if self.log_vars is not None:
                # Uncertainty weighting
                precision = torch.exp(-self.log_vars[key])
                weighted = precision * loss + self.log_vars[key]
            else:
                weighted = self.weights[key] * loss

            total_loss = total_loss + weighted
            losses[f'{key}_weighted'] = weighted

        return total_loss, losses

    def _compute_object_loss(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """Focal loss for classification + Smooth L1 for regression"""
        cls_loss = self.focal_loss(preds, targets)
        reg_loss = self.smooth_l1(preds['bbox_pred'], targets['boxes'])
        return cls_loss + reg_loss

    def _compute_traffic_light_loss(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """Combined detection + classification loss"""
        det_loss = self.bce_loss(preds['detection'], targets['detection'])
        state_loss = self.ce_loss(preds['state'], targets['state'])
        return det_loss + state_loss

    def _compute_lane_loss(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """Lane segmentation + embedding loss"""
        seg_loss = self.bce_loss(preds['lane_segmentation'], targets['lane_mask'])
        return seg_loss

    def _compute_depth_loss(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """Scale-invariant depth loss"""
        pred_depth = preds['log_depth']
        gt_depth = torch.log(targets['depth'].clamp(min=1e-3))

        diff = pred_depth - gt_depth
        loss = (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)
        return loss

    def _compute_path_loss(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """Path prediction loss"""
        pred_points = preds['path_points']
        gt_points = targets['path_points']
        return self.smooth_l1(pred_points, gt_points)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in detection"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: Dict, targets: Dict) -> torch.Tensor:
        """Compute focal loss for classification"""
        # Simplified implementation
        logits = preds['p3']['cls_logits'] if 'p3' in preds else list(preds.values())[0]['cls_logits']
        targets_cls = targets.get('labels', torch.zeros_like(logits[:, 0]))

        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_cls.float().unsqueeze(1).expand_as(logits),
            reduction='none'
        )

        p = torch.sigmoid(logits)
        pt = targets_cls.unsqueeze(1) * p + (1 - targets_cls.unsqueeze(1)) * (1 - p)
        focal_weight = (1 - pt) ** self.gamma

        loss = self.alpha * focal_weight * ce_loss
        return loss.mean()
