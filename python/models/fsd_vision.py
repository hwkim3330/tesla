"""
Tesla FSD Vision - Complete End-to-End System

This module integrates all components into a complete vision system:
1. HydraNet (multi-task backbone + heads)
2. BEV Transformer (2D to 3D projection)
3. Occupancy Network (3D scene understanding)
4. Temporal fusion (video processing)

This is the full pipeline that processes camera images and outputs
all the information needed for autonomous driving.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .backbone import RegNetBackbone, FPN
from .hydranet import HydraNet
from .bev_transformer import BEVTransformer, TemporalBEVFusion
from .occupancy_network import OccupancyNetwork, CollisionChecker


@dataclass
class CameraConfig:
    """Camera configuration for multi-camera setup"""
    name: str
    width: int
    height: int
    fov: float  # Field of view in degrees
    position: Tuple[float, float, float]  # x, y, z relative to ego
    rotation: Tuple[float, float, float]  # roll, pitch, yaw


# Tesla's 8 camera configuration
TESLA_CAMERA_CONFIG = [
    CameraConfig('front_main', 1280, 960, 50, (2.0, 0.0, 1.5), (0, 0, 0)),
    CameraConfig('front_narrow', 1280, 960, 35, (2.0, 0.0, 1.5), (0, 0, 0)),
    CameraConfig('front_wide', 1280, 960, 120, (2.0, 0.0, 1.5), (0, 0, 0)),
    CameraConfig('front_left', 1280, 960, 80, (2.0, 0.8, 1.5), (0, 0, -60)),
    CameraConfig('front_right', 1280, 960, 80, (2.0, -0.8, 1.5), (0, 0, 60)),
    CameraConfig('side_left', 1280, 960, 80, (0.5, 0.9, 1.2), (0, 0, -90)),
    CameraConfig('side_right', 1280, 960, 80, (0.5, -0.9, 1.2), (0, 0, 90)),
    CameraConfig('rear', 1280, 960, 80, (-0.5, 0.0, 1.5), (0, 0, 180)),
]


class CameraEncoder(nn.Module):
    """
    Per-camera feature encoder

    Processes each camera image through the shared backbone
    but maintains camera-specific information.
    """

    def __init__(
        self,
        backbone: nn.Module,
        fpn: nn.Module,
        d_model: int = 256
    ):
        super().__init__()

        self.backbone = backbone
        self.fpn = fpn

        # Camera embedding to differentiate cameras
        self.camera_embed = nn.Embedding(8, d_model)

    def forward(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Encode camera images

        Args:
            images: [B, N_cams, 3, H, W] multi-camera images
            camera_ids: [N_cams] camera indices

        Returns:
            Dictionary of features per camera
        """
        B, N_cams, C, H, W = images.shape

        # Flatten batch and cameras
        images_flat = images.view(B * N_cams, C, H, W)

        # Extract features
        backbone_features = self.backbone(images_flat)
        fpn_features = self.fpn(backbone_features)

        # Add camera embeddings
        cam_embeds = self.camera_embed(camera_ids)  # [N_cams, d_model]
        cam_embeds = cam_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, N_cams, d_model]

        # Reshape features back
        features_per_cam = {}
        for level, feat in fpn_features.items():
            _, c, h, w = feat.shape
            feat = feat.view(B, N_cams, c, h, w)

            # Add camera embedding
            cam_embed_expanded = cam_embeds.view(B, N_cams, -1, 1, 1).expand(-1, -1, -1, h, w)
            feat = feat + cam_embed_expanded[:, :, :c, :, :]

            features_per_cam[level] = feat

        return features_per_cam


class TeslaFSDVision(nn.Module):
    """
    Tesla FSD Vision - Complete End-to-End Vision System

    This is the main class that combines all components:

    Pipeline:
    ┌──────────────────────────────────────────────────────────┐
    │                    8 Camera Images                        │
    └────────────────────────┬─────────────────────────────────┘
                             │
    ┌────────────────────────▼─────────────────────────────────┐
    │              Camera Encoder (RegNet + FPN)                │
    │                    (Per-camera features)                  │
    └────────────────────────┬─────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌──────▼──────┐ ┌───────▼───────┐
    │   HydraNet    │ │    BEV      │ │   Temporal    │
    │   (2D tasks)  │ │ Transformer │ │    Fusion     │
    └───────┬───────┘ └──────┬──────┘ └───────┬───────┘
            │                │                │
            │         ┌──────▼──────┐         │
            │         │  Occupancy  │◄────────┘
            │         │   Network   │
            │         └──────┬──────┘
            │                │
    ┌───────▼────────────────▼─────────────────────────────────┐
    │                      Outputs                              │
    │  - Object Detections (2D + 3D)                           │
    │  - Traffic Light State + Distance                         │
    │  - Lane Lines + Drivable Area                            │
    │  - Depth Map                                              │
    │  - Semantic Segmentation                                  │
    │  - Occupancy Grid (3D)                                    │
    │  - Object Velocities                                      │
    │  - Predicted Path                                         │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        backbone_name: str = 'tesla_custom',
        d_model: int = 256,
        n_cameras: int = 8,
        bev_h: int = 200,
        bev_w: int = 200,
        bev_resolution: float = 0.5,
        num_object_classes: int = 10,
        num_semantic_classes: int = 20,
        temporal_frames: int = 4
    ):
        super().__init__()

        self.n_cameras = n_cameras
        self.d_model = d_model
        self.temporal_frames = temporal_frames

        # Backbone (shared across cameras)
        self.backbone = RegNetBackbone(config_name=backbone_name)
        self.fpn = FPN(self.backbone.out_channels, d_model)

        # Camera encoder
        self.camera_encoder = CameraEncoder(self.backbone, self.fpn, d_model)

        # HydraNet for 2D perception tasks
        self.hydranet = HydraNet(
            backbone_name=backbone_name,
            fpn_channels=d_model,
            num_object_classes=num_object_classes,
            num_semantic_classes=num_semantic_classes
        )

        # BEV Transformer for 3D projection
        self.bev_transformer = BEVTransformer(
            d_model=d_model,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_resolution=bev_resolution,
            n_cameras=n_cameras
        )

        # Temporal BEV fusion
        self.temporal_fusion = TemporalBEVFusion(
            d_model=d_model,
            n_frames=temporal_frames
        )

        # Occupancy Network
        self.occupancy_net = OccupancyNetwork(
            bev_channels=d_model,
            temporal_frames=temporal_frames
        )

        # Collision checker
        self.collision_checker = CollisionChecker()

        # BEV history for temporal processing
        self.bev_history = []

    def forward(
        self,
        images: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
        ego_motion: Optional[torch.Tensor] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through the FSD Vision system

        Args:
            images: [B, N_cams, 3, H, W] multi-camera images
            camera_intrinsics: [B, N_cams, 3, 3] intrinsic matrices
            camera_extrinsics: [B, N_cams, 4, 4] extrinsic matrices
            ego_motion: [B, 6] ego motion (for temporal alignment)
            return_all: Whether to return all intermediate outputs

        Returns:
            Dictionary containing all perception outputs
        """
        B, N_cams, C, H, W = images.shape
        outputs = {}

        # 1. Camera feature extraction
        camera_ids = torch.arange(N_cams, device=images.device)
        cam_features = self.camera_encoder(images, camera_ids)

        # 2. 2D perception via HydraNet (per-camera)
        # Use front main camera for primary 2D tasks
        front_main = images[:, 0]  # [B, 3, H, W]
        hydranet_outputs = self.hydranet(front_main, return_features=True)
        outputs['hydranet'] = hydranet_outputs

        # 3. BEV projection
        # Prepare features for BEV transformer
        multi_cam_features = [cam_features['p4'][:, i] for i in range(N_cams)]
        bev_features = self.bev_transformer(
            multi_cam_features,
            camera_intrinsics,
            camera_extrinsics
        )

        # 4. Temporal fusion
        self.bev_history.append(bev_features)
        if len(self.bev_history) > self.temporal_frames:
            self.bev_history.pop(0)

        if len(self.bev_history) >= 2:
            fused_bev = self.temporal_fusion(self.bev_history, ego_motion)
        else:
            fused_bev = bev_features

        outputs['bev_features'] = fused_bev

        # 5. Occupancy prediction
        occupancy_outputs = self.occupancy_net(fused_bev)
        outputs['occupancy'] = occupancy_outputs

        # 6. Extract key outputs
        outputs['detections'] = self._extract_detections(hydranet_outputs)
        outputs['traffic_lights'] = self._extract_traffic_lights(hydranet_outputs)
        outputs['lanes'] = self._extract_lanes(hydranet_outputs)
        outputs['depth'] = hydranet_outputs['depth']['depth']
        outputs['path'] = hydranet_outputs['path']

        return outputs

    def _extract_detections(self, hydranet_outputs: Dict) -> List[Dict]:
        """Extract object detections from HydraNet output"""
        # Simplified - would include NMS and proper decoding
        return []

    def _extract_traffic_lights(self, hydranet_outputs: Dict) -> List[Dict]:
        """Extract traffic light detections"""
        return []

    def _extract_lanes(self, hydranet_outputs: Dict) -> Dict:
        """Extract lane information"""
        return hydranet_outputs.get('lanes', {})

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Run inference and return processed results

        This is the main entry point for real-time inference.
        Returns results in a format ready for visualization.
        """
        outputs = self.forward(
            images,
            camera_intrinsics,
            camera_extrinsics
        )

        results = {
            'objects': [],
            'traffic_lights': [],
            'lanes': {},
            'depth_map': None,
            'occupancy_grid': None,
            'predicted_path': None,
            'ego_speed': 0.0
        }

        # Process detections
        if 'detections' in outputs:
            results['objects'] = outputs['detections']

        # Process traffic lights
        if 'traffic_lights' in outputs:
            for tl in outputs['traffic_lights']:
                results['traffic_lights'].append({
                    'state': tl.get('state', 'unknown'),
                    'distance': tl.get('distance', 0),
                    'confidence': tl.get('confidence', 0),
                    'relevance': tl.get('relevance', 0)
                })

        # Depth map
        if 'depth' in outputs:
            results['depth_map'] = outputs['depth'].cpu().numpy()

        # Occupancy
        if 'occupancy' in outputs:
            occ = outputs['occupancy']
            results['occupancy_grid'] = {
                'occupancy': occ['occupancy'].cpu().numpy(),
                'semantic': occ['semantic_probs'].argmax(dim=1).cpu().numpy(),
                'flow': occ['flow'].cpu().numpy()
            }

        # Path
        if 'path' in outputs:
            path = outputs['path']
            results['predicted_path'] = path['path_points'].cpu().numpy()

        return results

    def check_collision(
        self,
        trajectory: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Check if a planned trajectory will result in collision

        Args:
            trajectory: [T, 3] planned positions (x, y, yaw)

        Returns:
            is_safe: Whether trajectory is collision-free
            min_distance: Minimum distance to any obstacle
        """
        if not self.bev_history:
            return True, float('inf')

        # Get latest occupancy prediction
        bev = self.bev_history[-1]
        occ_outputs = self.occupancy_net(bev)

        collision_free, min_distances = self.collision_checker.check_trajectory(
            trajectory,
            occ_outputs['occupancy'].squeeze(),
            occ_outputs['voxel_coords']
        )

        is_safe = collision_free.all().item()
        min_dist = min_distances.min().item()

        return is_safe, min_dist

    def get_visualization_data(self) -> Dict:
        """
        Get data formatted for Tesla-style visualization

        Returns data suitable for rendering in the UI:
        - Object bounding boxes with distance labels
        - Traffic light indicators
        - Lane lines
        - Predicted path
        - BEV representation
        """
        return {
            'bev_grid': self._get_bev_visualization(),
            'camera_overlay': self._get_camera_overlay(),
            'stats': self._get_stats()
        }

    def _get_bev_visualization(self) -> Dict:
        """Generate BEV visualization data"""
        return {
            'grid_size': (self.bev_transformer.bev_h, self.bev_transformer.bev_w),
            'resolution': self.bev_transformer.bev_resolution,
            'ego_position': (self.bev_transformer.bev_w // 2, self.bev_transformer.bev_h - 20)
        }

    def _get_camera_overlay(self) -> Dict:
        """Generate camera view overlay data"""
        return {
            'detection_boxes': [],
            'traffic_lights': [],
            'lane_lines': [],
            'predicted_path': []
        }

    def _get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'fps': 0,
            'latency_ms': 0,
            'objects_detected': 0,
            'confidence': 0
        }


def create_fsd_vision(
    config: str = 'default',
    pretrained: bool = False
) -> TeslaFSDVision:
    """
    Factory function to create FSD Vision model

    Args:
        config: Configuration name ('default', 'small', 'large')
        pretrained: Whether to load pretrained weights

    Returns:
        Configured TeslaFSDVision model
    """
    configs = {
        'small': {
            'backbone_name': 'regnet_y_400mf',
            'd_model': 128,
            'bev_h': 100,
            'bev_w': 100
        },
        'default': {
            'backbone_name': 'tesla_custom',
            'd_model': 256,
            'bev_h': 200,
            'bev_w': 200
        },
        'large': {
            'backbone_name': 'regnet_y_8gf',
            'd_model': 512,
            'bev_h': 400,
            'bev_w': 400
        }
    }

    cfg = configs.get(config, configs['default'])
    model = TeslaFSDVision(**cfg)

    if pretrained:
        # Would load pretrained weights here
        pass

    return model
