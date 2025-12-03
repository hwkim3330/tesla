"""
Tesla FSD Stack - Neural Network Models

This module contains all the neural network components for the FSD stack:
- Backbone: Image encoders (RegNet, EfficientNet, ViT)
- Perception: Detection heads (objects, lanes, traffic lights, depth)
- BEV: Bird's Eye View transformation (LSS, BEVFormer)
- Temporal: Video/temporal fusion
- Planning: Trajectory planning
- E2E: End-to-end FSD network
"""

from .backbone import RegNetBackbone, EfficientNetBackbone, ViTBackbone
from .perception import ObjectDetector, LaneDetector, TrafficLightDetector, DepthEstimator
from .bev import LiftSplatShoot, BEVFormer, SimpleBEV
from .temporal import TemporalFusion, VideoTransformer
from .planning import TrajectoryPlanner, DiffusionPolicy
from .e2e import FSDNetwork, FSDModel

__all__ = [
    # Backbone
    'RegNetBackbone', 'EfficientNetBackbone', 'ViTBackbone',
    # Perception
    'ObjectDetector', 'LaneDetector', 'TrafficLightDetector', 'DepthEstimator',
    # BEV
    'LiftSplatShoot', 'BEVFormer', 'SimpleBEV',
    # Temporal
    'TemporalFusion', 'VideoTransformer',
    # Planning
    'TrajectoryPlanner', 'DiffusionPolicy',
    # E2E
    'FSDNetwork', 'FSDModel',
]
