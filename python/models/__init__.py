# Tesla FSD Vision Models
# HydraNet-based Multi-Task Learning Architecture

from .backbone import RegNetBackbone, EfficientNetBackbone
from .hydranet import HydraNet
from .detection_heads import (
    ObjectDetectionHead,
    TrafficLightHead,
    LaneDetectionHead,
    DepthEstimationHead,
    SemanticSegmentationHead
)
from .bev_transformer import BEVTransformer, BEVEncoder
from .occupancy_network import OccupancyNetwork
from .fsd_vision import TeslaFSDVision

__all__ = [
    'RegNetBackbone',
    'EfficientNetBackbone',
    'HydraNet',
    'ObjectDetectionHead',
    'TrafficLightHead',
    'LaneDetectionHead',
    'DepthEstimationHead',
    'SemanticSegmentationHead',
    'BEVTransformer',
    'BEVEncoder',
    'OccupancyNetwork',
    'TeslaFSDVision'
]
