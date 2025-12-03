"""
Tesla FSD Detection Heads
Multi-task learning heads for various perception tasks

Each head shares the backbone features but specializes in different tasks:
- Object Detection: Vehicles, pedestrians, cyclists
- Traffic Light: Detection + classification (color state)
- Lane Detection: Lane lines, road edges, path prediction
- Depth Estimation: Monocular depth for distance calculation
- Semantic Segmentation: Drivable area, road structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class ConvBNReLU(nn.Module):
    """Conv + BatchNorm + ReLU block"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ObjectDetectionHead(nn.Module):
    """
    Object Detection Head - YOLO/FCOS style anchor-free detection

    Detects: Vehicles, Pedestrians, Cyclists, Motorcycles, Trucks, Buses

    Output per location:
    - Classification scores for each class
    - Bounding box regression (x, y, w, h)
    - Centerness score (for FCOS-style detection)
    - 3D attributes: depth, dimensions, orientation
    """

    NUM_CLASSES = 10  # vehicle, pedestrian, cyclist, motorcycle, truck, bus, etc.
    ATTRIBUTES_3D = 7  # depth, width, height, length, yaw, pitch, roll

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        num_convs: int = 4
    ):
        super().__init__()

        self.num_classes = num_classes

        # Shared convolution tower
        cls_tower = []
        reg_tower = []

        for i in range(num_convs):
            cls_tower.append(ConvBNReLU(in_channels, in_channels))
            reg_tower.append(ConvBNReLU(in_channels, in_channels))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        # Classification head
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # Bounding box regression (l, t, r, b)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

        # Centerness prediction
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # 3D attributes prediction
        self.attr_3d = nn.Conv2d(in_channels, self.ATTRIBUTES_3D, 3, padding=1)

        # Velocity prediction (vx, vy, vz)
        self.velocity = nn.Conv2d(in_channels, 3, 3, padding=1)

        # Object confidence
        self.objectness = nn.Conv2d(in_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_logits, self.bbox_pred, self.centerness,
                  self.attr_3d, self.velocity, self.objectness]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for object detection

        Args:
            features: Multi-scale feature maps from FPN

        Returns:
            Dictionary containing predictions at each scale
        """
        outputs = {}

        for level, feat in features.items():
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            outputs[level] = {
                'cls_logits': self.cls_logits(cls_feat),
                'bbox_pred': F.relu(self.bbox_pred(reg_feat)),  # distances must be positive
                'centerness': self.centerness(reg_feat),
                'objectness': self.objectness(cls_feat),
                'attr_3d': self.attr_3d(reg_feat),
                'velocity': self.velocity(reg_feat)
            }

        return outputs


class TrafficLightHead(nn.Module):
    """
    Traffic Light Detection + Classification Head

    Tasks:
    1. Detect traffic light locations
    2. Classify light state (red, yellow, green, off, unknown)
    3. Classify light type (circular, arrow_left, arrow_right, arrow_straight)
    4. Estimate distance
    5. Relevance classification (is this light relevant to ego lane?)
    """

    LIGHT_STATES = ['red', 'yellow', 'green', 'off', 'unknown']
    LIGHT_TYPES = ['circular', 'arrow_left', 'arrow_right', 'arrow_straight', 'pedestrian', 'other']

    def __init__(self, in_channels: int = 256, num_convs: int = 4):
        super().__init__()

        # Feature extraction tower
        tower = []
        for _ in range(num_convs):
            tower.append(ConvBNReLU(in_channels, in_channels))
        self.tower = nn.Sequential(*tower)

        # Detection head (is there a traffic light?)
        self.detect = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Bounding box
        self.bbox = nn.Conv2d(in_channels, 4, 3, padding=1)

        # Light state classification
        self.state_cls = nn.Conv2d(in_channels, len(self.LIGHT_STATES), 3, padding=1)

        # Light type classification
        self.type_cls = nn.Conv2d(in_channels, len(self.LIGHT_TYPES), 3, padding=1)

        # Distance estimation (in meters)
        self.distance = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Relevance score (is this light for our lane?)
        self.relevance = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Time until state change (optional, if detectable from pattern)
        self.time_to_change = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process features for traffic light detection"""
        outputs = {}

        # Use higher resolution features for small objects like traffic lights
        for level in ['p3', 'p4']:
            if level in features:
                feat = self.tower(features[level])

                outputs[level] = {
                    'detection': torch.sigmoid(self.detect(feat)),
                    'bbox': self.bbox(feat),
                    'state': self.state_cls(feat),
                    'type': self.type_cls(feat),
                    'distance': F.relu(self.distance(feat)),  # distance is positive
                    'relevance': torch.sigmoid(self.relevance(feat)),
                    'time_to_change': F.relu(self.time_to_change(feat))
                }

        return outputs


class LaneDetectionHead(nn.Module):
    """
    Lane Detection Head

    Tasks:
    1. Lane line segmentation (pixel-wise)
    2. Lane instance embedding
    3. Lane line type classification (solid, dashed, double, etc.)
    4. Lane color classification (white, yellow)
    5. Road edge detection
    6. Drivable area segmentation
    """

    LANE_TYPES = ['solid', 'dashed', 'double_solid', 'double_dashed', 'solid_dashed', 'curb', 'road_edge']
    LANE_COLORS = ['white', 'yellow', 'blue', 'other']

    def __init__(self, in_channels: int = 256, embedding_dim: int = 16):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Decoder for high-resolution output
        self.decoder = nn.Sequential(
            ConvBNReLU(in_channels, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(64, 32)
        )

        # Lane line segmentation (binary: lane vs background)
        self.lane_seg = nn.Conv2d(32, 1, 1)

        # Instance embedding for lane differentiation
        self.lane_embedding = nn.Conv2d(32, embedding_dim, 1)

        # Lane type classification
        self.lane_type = nn.Conv2d(32, len(self.LANE_TYPES), 1)

        # Lane color classification
        self.lane_color = nn.Conv2d(32, len(self.LANE_COLORS), 1)

        # Drivable area segmentation
        self.drivable = nn.Conv2d(32, 3, 1)  # ego lane, other lane, not drivable

        # Road edge detection
        self.road_edge = nn.Conv2d(32, 1, 1)

        # Lane offset (distance from ego vehicle to lane center)
        self.lane_offset = nn.Conv2d(32, 2, 1)  # offset in x, y

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process features for lane detection"""
        # Use P2 for highest resolution
        feat = features['p2']
        decoded = self.decoder(feat)

        return {
            'lane_segmentation': torch.sigmoid(self.lane_seg(decoded)),
            'lane_embedding': self.lane_embedding(decoded),
            'lane_type': self.lane_type(decoded),
            'lane_color': self.lane_color(decoded),
            'drivable_area': self.drivable(decoded),
            'road_edge': torch.sigmoid(self.road_edge(decoded)),
            'lane_offset': self.lane_offset(decoded)
        }


class DepthEstimationHead(nn.Module):
    """
    Monocular Depth Estimation Head

    Predicts depth for every pixel - critical for:
    1. Distance calculation to objects
    2. 3D scene reconstruction
    3. BEV projection
    4. Collision avoidance

    Uses scale-invariant depth prediction with uncertainty estimation
    """

    def __init__(self, in_channels: int = 256, max_depth: float = 200.0):
        super().__init__()

        self.max_depth = max_depth

        # Multi-scale decoder
        self.decoder4 = nn.Sequential(
            ConvBNReLU(in_channels, 256),
            ConvBNReLU(256, 256)
        )

        self.decoder3 = nn.Sequential(
            ConvBNReLU(256 + in_channels, 128),
            ConvBNReLU(128, 128)
        )

        self.decoder2 = nn.Sequential(
            ConvBNReLU(128 + in_channels, 64),
            ConvBNReLU(64, 64)
        )

        # Depth prediction (log-space for better gradient flow)
        self.depth_pred = nn.Conv2d(64, 1, 3, padding=1)

        # Uncertainty estimation (aleatoric uncertainty)
        self.uncertainty = nn.Conv2d(64, 1, 3, padding=1)

        # Surface normal prediction (helps with depth consistency)
        self.normal = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict depth and uncertainty"""
        # Decode from coarse to fine
        d4 = self.decoder4(features['p5'])
        d4_up = F.interpolate(d4, size=features['p4'].shape[2:], mode='bilinear', align_corners=False)

        d3 = self.decoder3(torch.cat([d4_up, features['p4']], dim=1))
        d3_up = F.interpolate(d3, size=features['p3'].shape[2:], mode='bilinear', align_corners=False)

        d2 = self.decoder2(torch.cat([d3_up, features['p3']], dim=1))

        # Predict depth in log space, then convert
        log_depth = self.depth_pred(d2)
        depth = self.max_depth * torch.sigmoid(log_depth)

        # Uncertainty (log variance)
        log_var = self.uncertainty(d2)
        uncertainty = torch.exp(log_var)

        # Surface normals (normalized)
        normals = self.normal(d2)
        normals = F.normalize(normals, dim=1)

        return {
            'depth': depth,
            'log_depth': log_depth,
            'uncertainty': uncertainty,
            'surface_normal': normals
        }


class SemanticSegmentationHead(nn.Module):
    """
    Semantic Segmentation Head

    Segments the scene into semantic classes:
    - Road, Sidewalk, Building, Wall, Fence
    - Pole, Traffic Light, Traffic Sign
    - Vegetation, Terrain, Sky
    - Person, Rider, Car, Truck, Bus, Train
    - Motorcycle, Bicycle
    """

    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic_light', 'traffic_sign', 'vegetation',
        'terrain', 'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle', 'unknown'
    ]

    def __init__(self, in_channels: int = 256, num_classes: int = 20):
        super().__init__()

        self.num_classes = num_classes

        # ASPP-like multi-scale context
        self.aspp1 = nn.Conv2d(in_channels, 256, 1)
        self.aspp2 = nn.Conv2d(in_channels, 256, 3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, 256, 3, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, 256, 3, padding=18, dilation=18)
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1)
        )

        self.aspp_combine = nn.Sequential(
            nn.Conv2d(256 * 5, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(256 + in_channels, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64)
        )

        # Final classification
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Semantic segmentation prediction"""
        x = features['p5']
        size = x.shape[2:]

        # ASPP
        a1 = self.aspp1(x)
        a2 = self.aspp2(x)
        a3 = self.aspp3(x)
        a4 = self.aspp4(x)
        a5 = self.aspp_pool(x)
        a5 = F.interpolate(a5, size=size, mode='bilinear', align_corners=False)

        aspp_out = self.aspp_combine(torch.cat([a1, a2, a3, a4, a5], dim=1))

        # Decoder with skip connection
        aspp_up = F.interpolate(aspp_out, size=features['p3'].shape[2:], mode='bilinear', align_corners=False)
        decoded = self.decoder(torch.cat([aspp_up, features['p3']], dim=1))

        logits = self.classifier(decoded)

        return {
            'segmentation': logits,
            'probabilities': F.softmax(logits, dim=1)
        }


class PathPredictionHead(nn.Module):
    """
    Path Prediction Head

    Predicts the future trajectory/path of the ego vehicle
    Used for navigation visualization (the green path in Tesla UI)
    """

    def __init__(self, in_channels: int = 256, num_points: int = 50):
        super().__init__()

        self.num_points = num_points

        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Path prediction network
        self.path_net = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 3)  # x, y, confidence for each point
        )

        # Curvature prediction
        self.curvature = nn.Linear(in_channels, 1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict ego vehicle path"""
        x = features['p5']
        global_feat = self.global_pool(x).flatten(1)

        # Predict path points
        path = self.path_net(global_feat)
        path = path.view(-1, self.num_points, 3)

        # Split into coordinates and confidence
        path_xy = path[:, :, :2]
        path_conf = torch.sigmoid(path[:, :, 2:])

        # Curvature
        curvature = self.curvature(global_feat)

        return {
            'path_points': path_xy,
            'path_confidence': path_conf,
            'curvature': curvature
        }
