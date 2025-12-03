"""
Occupancy Network

Tesla's latest approach (FSD V11+) for 3D scene understanding.
Instead of detecting individual objects, predicts occupancy of 3D voxels.

Key advantages:
1. Can handle arbitrary object shapes (not just boxes)
2. Better at detecting unknown objects
3. More robust for collision avoidance
4. Unified representation for planning

The network predicts for each voxel:
- Occupancy probability (is something there?)
- Semantic class
- Motion flow (for dynamic objects)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class VoxelEncoder(nn.Module):
    """
    Encodes 3D voxel features from BEV + height information
    """

    def __init__(
        self,
        bev_channels: int = 256,
        voxel_channels: int = 128,
        n_heights: int = 16
    ):
        super().__init__()

        self.n_heights = n_heights

        # Lift BEV to 3D with learned height embedding
        self.height_embed = nn.Parameter(torch.randn(n_heights, voxel_channels))

        # Project BEV features
        self.bev_proj = nn.Conv2d(bev_channels, voxel_channels, 1)

        # 3D convolution for voxel processing
        self.voxel_conv = nn.Sequential(
            nn.Conv3d(voxel_channels, voxel_channels, 3, padding=1),
            nn.BatchNorm3d(voxel_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(voxel_channels, voxel_channels, 3, padding=1),
            nn.BatchNorm3d(voxel_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Lift BEV features to 3D voxel space

        Args:
            bev_features: [B, C, H, W] BEV features

        Returns:
            Voxel features [B, C, Z, H, W]
        """
        B, C, H, W = bev_features.shape

        # Project BEV features
        bev_proj = self.bev_proj(bev_features)  # [B, voxel_C, H, W]

        # Expand to 3D with height encoding
        # [B, voxel_C, H, W] -> [B, voxel_C, Z, H, W]
        bev_expanded = bev_proj.unsqueeze(2).expand(-1, -1, self.n_heights, -1, -1)

        # Add height embedding
        height_embed = self.height_embed.view(1, -1, self.n_heights, 1, 1)
        height_embed = height_embed.expand(B, -1, -1, H, W)

        voxel_features = bev_expanded + height_embed

        # 3D convolution
        voxel_features = self.voxel_conv(voxel_features)

        return voxel_features


class SparseConv3D(nn.Module):
    """
    Sparse 3D convolution for efficient voxel processing

    Since most voxels are empty, sparse convolution is much more efficient.
    This is a simplified dense implementation - production would use
    libraries like spconv or MinkowskiEngine.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sparse 3D convolution (simplified dense version)

        Args:
            x: [B, C, Z, H, W] voxel features
            mask: Optional [B, 1, Z, H, W] occupancy mask

        Returns:
            Processed voxel features
        """
        out = self.relu(self.bn(self.conv(x)))

        if mask is not None:
            out = out * mask

        return out


class OccupancyHead(nn.Module):
    """
    Prediction head for occupancy, semantics, and flow
    """

    # Semantic classes for occupancy
    SEMANTIC_CLASSES = [
        'empty',  # 0
        'vehicle',  # 1
        'pedestrian',  # 2
        'cyclist',  # 3
        'road',  # 4
        'sidewalk',  # 5
        'building',  # 6
        'vegetation',  # 7
        'terrain',  # 8
        'pole',  # 9
        'traffic_sign',  # 10
        'traffic_light',  # 11
        'barrier',  # 12
        'construction',  # 13
        'unknown_obstacle'  # 14
    ]

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 15
    ):
        super().__init__()

        self.num_classes = num_classes

        # Occupancy prediction (binary)
        self.occupancy = nn.Conv3d(in_channels, 1, 1)

        # Semantic classification
        self.semantic = nn.Conv3d(in_channels, num_classes, 1)

        # Flow prediction (vx, vy, vz for motion)
        self.flow = nn.Conv3d(in_channels, 3, 1)

        # Instance embedding (for separating instances)
        self.instance_embed = nn.Conv3d(in_channels, 16, 1)

        # Confidence/uncertainty
        self.confidence = nn.Conv3d(in_channels, 1, 1)

    def forward(self, voxel_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict occupancy and attributes for each voxel

        Args:
            voxel_features: [B, C, Z, H, W]

        Returns:
            Dictionary of predictions
        """
        return {
            'occupancy': torch.sigmoid(self.occupancy(voxel_features)),
            'semantic_logits': self.semantic(voxel_features),
            'semantic_probs': F.softmax(self.semantic(voxel_features), dim=1),
            'flow': self.flow(voxel_features),
            'instance_embedding': self.instance_embed(voxel_features),
            'confidence': torch.sigmoid(self.confidence(voxel_features))
        }


class OccupancyNetwork(nn.Module):
    """
    Complete Occupancy Network

    End-to-end network that:
    1. Takes BEV features (from BEV Transformer)
    2. Lifts to 3D voxel space
    3. Processes with 3D convolutions
    4. Predicts occupancy, semantics, and motion

    This is Tesla's key technology for handling the "long tail" of
    unusual objects that aren't in training data.
    """

    def __init__(
        self,
        bev_channels: int = 256,
        voxel_channels: int = 128,
        voxel_size: Tuple[float, float, float] = (0.5, 0.5, 0.5),  # meters
        voxel_range: Tuple[float, float, float, float, float, float] = (-50, -50, -5, 50, 50, 3),
        num_classes: int = 15,
        temporal_frames: int = 4
    ):
        super().__init__()

        self.voxel_size = voxel_size
        self.voxel_range = voxel_range

        # Calculate grid dimensions
        self.grid_x = int((voxel_range[3] - voxel_range[0]) / voxel_size[0])
        self.grid_y = int((voxel_range[4] - voxel_range[1]) / voxel_size[1])
        self.grid_z = int((voxel_range[5] - voxel_range[2]) / voxel_size[2])

        # Voxel encoder
        self.voxel_encoder = VoxelEncoder(bev_channels, voxel_channels, self.grid_z)

        # 3D U-Net style encoder-decoder
        self.encoder1 = nn.Sequential(
            SparseConv3D(voxel_channels, voxel_channels),
            SparseConv3D(voxel_channels, voxel_channels)
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            SparseConv3D(voxel_channels, voxel_channels * 2),
            SparseConv3D(voxel_channels * 2, voxel_channels * 2)
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            SparseConv3D(voxel_channels * 2, voxel_channels * 4),
            SparseConv3D(voxel_channels * 4, voxel_channels * 4)
        )

        # Decoder with skip connections
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(voxel_channels * 4, voxel_channels * 2, 2, stride=2),
            SparseConv3D(voxel_channels * 4, voxel_channels * 2)  # after concat
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(voxel_channels * 2, voxel_channels, 2, stride=2),
            SparseConv3D(voxel_channels * 2, voxel_channels)  # after concat
        )

        # Prediction head
        self.head = OccupancyHead(voxel_channels, num_classes)

        # Temporal fusion for flow prediction
        self.temporal_fusion = nn.GRU(
            voxel_channels, voxel_channels,
            batch_first=True
        )

        # History buffer for temporal processing
        self.register_buffer('voxel_history', None)
        self.temporal_frames = temporal_frames

    def forward(
        self,
        bev_features: torch.Tensor,
        prev_voxels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for occupancy prediction

        Args:
            bev_features: [B, C, H, W] BEV features from BEV Transformer
            prev_voxels: Optional previous voxel features for temporal fusion

        Returns:
            Dictionary containing all predictions
        """
        # Lift to 3D
        voxel_features = self.voxel_encoder(bev_features)

        # Encoder
        e1 = self.encoder1(voxel_features)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Decoder with skip connections
        d2 = self.decoder2[0](e3)  # Upsample
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.decoder2[1](d2)

        d1 = self.decoder1[0](d2)  # Upsample
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.decoder1[1](d1)

        # Predict
        predictions = self.head(d1)

        # Add grid coordinates
        predictions['voxel_coords'] = self.get_voxel_coordinates(bev_features.device)

        return predictions

    def get_voxel_coordinates(self, device: torch.device) -> torch.Tensor:
        """Get real-world coordinates for each voxel"""
        x = torch.linspace(
            self.voxel_range[0] + self.voxel_size[0] / 2,
            self.voxel_range[3] - self.voxel_size[0] / 2,
            self.grid_x, device=device
        )
        y = torch.linspace(
            self.voxel_range[1] + self.voxel_size[1] / 2,
            self.voxel_range[4] - self.voxel_size[1] / 2,
            self.grid_y, device=device
        )
        z = torch.linspace(
            self.voxel_range[2] + self.voxel_size[2] / 2,
            self.voxel_range[5] - self.voxel_size[2] / 2,
            self.grid_z, device=device
        )

        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1)  # [Z, Y, X, 3]

        return coords

    def get_occupied_points(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract occupied voxel positions and their attributes

        Args:
            predictions: Output from forward pass
            threshold: Occupancy threshold

        Returns:
            positions: [N, 3] xyz coordinates of occupied voxels
            attributes: [N, num_classes] semantic probabilities
        """
        occupancy = predictions['occupancy']
        semantic = predictions['semantic_probs']
        coords = predictions['voxel_coords']

        # Find occupied voxels
        mask = occupancy.squeeze(1) > threshold  # [B, Z, H, W]

        # Get positions (for first batch element)
        positions = coords[mask[0]]

        # Get semantic attributes
        attributes = semantic[0].permute(1, 2, 3, 0)[mask[0]]  # [N, num_classes]

        return positions, attributes


class CollisionChecker(nn.Module):
    """
    Collision checking using occupancy predictions

    Given the ego vehicle trajectory, checks for potential collisions
    with occupied voxels.
    """

    def __init__(
        self,
        ego_length: float = 4.7,
        ego_width: float = 1.85,
        ego_height: float = 1.5,
        safety_margin: float = 0.5
    ):
        super().__init__()

        self.ego_length = ego_length
        self.ego_width = ego_width
        self.ego_height = ego_height
        self.safety_margin = safety_margin

    def check_trajectory(
        self,
        trajectory: torch.Tensor,
        occupancy: torch.Tensor,
        voxel_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check trajectory for collisions

        Args:
            trajectory: [T, 3] future ego positions (x, y, yaw)
            occupancy: [Z, H, W] occupancy grid
            voxel_coords: [Z, H, W, 3] voxel coordinates

        Returns:
            collision_free: [T] boolean for each timestep
            min_distance: [T] minimum distance to obstacle
        """
        T = trajectory.shape[0]
        collision_free = torch.ones(T, dtype=torch.bool, device=trajectory.device)
        min_distances = torch.full((T,), float('inf'), device=trajectory.device)

        # Get occupied positions
        occupied_mask = occupancy > 0.5
        occupied_positions = voxel_coords[occupied_mask]  # [N, 3]

        if occupied_positions.shape[0] == 0:
            return collision_free, min_distances

        for t in range(T):
            pos = trajectory[t, :2]  # x, y

            # Compute distances to occupied voxels (simplified - ignores rotation)
            distances = torch.norm(occupied_positions[:, :2] - pos, dim=1)

            min_dist = distances.min()
            min_distances[t] = min_dist

            # Check collision (simplified bounding box)
            half_diag = math.sqrt(
                (self.ego_length / 2 + self.safety_margin) ** 2 +
                (self.ego_width / 2 + self.safety_margin) ** 2
            )

            if min_dist < half_diag:
                collision_free[t] = False

        return collision_free, min_distances


class OccupancyLoss(nn.Module):
    """
    Multi-task loss for occupancy network training
    """

    def __init__(
        self,
        occupancy_weight: float = 1.0,
        semantic_weight: float = 1.0,
        flow_weight: float = 0.5,
        lovasz_weight: float = 0.5
    ):
        super().__init__()

        self.occupancy_weight = occupancy_weight
        self.semantic_weight = semantic_weight
        self.flow_weight = flow_weight
        self.lovasz_weight = lovasz_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss

        Args:
            predictions: Network predictions
            targets: Ground truth labels

        Returns:
            Dictionary of individual losses and total
        """
        losses = {}

        # Occupancy loss (binary cross entropy)
        if 'occupancy_gt' in targets:
            losses['occupancy'] = self.bce_loss(
                predictions['occupancy'],
                targets['occupancy_gt']
            ) * self.occupancy_weight

        # Semantic loss (cross entropy)
        if 'semantic_gt' in targets:
            losses['semantic'] = self.ce_loss(
                predictions['semantic_logits'],
                targets['semantic_gt']
            ) * self.semantic_weight

        # Flow loss (L1)
        if 'flow_gt' in targets:
            # Only compute on occupied voxels
            mask = targets['occupancy_gt'] > 0.5
            if mask.sum() > 0:
                losses['flow'] = self.l1_loss(
                    predictions['flow'][mask],
                    targets['flow_gt'][mask]
                ) * self.flow_weight

        # Total loss
        losses['total'] = sum(losses.values())

        return losses
