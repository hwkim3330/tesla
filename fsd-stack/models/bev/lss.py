"""
Lift-Splat-Shoot (LSS) BEV Transformation

This is the core BEV transformation used in Tesla FSD.
It projects 2D image features into 3D space using predicted depth,
then collapses to BEV representation.

Reference: "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs"
           (Philion & Fidler, 2020) - https://arxiv.org/abs/2008.05711

Tesla's implementation is more sophisticated but follows the same principle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat
import math


class DepthNet(nn.Module):
    """
    Depth distribution prediction network.

    Predicts a categorical depth distribution for each pixel,
    rather than a single depth value. This allows uncertainty modeling.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 512,
        depth_channels: int = 118,  # Number of depth bins
        context_channels: int = 64,  # Output context channels
    ):
        super().__init__()

        self.depth_channels = depth_channels
        self.context_channels = context_channels

        # Depth distribution head
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels + context_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict depth distribution and context features.

        Args:
            x: Image features (B, C, H, W)

        Returns:
            depth: Depth distribution (B, D, H, W)
            context: Context features (B, C_ctx, H, W)
        """
        out = self.depth_conv(x)
        depth = out[:, :self.depth_channels]
        context = out[:, self.depth_channels:]

        # Softmax over depth dimension
        depth = F.softmax(depth, dim=1)

        return depth, context


class FrustumPooling(nn.Module):
    """
    Frustum-based feature pooling.

    Creates a 3D frustum of features by combining 2D image features
    with predicted depth distributions, then pools into BEV grid.
    """

    def __init__(
        self,
        depth_channels: int = 118,
        bev_size: Tuple[int, int] = (200, 200),
        bev_range: Tuple[float, float, float, float] = (-50.0, -50.0, 50.0, 50.0),
        depth_range: Tuple[float, float, float] = (1.0, 60.0, 1.0),  # min, max, step
    ):
        super().__init__()

        self.depth_channels = depth_channels
        self.bev_size = bev_size
        self.bev_range = bev_range  # x_min, y_min, x_max, y_max

        # Depth bins
        d_min, d_max, d_step = depth_range
        self.depth_bins = torch.arange(d_min, d_max, d_step)

        # BEV grid resolution
        self.bev_res_x = (bev_range[2] - bev_range[0]) / bev_size[0]
        self.bev_res_y = (bev_range[3] - bev_range[1]) / bev_size[1]

    def create_frustum(
        self,
        img_h: int,
        img_w: int,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create frustum grid for a camera.

        Args:
            img_h, img_w: Image dimensions
            intrinsics: Camera intrinsics (B, N, 3, 3)
            extrinsics: Camera extrinsics (B, N, 4, 4)

        Returns:
            frustum: 3D points in ego coordinates (B, N, D, H, W, 3)
        """
        device = intrinsics.device
        B, N = intrinsics.shape[:2]
        D = len(self.depth_bins)

        # Create pixel grid
        xs = torch.linspace(0, img_w - 1, img_w, device=device)
        ys = torch.linspace(0, img_h - 1, img_h, device=device)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')

        # Stack with ones for homogeneous coordinates
        pixel_coords = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # (H, W, 3)

        # Unproject to camera coordinates for each depth
        depth_bins = self.depth_bins.to(device)
        frustum_cam = pixel_coords.unsqueeze(0) * depth_bins.view(-1, 1, 1, 1)  # (D, H, W, 3)

        # Apply inverse intrinsics
        K_inv = torch.inverse(intrinsics)  # (B, N, 3, 3)
        frustum_cam = frustum_cam.view(1, 1, D, img_h, img_w, 3, 1)
        frustum_cam = (K_inv.view(B, N, 1, 1, 1, 3, 3) @ frustum_cam).squeeze(-1)

        # Transform to ego coordinates
        R = extrinsics[..., :3, :3]  # (B, N, 3, 3)
        t = extrinsics[..., :3, 3]   # (B, N, 3)

        frustum_ego = (R.view(B, N, 1, 1, 1, 3, 3) @ frustum_cam.unsqueeze(-1)).squeeze(-1)
        frustum_ego = frustum_ego + t.view(B, N, 1, 1, 1, 3)

        return frustum_ego

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool features into BEV grid.

        Args:
            features: Image features (B, N, C, H, W)
            depth: Depth distributions (B, N, D, H, W)
            intrinsics: Camera intrinsics (B, N, 3, 3)
            extrinsics: Camera extrinsics (B, N, 4, 4)

        Returns:
            bev: BEV features (B, C, Bh, Bw)
        """
        B, N, C, H, W = features.shape
        D = depth.shape[2]
        Bh, Bw = self.bev_size

        # Create frustum points
        frustum = self.create_frustum(H, W, intrinsics, extrinsics)  # (B, N, D, H, W, 3)

        # Lift features to 3D using depth distribution
        # (B, N, C, H, W) * (B, N, D, H, W) -> (B, N, D, C, H, W)
        lifted = features.unsqueeze(2) * depth.unsqueeze(3)
        lifted = rearrange(lifted, 'b n d c h w -> b (n d h w) c')

        # Get frustum coordinates
        frustum_flat = rearrange(frustum, 'b n d h w xyz -> b (n d h w) xyz')

        # Map to BEV grid indices
        x = frustum_flat[..., 0]
        y = frustum_flat[..., 1]

        # Convert to BEV indices
        bev_x = ((x - self.bev_range[0]) / self.bev_res_x).long()
        bev_y = ((y - self.bev_range[1]) / self.bev_res_y).long()

        # Mask valid points
        valid = (
            (bev_x >= 0) & (bev_x < Bw) &
            (bev_y >= 0) & (bev_y < Bh) &
            (frustum_flat[..., 2] > 0)  # In front of camera
        )

        # Splat into BEV grid
        bev = torch.zeros(B, C, Bh, Bw, device=features.device)

        for b in range(B):
            valid_mask = valid[b]
            valid_features = lifted[b, valid_mask]
            valid_x = bev_x[b, valid_mask]
            valid_y = bev_y[b, valid_mask]

            # Accumulate features
            bev_indices = valid_y * Bw + valid_x
            bev[b] = bev[b].view(C, -1).scatter_add(
                1,
                bev_indices.unsqueeze(0).expand(C, -1),
                valid_features.t()
            ).view(C, Bh, Bw)

        return bev


class BEVEncoder(nn.Module):
    """
    BEV feature encoder.

    Processes the raw BEV features with convolutions
    to produce rich BEV representations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        layers = []
        ch = in_channels

        for i in range(num_layers):
            out_ch = out_channels if i == num_layers - 1 else min(ch * 2, out_channels)
            layers.extend([
                nn.Conv2d(ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            if i < num_layers - 1:
                layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            ch = out_ch

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LiftSplatShoot(nn.Module):
    """
    Complete Lift-Splat-Shoot BEV transformation module.

    This is the core of Tesla's multi-camera to BEV transformation.

    Pipeline:
    1. Extract image features with backbone
    2. Predict depth distribution for each pixel
    3. Lift 2D features to 3D frustum using depth
    4. Splat 3D features onto BEV grid
    5. Encode BEV features

    Args:
        in_channels: Input feature channels from backbone
        depth_channels: Number of depth bins
        bev_channels: BEV output channels
        bev_size: BEV grid size (H, W)
        bev_range: BEV range in meters (x_min, y_min, x_max, y_max)
        depth_range: Depth range (min, max, step)
    """

    def __init__(
        self,
        in_channels: int = 512,
        depth_channels: int = 118,
        context_channels: int = 64,
        bev_channels: int = 256,
        bev_size: Tuple[int, int] = (200, 200),
        bev_range: Tuple[float, float, float, float] = (-50.0, -50.0, 50.0, 50.0),
        depth_range: Tuple[float, float, float] = (1.0, 60.0, 0.5),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.depth_channels = depth_channels
        self.context_channels = context_channels
        self.bev_channels = bev_channels
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.depth_range = depth_range

        # Depth prediction network
        self.depth_net = DepthNet(
            in_channels=in_channels,
            depth_channels=depth_channels,
            context_channels=context_channels,
        )

        # Frustum pooling
        self.frustum_pooling = FrustumPooling(
            depth_channels=depth_channels,
            bev_size=bev_size,
            bev_range=bev_range,
            depth_range=depth_range,
        )

        # BEV encoder
        self.bev_encoder = BEVEncoder(
            in_channels=context_channels,
            out_channels=bev_channels,
        )

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform multi-camera features to BEV.

        Args:
            features: Image features from backbone (B, N, C, H, W)
            intrinsics: Camera intrinsic matrices (B, N, 3, 3)
            extrinsics: Camera extrinsic matrices (B, N, 4, 4)

        Returns:
            Dictionary with:
                - 'bev': BEV features (B, C_bev, Bh, Bw)
                - 'depth': Predicted depth distributions (B, N, D, H, W)
        """
        B, N, C, H, W = features.shape

        # Reshape for depth prediction
        features_flat = rearrange(features, 'b n c h w -> (b n) c h w')

        # Predict depth and context
        depth, context = self.depth_net(features_flat)

        # Reshape back
        depth = rearrange(depth, '(b n) d h w -> b n d h w', b=B, n=N)
        context = rearrange(context, '(b n) c h w -> b n c h w', b=B, n=N)

        # Pool to BEV
        bev = self.frustum_pooling(context, depth, intrinsics, extrinsics)

        # Encode BEV
        bev = self.bev_encoder(bev)

        return {
            'bev': bev,
            'depth': depth,
        }

    def get_depth_loss(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute depth supervision loss.

        Args:
            pred_depth: Predicted depth distribution (B, N, D, H, W)
            gt_depth: Ground truth depth (B, N, H, W)

        Returns:
            Depth loss
        """
        B, N, D, H, W = pred_depth.shape

        # Create depth bins
        depth_bins = self.frustum_pooling.depth_bins.to(pred_depth.device)

        # Find closest bin for GT depth
        gt_depth = gt_depth.unsqueeze(2)  # (B, N, 1, H, W)
        depth_bins = depth_bins.view(1, 1, -1, 1, 1)  # (1, 1, D, 1, 1)

        # Soft assignment (Gaussian)
        sigma = 1.0
        weights = torch.exp(-0.5 * ((gt_depth - depth_bins) / sigma) ** 2)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)

        # Cross-entropy style loss
        loss = -weights * torch.log(pred_depth + 1e-6)
        loss = loss.sum(dim=2).mean()

        return loss


# Utility function
def build_lss_bev(
    in_channels: int = 512,
    bev_size: Tuple[int, int] = (200, 200),
    bev_range: float = 50.0,
    depth_range: Tuple[float, float] = (1.0, 60.0),
) -> LiftSplatShoot:
    """Build LSS BEV module with common defaults."""
    return LiftSplatShoot(
        in_channels=in_channels,
        bev_size=bev_size,
        bev_range=(-bev_range, -bev_range, bev_range, bev_range),
        depth_range=(depth_range[0], depth_range[1], 0.5),
    )


if __name__ == '__main__':
    # Test LSS module
    lss = LiftSplatShoot(
        in_channels=256,
        bev_size=(100, 100),
        bev_range=(-25.0, -25.0, 25.0, 25.0),
    )

    # Simulate 6-camera input
    B, N = 2, 6
    features = torch.randn(B, N, 256, 30, 40)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3).clone()
    intrinsics[..., 0, 0] = 500  # fx
    intrinsics[..., 1, 1] = 500  # fy
    intrinsics[..., 0, 2] = 320  # cx
    intrinsics[..., 1, 2] = 240  # cy

    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, 4, 4).clone()

    outputs = lss(features, intrinsics, extrinsics)

    print(f"BEV shape: {outputs['bev'].shape}")
    print(f"Depth shape: {outputs['depth'].shape}")
