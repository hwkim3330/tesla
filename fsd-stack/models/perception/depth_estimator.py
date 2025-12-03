"""
Monocular Depth Estimation for Distance Calculation

Camera-based depth estimation using neural networks.
Provides distance information without LiDAR.

Key approaches:
- Dense depth prediction (full image)
- Object-centric depth (per-detection)
- Multi-scale depth with uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class DepthDecoder(nn.Module):
    """
    Multi-scale depth decoder.

    Takes FPN features and produces depth maps at multiple scales.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 1,
        scales: List[int] = [0, 1, 2, 3],
    ):
        super().__init__()

        self.scales = scales
        self.num_scales = len(scales)

        # Upsampling layers
        self.convs = nn.ModuleDict()

        for i in range(4, 0, -1):
            # Upconv layers
            in_ch = in_channels[i-1] if i == 4 else in_channels[i-1] + out_ch
            out_ch = in_channels[i-1] // 2

            self.convs[f'upconv_{i}'] = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ELU(inplace=True),
            )

            # Output layer for this scale
            if i - 1 in scales:
                self.convs[f'dispconv_{i-1}'] = nn.Sequential(
                    nn.Conv2d(out_ch, out_channels, 3, padding=1),
                    nn.Sigmoid(),
                )

    def forward(self, features: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Decode features to multi-scale depth.

        Args:
            features: FPN features [P0, P1, P2, P3]

        Returns:
            Dictionary of depth maps at each scale
        """
        outputs = {}
        x = features[-1]

        for i in range(4, 0, -1):
            x = self.convs[f'upconv_{i}'](x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            if i > 1:
                x = torch.cat([x, features[i-2]], dim=1)

            if i - 1 in self.scales:
                outputs[i-1] = self.convs[f'dispconv_{i-1}'](x)

        return outputs


class MonoDepth(nn.Module):
    """
    Monocular depth estimation network.

    Based on MonoDepth2 architecture with improvements for autonomous driving.

    Args:
        encoder: Backbone network
        min_depth: Minimum depth value
        max_depth: Maximum depth value
        scales: Output scales
    """

    def __init__(
        self,
        in_channels: List[int] = [64, 128, 256, 512],
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        scales: List[int] = [0, 1, 2, 3],
    ):
        super().__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scales = scales

        # Depth decoder
        self.decoder = DepthDecoder(in_channels, out_channels=1, scales=scales)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict depth from features.

        Args:
            features: Multi-scale backbone features

        Returns:
            Dictionary with depth predictions at multiple scales
        """
        # Decode to disparity
        disp_outputs = self.decoder(features)

        # Convert disparity to depth
        depth_outputs = {}
        for scale, disp in disp_outputs.items():
            # Scale disparity to depth
            min_disp = 1 / self.max_depth
            max_disp = 1 / self.min_depth
            scaled_disp = min_disp + (max_disp - min_disp) * disp
            depth = 1 / scaled_disp
            depth_outputs[f'depth_{scale}'] = depth

        return depth_outputs

    def get_depth_at_points(
        self,
        depth: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample depth values at specific 2D points.

        Args:
            depth: Depth map (B, 1, H, W)
            points: 2D points (B, N, 2) in pixel coordinates

        Returns:
            Depth values at points (B, N)
        """
        B, _, H, W = depth.shape
        N = points.shape[1]

        # Normalize to [-1, 1] for grid_sample
        grid = points.clone()
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        grid = grid.view(B, N, 1, 2)

        # Sample
        sampled = F.grid_sample(depth, grid, mode='bilinear', align_corners=True)
        return sampled.view(B, N)


class DepthEstimator(nn.Module):
    """
    Full depth estimation module with distance computation.

    Combines dense depth prediction with object-specific distance estimation.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
    ):
        super().__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth

        # Dense depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

        # Uncertainty head (optional)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        features: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate depth from features.

        Args:
            features: Image features (B, C, H, W)
            bboxes: Optional bounding boxes for object-centric depth (B, N, 4)

        Returns:
            Dictionary with:
                - 'depth': Dense depth map
                - 'uncertainty': Depth uncertainty
                - 'object_depths': Per-object depths (if bboxes provided)
        """
        # Dense depth prediction
        disp = self.depth_head(features)

        # Convert to depth
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp

        # Uncertainty
        uncertainty = self.uncertainty_head(features)

        outputs = {
            'depth': depth,
            'uncertainty': uncertainty,
        }

        # Object-centric depth
        if bboxes is not None:
            object_depths = self._compute_object_depths(depth, bboxes)
            outputs['object_depths'] = object_depths

        return outputs

    def _compute_object_depths(
        self,
        depth: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute depth for each detected object.

        Uses the median depth within the bounding box.
        """
        B, _, H, W = depth.shape
        N = bboxes.shape[1]

        object_depths = torch.zeros(B, N, device=depth.device)

        for b in range(B):
            for n in range(N):
                x1, y1, x2, y2 = bboxes[b, n].long()

                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                if x2 > x1 and y2 > y1:
                    roi_depth = depth[b, 0, y1:y2, x1:x2]
                    # Use median for robustness
                    object_depths[b, n] = roi_depth.median()

        return object_depths

    def depth_to_3d(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert depth map to 3D point cloud.

        Args:
            depth: Depth map (B, 1, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)

        Returns:
            Point cloud (B, 3, H, W)
        """
        B, _, H, W = depth.shape
        device = depth.device

        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        x = x.float().unsqueeze(0).expand(B, -1, -1)
        y = y.float().unsqueeze(0).expand(B, -1, -1)

        # Unproject
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)

        z = depth.squeeze(1)
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy

        return torch.stack([x_3d, y_3d, z], dim=1)


class DepthLoss(nn.Module):
    """
    Depth estimation loss functions.

    Combines:
    - Scale-invariant log loss
    - Gradient loss for edge preservation
    - Smoothness loss
    """

    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 0.85,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute depth losses.

        Args:
            pred: Predicted depth (B, 1, H, W)
            target: Ground truth depth (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W)

        Returns:
            Dictionary of loss components
        """
        if mask is None:
            mask = (target > 0).float()

        # Scale-invariant log loss
        log_diff = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
        log_diff = log_diff * mask

        n = mask.sum() + 1e-6
        si_loss = (log_diff ** 2).sum() / n - self.beta * (log_diff.sum() / n) ** 2

        # Gradient loss
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss = (
            F.l1_loss(pred_grad_x, target_grad_x, reduction='mean') +
            F.l1_loss(pred_grad_y, target_grad_y, reduction='mean')
        )

        # Smoothness loss (edge-aware)
        disp = 1 / (pred + 1e-6)
        smooth_loss = self._edge_aware_smoothness(disp, target)

        return {
            'si_loss': si_loss,
            'grad_loss': grad_loss,
            'smooth_loss': smooth_loss,
            'total': self.alpha * si_loss + grad_loss + 0.1 * smooth_loss,
        }

    def _edge_aware_smoothness(
        self,
        disp: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-aware smoothness loss."""
        disp_grad_x = disp[:, :, :, 1:] - disp[:, :, :, :-1]
        disp_grad_y = disp[:, :, 1:, :] - disp[:, :, :-1, :]

        image_grad_x = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        image_grad_y = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

        disp_grad_x = disp_grad_x.abs() * torch.exp(-image_grad_x)
        disp_grad_y = disp_grad_y.abs() * torch.exp(-image_grad_y)

        return disp_grad_x.mean() + disp_grad_y.mean()


if __name__ == '__main__':
    # Test depth estimator
    estimator = DepthEstimator(in_channels=256)

    features = torch.randn(2, 256, 60, 80)
    bboxes = torch.tensor([
        [[100, 100, 200, 200], [300, 150, 400, 250]],
        [[50, 50, 150, 150], [250, 100, 350, 200]],
    ]).float()

    outputs = estimator(features, bboxes)

    print("Depth Estimator outputs:")
    print(f"  depth: {outputs['depth'].shape}")
    print(f"  uncertainty: {outputs['uncertainty'].shape}")
    print(f"  object_depths: {outputs['object_depths'].shape}")
    print(f"  object_depths values: {outputs['object_depths']}")
