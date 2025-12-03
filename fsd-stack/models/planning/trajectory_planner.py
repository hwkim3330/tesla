"""Trajectory planning networks."""

import torch
import torch.nn as nn


class TrajectoryPlanner(nn.Module):
    """MLP-based trajectory planner."""

    def __init__(self, in_channels: int = 256, num_waypoints: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(in_channels * 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * 3),
        )
        self.num_waypoints = num_waypoints

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.encoder(bev).view(-1, self.num_waypoints, 3)


class DiffusionPolicy(nn.Module):
    """Diffusion-based policy for trajectory generation."""

    def __init__(self, channels: int = 256, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.net = nn.Sequential(
            nn.Linear(channels + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, bev: torch.Tensor, noisy_traj: torch.Tensor = None) -> torch.Tensor:
        B = bev.shape[0]
        if noisy_traj is None:
            noisy_traj = torch.randn(B, self.num_steps, 3, device=bev.device)
        feat = bev.mean(dim=[2, 3]).unsqueeze(1).expand(-1, self.num_steps, -1)
        return self.net(torch.cat([feat, noisy_traj], dim=-1))
