"""
Camera utilities for Tesla FSD Vision

Handles:
- Camera intrinsic/extrinsic matrices
- Projection between image and 3D coordinates
- Multi-camera setup configuration
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class CameraModel:
    """
    Camera model with intrinsic and extrinsic parameters

    Intrinsics: focal length, principal point, distortion
    Extrinsics: position and orientation relative to ego vehicle
    """

    # Camera name
    name: str

    # Image dimensions
    width: int
    height: int

    # Intrinsic parameters
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x
    cy: float  # Principal point y

    # Distortion coefficients (optional)
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    # Extrinsic parameters (relative to ego vehicle center)
    x: float = 0.0  # Position x (meters, forward)
    y: float = 0.0  # Position y (meters, left)
    z: float = 0.0  # Position z (meters, up)
    roll: float = 0.0  # Roll angle (radians)
    pitch: float = 0.0  # Pitch angle (radians)
    yaw: float = 0.0  # Yaw angle (radians)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

    @property
    def intrinsic_tensor(self) -> torch.Tensor:
        """Get intrinsic matrix as torch tensor"""
        return torch.from_numpy(self.intrinsic_matrix)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from ego to camera frame"""
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.roll), -np.sin(self.roll)],
            [0, np.sin(self.roll), np.cos(self.roll)]
        ])
        Ry = np.array([
            [np.cos(self.pitch), 0, np.sin(self.pitch)],
            [0, 1, 0],
            [-np.sin(self.pitch), 0, np.cos(self.pitch)]
        ])
        Rz = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])

        return (Rz @ Ry @ Rx).astype(np.float32)

    @property
    def translation_vector(self) -> np.ndarray:
        """Get 3x1 translation vector"""
        return np.array([[self.x], [self.y], [self.z]], dtype=np.float32)

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """Get 4x4 extrinsic matrix (world to camera)"""
        R = self.rotation_matrix
        t = self.translation_vector

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R.T  # Inverse rotation
        extrinsic[:3, 3:4] = -R.T @ t  # Inverse translation

        return extrinsic

    @property
    def extrinsic_tensor(self) -> torch.Tensor:
        """Get extrinsic matrix as torch tensor"""
        return torch.from_numpy(self.extrinsic_matrix)

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates

        Args:
            points_3d: [N, 3] 3D points in world coordinates

        Returns:
            [N, 2] 2D image coordinates
        """
        # Transform to camera coordinates
        points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        points_cam = (self.extrinsic_matrix @ points_homo.T).T[:, :3]

        # Project to image
        points_2d_homo = (self.intrinsic_matrix @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]

        return points_2d

    def unproject_2d_to_3d(
        self,
        points_2d: np.ndarray,
        depths: np.ndarray
    ) -> np.ndarray:
        """
        Unproject 2D points with depths to 3D

        Args:
            points_2d: [N, 2] 2D image coordinates
            depths: [N] depth values

        Returns:
            [N, 3] 3D points in world coordinates
        """
        # Normalized camera coordinates
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy

        # 3D points in camera frame
        points_cam = np.stack([
            x * depths,
            y * depths,
            depths
        ], axis=1)

        # Transform to world coordinates
        R = self.rotation_matrix
        t = self.translation_vector.flatten()
        points_world = (R @ points_cam.T).T + t

        return points_world

    def is_in_fov(self, points_2d: np.ndarray) -> np.ndarray:
        """Check if 2D points are within field of view"""
        in_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.width)
        in_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.height)
        return in_x & in_y


def create_tesla_cameras() -> List[CameraModel]:
    """
    Create Tesla's 8-camera configuration

    Camera positions based on public information about Tesla's camera setup:
    - 3 forward-facing cameras (main, narrow, wide)
    - 2 front-side cameras (B-pillar)
    - 2 side repeater cameras
    - 1 rear camera

    All positions are approximate and in meters.
    """
    cameras = []

    # Common parameters
    width, height = 1280, 960

    # Forward Main Camera
    # Primary camera for most driving scenarios
    cameras.append(CameraModel(
        name='front_main',
        width=width, height=height,
        fx=1000, fy=1000, cx=640, cy=480,
        x=2.0, y=0.0, z=1.5,
        roll=0, pitch=0, yaw=0
    ))

    # Forward Narrow Camera
    # Long-range detection (highway, traffic lights)
    cameras.append(CameraModel(
        name='front_narrow',
        width=width, height=height,
        fx=2500, fy=2500, cx=640, cy=480,  # Longer focal length
        x=2.0, y=0.0, z=1.5,
        roll=0, pitch=0, yaw=0
    ))

    # Forward Wide Camera
    # Close-range, intersections, cutting vehicles
    cameras.append(CameraModel(
        name='front_wide',
        width=width, height=height,
        fx=500, fy=500, cx=640, cy=480,  # Shorter focal length
        x=2.0, y=0.0, z=1.5,
        roll=0, pitch=0, yaw=0
    ))

    # Front Left B-Pillar Camera
    cameras.append(CameraModel(
        name='front_left_pillar',
        width=width, height=height,
        fx=800, fy=800, cx=640, cy=480,
        x=1.5, y=0.9, z=1.3,
        roll=0, pitch=0, yaw=math.radians(-60)
    ))

    # Front Right B-Pillar Camera
    cameras.append(CameraModel(
        name='front_right_pillar',
        width=width, height=height,
        fx=800, fy=800, cx=640, cy=480,
        x=1.5, y=-0.9, z=1.3,
        roll=0, pitch=0, yaw=math.radians(60)
    ))

    # Left Side Repeater Camera (in side mirror)
    cameras.append(CameraModel(
        name='side_left',
        width=width, height=height,
        fx=600, fy=600, cx=640, cy=480,
        x=0.5, y=1.0, z=1.0,
        roll=0, pitch=0, yaw=math.radians(-90)
    ))

    # Right Side Repeater Camera (in side mirror)
    cameras.append(CameraModel(
        name='side_right',
        width=width, height=height,
        fx=600, fy=600, cx=640, cy=480,
        x=0.5, y=-1.0, z=1.0,
        roll=0, pitch=0, yaw=math.radians(90)
    ))

    # Rear Camera
    cameras.append(CameraModel(
        name='rear',
        width=width, height=height,
        fx=700, fy=700, cx=640, cy=480,
        x=-0.5, y=0.0, z=1.5,
        roll=0, pitch=0, yaw=math.radians(180)
    ))

    return cameras


def get_camera_coverage_visualization() -> dict:
    """
    Get camera coverage angles for visualization

    Returns a dictionary with FOV information for each camera.
    """
    cameras = create_tesla_cameras()

    coverage = {}
    for cam in cameras:
        # Approximate horizontal FOV
        hfov = 2 * math.atan(cam.width / (2 * cam.fx))
        hfov_deg = math.degrees(hfov)

        coverage[cam.name] = {
            'position': (cam.x, cam.y),
            'yaw': math.degrees(cam.yaw),
            'hfov': hfov_deg
        }

    return coverage


def create_camera_tensors(
    cameras: List[CameraModel],
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create batched intrinsic and extrinsic tensors

    Args:
        cameras: List of camera models
        device: Target device

    Returns:
        intrinsics: [N_cams, 3, 3] intrinsic matrices
        extrinsics: [N_cams, 4, 4] extrinsic matrices
    """
    intrinsics = torch.stack([cam.intrinsic_tensor for cam in cameras])
    extrinsics = torch.stack([cam.extrinsic_tensor for cam in cameras])

    return intrinsics.to(device), extrinsics.to(device)
