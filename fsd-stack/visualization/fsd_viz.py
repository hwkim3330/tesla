"""
Tesla FSD Developer Mode Visualization

Replicates the Tesla developer visualization showing:
- Object detections with 3D bounding boxes
- Lane markings
- Traffic lights with states
- BEV occupancy grid
- Planned trajectory
- Neural network confidence

Can be used in real-time or for post-processing.
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import colorsys


@dataclass
class VizConfig:
    """Visualization configuration."""
    # Window sizes
    main_width: int = 1280
    main_height: int = 720
    bev_size: int = 400

    # Colors (BGR for OpenCV)
    color_ego: Tuple[int, int, int] = (255, 100, 100)
    color_car: Tuple[int, int, int] = (100, 200, 255)
    color_truck: Tuple[int, int, int] = (100, 150, 255)
    color_pedestrian: Tuple[int, int, int] = (100, 255, 200)
    color_cyclist: Tuple[int, int, int] = (200, 255, 100)
    color_lane_white: Tuple[int, int, int] = (255, 255, 255)
    color_lane_yellow: Tuple[int, int, int] = (0, 200, 255)
    color_trajectory: Tuple[int, int, int] = (50, 255, 50)
    color_traffic_red: Tuple[int, int, int] = (0, 0, 255)
    color_traffic_yellow: Tuple[int, int, int] = (0, 255, 255)
    color_traffic_green: Tuple[int, int, int] = (0, 255, 0)

    # BEV settings
    bev_range: float = 50.0  # meters
    bev_ego_y: float = 0.75  # Ego position in BEV (normalized)


class FSDVisualizer:
    """
    Tesla FSD Developer Mode Visualization.

    Creates a visualization similar to Tesla's developer mode,
    showing detections, BEV, and driving decisions.
    """

    CLASS_COLORS = {
        'car': (100, 200, 255),
        'truck': (100, 150, 255),
        'bus': (100, 100, 255),
        'motorcycle': (200, 100, 255),
        'bicycle': (200, 255, 100),
        'pedestrian': (100, 255, 200),
        'traffic_cone': (0, 150, 255),
        'barrier': (150, 150, 150),
    }

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()
        self.frame_count = 0

    def create_visualization(
        self,
        camera_image: np.ndarray,
        outputs: Dict[str, Any],
        show_bev: bool = True,
        show_info: bool = True,
    ) -> np.ndarray:
        """
        Create complete FSD visualization.

        Args:
            camera_image: Front camera image (H, W, 3)
            outputs: FSD network outputs
            show_bev: Show BEV view
            show_info: Show info panel

        Returns:
            Visualization image (H, W, 3)
        """
        cfg = self.config

        # Create main canvas
        canvas = np.zeros((cfg.main_height, cfg.main_width, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 20)  # Dark background

        # Main camera view
        cam_h = int(cfg.main_height * 0.7)
        cam_w = int(cfg.main_width * 0.65)
        cam_img = cv2.resize(camera_image, (cam_w, cam_h))

        # Draw detections on camera image
        cam_img = self._draw_2d_detections(cam_img, outputs)
        cam_img = self._draw_lanes(cam_img, outputs)
        cam_img = self._draw_traffic_lights(cam_img, outputs)

        # Place camera view
        canvas[20:20+cam_h, 20:20+cam_w] = cam_img

        # Draw border
        cv2.rectangle(canvas, (20, 20), (20+cam_w, 20+cam_h), (50, 50, 60), 2)

        # BEV view
        if show_bev:
            bev_x = 20 + cam_w + 20
            bev_y = 20
            bev_img = self._create_bev_view(outputs)
            canvas[bev_y:bev_y+cfg.bev_size, bev_x:bev_x+cfg.bev_size] = bev_img

        # Info panel
        if show_info:
            info_x = 20 + cam_w + 20
            info_y = 20 + cfg.bev_size + 20
            info_h = cfg.main_height - info_y - 20
            info_w = cfg.bev_size
            canvas = self._draw_info_panel(canvas, outputs, info_x, info_y, info_w, info_h)

        # Control visualization at bottom
        control_y = 20 + cam_h + 20
        canvas = self._draw_controls(canvas, outputs, 20, control_y, cam_w, cfg.main_height - control_y - 20)

        # Frame counter
        cv2.putText(canvas, f"Frame: {self.frame_count}", (cfg.main_width - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        self.frame_count += 1

        return canvas

    def _draw_2d_detections(
        self,
        image: np.ndarray,
        outputs: Dict[str, Any],
    ) -> np.ndarray:
        """Draw 2D bounding boxes on image."""
        img = image.copy()

        # Get detections from outputs
        objects = outputs.get('objects', {})
        if not objects:
            return img

        # Simulated detections for demo
        # In real use, decode from network outputs
        demo_boxes = [
            {'bbox': [400, 300, 600, 450], 'class': 'car', 'conf': 0.92, 'dist': 15.3},
            {'bbox': [100, 350, 200, 420], 'class': 'pedestrian', 'conf': 0.87, 'dist': 8.2},
            {'bbox': [650, 320, 750, 400], 'class': 'car', 'conf': 0.89, 'dist': 22.5},
        ]

        for det in demo_boxes:
            x1, y1, x2, y2 = det['bbox']
            color = self.CLASS_COLORS.get(det['class'], (200, 200, 200))

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det['class']} {det['conf']:.0%} {det['dist']:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return img

    def _draw_lanes(
        self,
        image: np.ndarray,
        outputs: Dict[str, Any],
    ) -> np.ndarray:
        """Draw lane markings."""
        img = image.copy()
        h, w = img.shape[:2]

        # Demo lanes
        demo_lanes = [
            [(w*0.3, h), (w*0.35, h*0.5), (w*0.4, h*0.3)],  # Left lane
            [(w*0.5, h), (w*0.5, h*0.5), (w*0.5, h*0.3)],   # Center
            [(w*0.7, h), (w*0.65, h*0.5), (w*0.6, h*0.3)],  # Right lane
        ]

        for i, lane in enumerate(demo_lanes):
            points = np.array(lane, dtype=np.int32)
            color = self.config.color_lane_white if i != 1 else self.config.color_lane_yellow
            cv2.polylines(img, [points], False, color, 3)

        return img

    def _draw_traffic_lights(
        self,
        image: np.ndarray,
        outputs: Dict[str, Any],
    ) -> np.ndarray:
        """Draw traffic light detections."""
        img = image.copy()

        # Demo traffic lights
        demo_tl = [
            {'bbox': [550, 50, 590, 120], 'state': 'green', 'dist': 45.2},
        ]

        state_colors = {
            'red': self.config.color_traffic_red,
            'yellow': self.config.color_traffic_yellow,
            'green': self.config.color_traffic_green,
        }

        for tl in demo_tl:
            x1, y1, x2, y2 = tl['bbox']
            color = state_colors.get(tl['state'], (150, 150, 150))

            # Draw box with glow effect
            cv2.rectangle(img, (x1-2, y1-2), (x2+2, y2+2), color, 2)

            # Draw distance
            label = f"{tl['dist']:.0f}m"
            cv2.putText(img, label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return img

    def _create_bev_view(self, outputs: Dict[str, Any]) -> np.ndarray:
        """Create Bird's Eye View visualization."""
        cfg = self.config
        size = cfg.bev_size
        bev = np.zeros((size, size, 3), dtype=np.uint8)
        bev[:] = (25, 25, 30)

        # Grid
        for i in range(0, size, size // 10):
            cv2.line(bev, (i, 0), (i, size), (40, 40, 45), 1)
            cv2.line(bev, (0, i), (size, i), (40, 40, 45), 1)

        # Coordinate conversion
        def world_to_bev(x, y):
            bev_x = int((x / cfg.bev_range + 1) * size / 2)
            bev_y = int(size * cfg.bev_ego_y - y / cfg.bev_range * size / 2)
            return bev_x, bev_y

        # Draw ego vehicle
        ego_x, ego_y = world_to_bev(0, 0)
        cv2.circle(bev, (ego_x, ego_y), 8, cfg.color_ego, -1)

        # Draw direction indicator
        cv2.arrowedLine(bev, (ego_x, ego_y), (ego_x, ego_y - 20),
                       cfg.color_ego, 2, tipLength=0.3)

        # Draw detected objects (demo)
        demo_objects = [
            {'x': 5, 'y': 20, 'class': 'car'},
            {'x': -3, 'y': 15, 'class': 'car'},
            {'x': 2, 'y': 8, 'class': 'pedestrian'},
            {'x': -8, 'y': 25, 'class': 'truck'},
        ]

        for obj in demo_objects:
            bx, by = world_to_bev(obj['x'], obj['y'])
            color = self.CLASS_COLORS.get(obj['class'], (200, 200, 200))
            if obj['class'] in ['car', 'truck', 'bus']:
                # Draw rectangle for vehicles
                cv2.rectangle(bev, (bx-6, by-10), (bx+6, by+10), color, -1)
            else:
                # Draw circle for pedestrians
                cv2.circle(bev, (bx, by), 4, color, -1)

        # Draw trajectory
        trajectory = outputs.get('trajectory', None)
        if trajectory is not None and isinstance(trajectory, torch.Tensor):
            traj = trajectory.cpu().numpy()
            points = []
            for i in range(len(traj)):
                bx, by = world_to_bev(traj[i, 0], traj[i, 1])
                points.append((bx, by))
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(bev, [points], False, cfg.color_trajectory, 2)
        else:
            # Demo trajectory
            demo_traj = [(0, i*3) for i in range(15)]
            points = [world_to_bev(x, y) for x, y in demo_traj]
            cv2.polylines(bev, [np.array(points)], False, cfg.color_trajectory, 2)

        # Range rings
        for r in [10, 20, 30, 40]:
            radius = int(r / cfg.bev_range * size / 2)
            cv2.circle(bev, (ego_x, ego_y), radius, (50, 50, 55), 1)
            cv2.putText(bev, f"{r}m", (ego_x + radius + 2, ego_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 85), 1)

        # Title
        cv2.putText(bev, "BEV", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return bev

    def _draw_info_panel(
        self,
        canvas: np.ndarray,
        outputs: Dict[str, Any],
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """Draw information panel."""
        # Panel background
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (30, 30, 35), -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (50, 50, 55), 1)

        # Title
        cv2.putText(canvas, "NEURAL NETWORK", (x+10, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 255), 1)

        # Stats
        stats = [
            ("Inference", f"{outputs.get('inference_time_ms', 25.3):.1f}ms"),
            ("Objects", "5 detected"),
            ("Lanes", "3 detected"),
            ("Traffic Lights", "1 green"),
            ("Confidence", "94.2%"),
        ]

        for i, (label, value) in enumerate(stats):
            ty = y + 45 + i * 22
            cv2.putText(canvas, label, (x+10, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 105), 1)
            cv2.putText(canvas, value, (x+w-80, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return canvas

    def _draw_controls(
        self,
        canvas: np.ndarray,
        outputs: Dict[str, Any],
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """Draw control visualization (steering, throttle, brake)."""
        # Panel background
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (30, 30, 35), -1)

        # Get control values
        steering = outputs.get('steering', 0.0)
        if isinstance(steering, torch.Tensor):
            steering = steering.item()
        throttle = outputs.get('throttle', 0.3)
        if isinstance(throttle, torch.Tensor):
            throttle = throttle.item()
        brake = outputs.get('brake', 0.0)
        if isinstance(brake, torch.Tensor):
            brake = brake.item()

        # Draw steering wheel
        wheel_cx = x + 80
        wheel_cy = y + h // 2
        wheel_r = 35

        cv2.circle(canvas, (wheel_cx, wheel_cy), wheel_r, (60, 60, 65), 2)
        cv2.circle(canvas, (wheel_cx, wheel_cy), wheel_r - 8, (50, 50, 55), 2)

        # Steering indicator
        angle = steering * 45  # Max 45 degrees
        end_x = int(wheel_cx + wheel_r * 0.7 * np.sin(np.radians(angle)))
        end_y = int(wheel_cy - wheel_r * 0.7 * np.cos(np.radians(angle)))
        cv2.line(canvas, (wheel_cx, wheel_cy), (end_x, end_y), (100, 200, 255), 3)

        cv2.putText(canvas, f"Steering: {steering:.2f}", (x + 10, y + h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw throttle bar
        bar_x = x + 180
        bar_w = 150
        bar_h = 20
        bar_y = y + h // 2 - 15

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 55), 1)
        fill_w = int(bar_w * throttle)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (50, 200, 50), -1)
        cv2.putText(canvas, f"Throttle: {throttle:.0%}", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Draw brake bar
        brake_x = bar_x + bar_w + 30
        cv2.rectangle(canvas, (brake_x, bar_y), (brake_x + bar_w, bar_y + bar_h), (50, 50, 55), 1)
        fill_w = int(bar_w * brake)
        cv2.rectangle(canvas, (brake_x, bar_y), (brake_x + fill_w, bar_y + bar_h), (50, 50, 200), -1)
        cv2.putText(canvas, f"Brake: {brake:.0%}", (brake_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Speed display
        speed = 65 + np.random.randn() * 2  # Demo speed
        speed_x = x + w - 120
        cv2.putText(canvas, f"{speed:.0f}", (speed_x, y + h // 2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(canvas, "km/h", (speed_x + 70, y + h // 2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return canvas


def demo_visualization():
    """Run a demo of the FSD visualization."""
    viz = FSDVisualizer()

    # Create demo camera image
    camera_img = np.zeros((480, 640, 3), dtype=np.uint8)
    camera_img[:] = (50, 40, 30)

    # Add road
    road_pts = np.array([
        [0, 480], [640, 480],
        [500, 250], [140, 250]
    ], dtype=np.int32)
    cv2.fillPoly(camera_img, [road_pts], (60, 60, 60))

    # Demo outputs
    demo_outputs = {
        'steering': 0.1,
        'throttle': 0.4,
        'brake': 0.0,
        'inference_time_ms': 23.5,
    }

    # Create visualization
    viz_img = viz.create_visualization(camera_img, demo_outputs)

    # Display
    cv2.imshow('Tesla FSD Developer Mode', viz_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo_visualization()
