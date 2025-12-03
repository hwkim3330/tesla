"""
Tesla FSD Visualization Demo

Visualizes the output of the FSD Vision system in Tesla's style:
- Camera view with detection overlays
- BEV (Bird's Eye View) representation
- Traffic light status
- Neural network activity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
import torch
from typing import Dict, List, Tuple, Optional
import math


class TeslaFSDVisualizer:
    """
    Visualizer for Tesla FSD-style output display

    Creates a split-screen visualization with:
    - Left: Camera view with detections
    - Right: BEV top-down view
    """

    # Color scheme (Tesla style)
    COLORS = {
        'background': '#0a0a0f',
        'ego_vehicle': '#17b06b',
        'other_vehicle': '#3e6ae1',
        'pedestrian': '#f5a623',
        'cyclist': '#ff6b35',
        'lane_line': '#3e6ae1',
        'predicted_path': '#17b06b',
        'traffic_red': '#ff0000',
        'traffic_yellow': '#ffff00',
        'traffic_green': '#00ff00',
        'grid': '#1a3a5c',
        'text': '#ffffff'
    }

    # Object class names
    CLASS_NAMES = [
        'vehicle', 'pedestrian', 'cyclist', 'motorcycle',
        'truck', 'bus', 'construction', 'traffic_cone',
        'barrier', 'unknown'
    ]

    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 8),
        bev_range: float = 50.0,
        camera_aspect: float = 16 / 9
    ):
        self.figsize = figsize
        self.bev_range = bev_range
        self.camera_aspect = camera_aspect

        # Setup figure
        self.fig, (self.ax_camera, self.ax_bev) = plt.subplots(
            1, 2, figsize=figsize,
            facecolor=self.COLORS['background']
        )

        self._setup_camera_view()
        self._setup_bev_view()

    def _setup_camera_view(self):
        """Setup the camera view panel"""
        self.ax_camera.set_facecolor(self.COLORS['background'])
        self.ax_camera.set_xlim(0, 1280)
        self.ax_camera.set_ylim(960, 0)  # Flip Y for image coordinates
        self.ax_camera.set_aspect('equal')
        self.ax_camera.axis('off')

        # Title
        self.ax_camera.set_title(
            'FULL SELF-DRIVING (Supervised)',
            color=self.COLORS['text'],
            fontsize=14,
            fontweight='bold'
        )

    def _setup_bev_view(self):
        """Setup the BEV panel"""
        self.ax_bev.set_facecolor(self.COLORS['background'])
        self.ax_bev.set_xlim(-self.bev_range, self.bev_range)
        self.ax_bev.set_ylim(-10, self.bev_range * 2 - 10)
        self.ax_bev.set_aspect('equal')
        self.ax_bev.axis('off')

        # Title
        self.ax_bev.set_title(
            "BIRD'S EYE VIEW",
            color=self.COLORS['text'],
            fontsize=14,
            fontweight='bold'
        )

        # Draw grid
        self._draw_grid()

        # Draw ego vehicle
        self._draw_ego_vehicle()

    def _draw_grid(self):
        """Draw BEV grid"""
        for x in np.arange(-self.bev_range, self.bev_range + 1, 10):
            self.ax_bev.axvline(x, color=self.COLORS['grid'], alpha=0.2, linewidth=0.5)
        for y in np.arange(-10, self.bev_range * 2, 10):
            self.ax_bev.axhline(y, color=self.COLORS['grid'], alpha=0.2, linewidth=0.5)

        # Distance markers
        for d in [20, 40, 60, 80, 100]:
            self.ax_bev.text(
                self.bev_range - 2, d,
                f'{d}m',
                color='#666666',
                fontsize=8,
                ha='right'
            )

    def _draw_ego_vehicle(self):
        """Draw ego vehicle in BEV"""
        # Car body
        ego_rect = patches.FancyBboxPatch(
            (-1.5, -2.5), 3, 5,
            boxstyle="round,pad=0.1",
            facecolor=self.COLORS['ego_vehicle'],
            edgecolor='none',
            alpha=0.9
        )
        self.ax_bev.add_patch(ego_rect)

        # Direction arrow
        arrow = patches.FancyArrow(
            0, 2.5, 0, 3,
            width=0.5, head_width=1.5, head_length=1,
            fc=self.COLORS['ego_vehicle'],
            ec='none',
            alpha=0.8
        )
        self.ax_bev.add_patch(arrow)

    def visualize_frame(
        self,
        camera_image: Optional[np.ndarray] = None,
        detections: List[Dict] = None,
        traffic_lights: List[Dict] = None,
        lanes: Dict = None,
        depth_map: Optional[np.ndarray] = None,
        occupancy: Dict = None,
        predicted_path: Optional[np.ndarray] = None,
        speed: float = 65.0,
        set_speed: float = 70.0
    ):
        """
        Visualize a single frame

        Args:
            camera_image: [H, W, 3] camera image
            detections: List of object detections
            traffic_lights: List of traffic light detections
            lanes: Lane detection results
            depth_map: [H, W] depth map
            occupancy: Occupancy grid results
            predicted_path: [N, 2] path points
            speed: Current speed
            set_speed: Set/target speed
        """
        # Clear previous drawings (keep static elements)
        for artist in self.ax_camera.patches[:]:
            if not isinstance(artist, patches.Rectangle):
                artist.remove()
        for artist in self.ax_bev.patches[2:]:  # Keep ego vehicle
            artist.remove()

        # Camera view
        if camera_image is not None:
            self.ax_camera.imshow(camera_image)
        else:
            # Draw simulated road
            self._draw_simulated_road()

        # Draw detections
        if detections:
            self._draw_detections_camera(detections)
            self._draw_detections_bev(detections)

        # Draw traffic lights
        if traffic_lights:
            self._draw_traffic_lights(traffic_lights)

        # Draw lanes
        if lanes:
            self._draw_lanes(lanes)

        # Draw predicted path
        if predicted_path is not None:
            self._draw_predicted_path(predicted_path)
        else:
            self._draw_simulated_path()

        # Draw occupancy
        if occupancy:
            self._draw_occupancy_bev(occupancy)

        # Draw HUD
        self._draw_hud(speed, set_speed, traffic_lights)

        plt.tight_layout()

    def _draw_simulated_road(self):
        """Draw simulated road perspective"""
        # Road surface
        road_points = np.array([
            [400, 960],
            [880, 960],
            [720, 300],
            [560, 300]
        ])
        road_poly = patches.Polygon(
            road_points,
            facecolor='#1a1a2e',
            edgecolor='none',
            alpha=0.5
        )
        self.ax_camera.add_patch(road_poly)

        # Lane lines
        for x_offset in [-160, -80, 80, 160]:
            x_top = 640 + x_offset * 0.3
            x_bottom = 640 + x_offset
            self.ax_camera.plot(
                [x_bottom, x_top],
                [960, 300],
                color=self.COLORS['lane_line'],
                linewidth=2,
                alpha=0.7
            )

    def _draw_simulated_path(self):
        """Draw simulated predicted path"""
        # Camera view path
        path_points = np.array([
            [590, 960],
            [610, 960],
            [625, 500],
            [655, 500],
            [680, 250],
            [600, 250]
        ])
        path_poly = patches.Polygon(
            path_points,
            facecolor=self.COLORS['predicted_path'],
            edgecolor='none',
            alpha=0.3
        )
        self.ax_camera.add_patch(path_poly)

        # BEV path
        bev_path = np.array([
            [-2, 5],
            [2, 5],
            [1.5, 80],
            [-1.5, 80]
        ])
        bev_path_poly = patches.Polygon(
            bev_path,
            facecolor=self.COLORS['predicted_path'],
            edgecolor='none',
            alpha=0.2
        )
        self.ax_bev.add_patch(bev_path_poly)

    def _draw_detections_camera(self, detections: List[Dict]):
        """Draw detection boxes on camera view"""
        for det in detections:
            bbox = det.get('bbox', [0, 0, 100, 100])
            cls = det.get('class', 0)
            conf = det.get('confidence', 0.9)
            distance = det.get('distance', 45)

            color = self._get_class_color(cls)

            # Bounding box
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.1
            )
            self.ax_camera.add_patch(rect)

            # Label
            label = f"{self.CLASS_NAMES[cls].upper()}"
            self.ax_camera.text(
                bbox[0], bbox[1] - 5,
                label,
                color='white',
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
            )

            # Distance
            self.ax_camera.text(
                (bbox[0] + bbox[2]) / 2, bbox[3] + 5,
                f"{distance}m",
                color='white',
                fontsize=10,
                fontweight='bold',
                ha='center'
            )

    def _draw_detections_bev(self, detections: List[Dict]):
        """Draw detections on BEV"""
        for det in detections:
            pos_3d = det.get('position_3d', [0, 0, 45])
            cls = det.get('class', 0)
            dims = det.get('dimensions', [1.8, 4.5])  # width, length

            color = self._get_class_color(cls)

            # Draw as rectangle in BEV
            x, y = pos_3d[0], pos_3d[2]  # x, distance
            w, l = dims[0], dims[1]

            rect = patches.FancyBboxPatch(
                (x - w / 2, y - l / 2), w, l,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='white',
                linewidth=1,
                alpha=0.7
            )
            self.ax_bev.add_patch(rect)

    def _draw_traffic_lights(self, traffic_lights: List[Dict]):
        """Draw traffic light indicators"""
        for i, tl in enumerate(traffic_lights):
            state = tl.get('state', 'green')
            distance = tl.get('distance', 85)
            timer = tl.get('timer', 12)
            position = tl.get('position', [640, 200])

            # Draw on camera
            x, y = position

            # Traffic light box
            box_width, box_height = 25, 70
            box = patches.FancyBboxPatch(
                (x - box_width / 2, y),
                box_width, box_height,
                boxstyle="round,pad=0.05",
                facecolor='#222222',
                edgecolor='#444444',
                linewidth=2
            )
            self.ax_camera.add_patch(box)

            # Lights
            for j, (light_state, color) in enumerate([
                ('red', self.COLORS['traffic_red']),
                ('yellow', self.COLORS['traffic_yellow']),
                ('green', self.COLORS['traffic_green'])
            ]):
                light_y = y + 10 + j * 20
                alpha = 1.0 if state == light_state else 0.2

                circle = patches.Circle(
                    (x, light_y), 8,
                    facecolor=color,
                    edgecolor='none',
                    alpha=alpha
                )
                self.ax_camera.add_patch(circle)

                # Glow effect for active light
                if state == light_state:
                    glow = patches.Circle(
                        (x, light_y), 12,
                        facecolor=color,
                        edgecolor='none',
                        alpha=0.3
                    )
                    self.ax_camera.add_patch(glow)

            # Info label
            self.ax_camera.text(
                x, y + box_height + 10,
                f"{distance}m | {timer}s",
                color='white',
                fontsize=9,
                ha='center',
                bbox=dict(boxstyle='round', facecolor='#000000', alpha=0.7)
            )

    def _draw_lanes(self, lanes: Dict):
        """Draw lane lines"""
        # Simplified - would use actual lane data
        pass

    def _draw_predicted_path(self, path: np.ndarray):
        """Draw predicted ego path"""
        # BEV
        self.ax_bev.plot(
            path[:, 0], path[:, 1],
            color=self.COLORS['predicted_path'],
            linewidth=3,
            alpha=0.8
        )

        # Fill area
        path_left = path.copy()
        path_left[:, 0] -= 1.5
        path_right = path.copy()
        path_right[:, 0] += 1.5

        combined = np.vstack([path_left, path_right[::-1]])
        poly = patches.Polygon(
            combined,
            facecolor=self.COLORS['predicted_path'],
            edgecolor='none',
            alpha=0.2
        )
        self.ax_bev.add_patch(poly)

    def _draw_occupancy_bev(self, occupancy: Dict):
        """Draw occupancy grid on BEV"""
        # Simplified visualization
        pass

    def _draw_hud(
        self,
        speed: float,
        set_speed: float,
        traffic_lights: List[Dict]
    ):
        """Draw HUD elements"""
        # Speed display (camera view)
        self.ax_camera.text(
            1200, 50,
            f"{int(speed)}",
            color='white',
            fontsize=48,
            fontweight='300',
            ha='center'
        )
        self.ax_camera.text(
            1200, 90,
            "km/h",
            color='#888888',
            fontsize=12,
            ha='center'
        )
        self.ax_camera.text(
            1200, 110,
            f"Set: {int(set_speed)} km/h",
            color=self.COLORS['other_vehicle'],
            fontsize=10,
            ha='center'
        )

        # Detection counts
        self.ax_camera.text(
            50, 900,
            "Vehicles: 4\nPedestrians: 2\nTraffic Lights: 1",
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#000000', alpha=0.7)
        )

    def _get_class_color(self, cls: int) -> str:
        """Get color for object class"""
        color_map = {
            0: self.COLORS['other_vehicle'],  # vehicle
            1: self.COLORS['pedestrian'],  # pedestrian
            2: self.COLORS['cyclist'],  # cyclist
            3: self.COLORS['cyclist'],  # motorcycle
            4: self.COLORS['other_vehicle'],  # truck
            5: self.COLORS['other_vehicle'],  # bus
        }
        return color_map.get(cls, self.COLORS['other_vehicle'])

    def show(self):
        """Display the visualization"""
        plt.show()

    def save(self, filename: str):
        """Save visualization to file"""
        self.fig.savefig(
            filename,
            facecolor=self.COLORS['background'],
            dpi=150,
            bbox_inches='tight'
        )


def create_demo_visualization():
    """Create a demo visualization with simulated data"""
    viz = TeslaFSDVisualizer()

    # Simulated detections
    detections = [
        {
            'bbox': [500, 400, 700, 550],
            'class': 0,
            'confidence': 0.95,
            'distance': 45,
            'position_3d': [0, 0, 45],
            'dimensions': [1.8, 4.5]
        },
        {
            'bbox': [800, 350, 900, 500],
            'class': 0,
            'confidence': 0.88,
            'distance': 72,
            'position_3d': [8, 0, 72],
            'dimensions': [2.0, 5.0]
        },
        {
            'bbox': [300, 450, 380, 580],
            'class': 0,
            'confidence': 0.92,
            'distance': 35,
            'position_3d': [-6, 0, 35],
            'dimensions': [2.4, 6.0]
        },
        {
            'bbox': [1000, 500, 1050, 600],
            'class': 1,
            'confidence': 0.87,
            'distance': 28,
            'position_3d': [15, 0, 28],
            'dimensions': [0.5, 0.5]
        }
    ]

    # Simulated traffic lights
    traffic_lights = [
        {
            'state': 'green',
            'distance': 85,
            'timer': 12,
            'position': [640, 150]
        }
    ]

    # Visualize
    viz.visualize_frame(
        detections=detections,
        traffic_lights=traffic_lights,
        speed=65,
        set_speed=70
    )

    return viz


if __name__ == '__main__':
    print("Tesla FSD Visualization Demo")
    print("=" * 50)

    viz = create_demo_visualization()
    viz.show()
