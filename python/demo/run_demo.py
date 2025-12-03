#!/usr/bin/env python3
"""
Tesla FSD Vision - Demo Script

Demonstrates the FSD Vision system with:
1. Model architecture summary
2. Sample inference (with random data)
3. Visualization output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

# Import our models
from models.backbone import RegNetBackbone, FPN
from models.hydranet import HydraNet
from models.bev_transformer import BEVTransformer
from models.occupancy_network import OccupancyNetwork
from models.fsd_vision import TeslaFSDVision, create_fsd_vision


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_backbone():
    """Demonstrate RegNet backbone"""
    print_header("1. RegNet Backbone")

    backbone = RegNetBackbone(config_name='tesla_custom')
    print(f"Backbone: RegNet (Tesla Custom)")
    print(f"Parameters: {count_parameters(backbone):,}")
    print(f"Output channels: {backbone.out_channels}")

    # Test forward pass
    x = torch.randn(1, 3, 960, 1280)  # Tesla camera resolution
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        features = backbone(x)

    print("Output feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")


def demo_hydranet():
    """Demonstrate HydraNet multi-task architecture"""
    print_header("2. HydraNet (Multi-Task Network)")

    hydranet = HydraNet(
        backbone_name='regnet_y_400mf',  # Smaller for demo
        fpn_channels=128,
        num_object_classes=10,
        num_semantic_classes=20
    )
    print(f"Parameters: {count_parameters(hydranet):,}")

    print("\nActive heads:")
    for head_name, active in hydranet.active_heads.items():
        status = "✓" if active else "✗"
        print(f"  [{status}] {head_name}")

    # Test forward pass
    x = torch.randn(1, 3, 480, 640)  # Smaller for speed
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        outputs = hydranet(x)

    print("\nOutput heads:")
    for head_name, output in outputs.items():
        if isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {head_name}/{k}: {v.shape}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            print(f"  {head_name}/{k}/{kk}: {vv.shape}")


def demo_bev_transformer():
    """Demonstrate BEV Transformer"""
    print_header("3. BEV Transformer (2D→3D)")

    bev_transformer = BEVTransformer(
        d_model=128,
        bev_h=100,
        bev_w=100,
        bev_resolution=0.5,
        n_cameras=8
    )
    print(f"Parameters: {count_parameters(bev_transformer):,}")
    print(f"BEV Grid: {bev_transformer.bev_h} x {bev_transformer.bev_w}")
    print(f"Resolution: {bev_transformer.bev_resolution}m per cell")
    print(f"Coverage: {bev_transformer.bev_h * bev_transformer.bev_resolution}m x {bev_transformer.bev_w * bev_transformer.bev_resolution}m")

    # Test forward pass
    multi_cam_features = [torch.randn(1, 128, 30, 40) for _ in range(8)]
    print(f"\nInput: 8 cameras × {multi_cam_features[0].shape}")

    with torch.no_grad():
        bev_features = bev_transformer(multi_cam_features)

    print(f"Output BEV features: {bev_features.shape}")


def demo_occupancy_network():
    """Demonstrate Occupancy Network"""
    print_header("4. Occupancy Network (3D Scene)")

    occ_net = OccupancyNetwork(
        bev_channels=128,
        voxel_channels=64,
        voxel_size=(0.5, 0.5, 0.5),
        num_classes=15
    )
    print(f"Parameters: {count_parameters(occ_net):,}")
    print(f"Voxel size: {occ_net.voxel_size}")
    print(f"Grid dimensions: {occ_net.grid_x} x {occ_net.grid_y} x {occ_net.grid_z}")

    # Test forward pass
    bev_features = torch.randn(1, 128, 100, 100)
    print(f"\nInput BEV features: {bev_features.shape}")

    with torch.no_grad():
        occ_outputs = occ_net(bev_features)

    print("\nOccupancy outputs:")
    for name, tensor in occ_outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {name}: {tensor.shape}")


def demo_full_pipeline():
    """Demonstrate complete FSD Vision pipeline"""
    print_header("5. Complete FSD Vision Pipeline")

    # Create model (small config for demo)
    model = create_fsd_vision(config='small')
    print(f"Total parameters: {count_parameters(model):,}")

    # Simulate 8-camera input
    batch_size = 1
    n_cameras = 8
    height, width = 480, 640

    images = torch.randn(batch_size, n_cameras, 3, height, width)
    print(f"\nInput: {n_cameras} cameras × {images.shape[2:]} = {batch_size * n_cameras * 3 * height * width:,} pixels")

    # Camera matrices (identity for demo)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(batch_size, n_cameras, -1, -1)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, n_cameras, -1, -1)

    print("\nRunning inference...")
    with torch.no_grad():
        outputs = model(images, intrinsics, extrinsics)

    print("\n✓ Pipeline output keys:")
    for key in outputs.keys():
        value = outputs[key]
        if isinstance(value, torch.Tensor):
            print(f"  • {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  • {key}: (dict)")
        elif isinstance(value, list):
            print(f"  • {key}: (list, len={len(value)})")


def demo_inference_results():
    """Show sample inference results"""
    print_header("6. Sample Inference Results")

    # Simulated results (what the model would output)
    print("\n📦 Object Detections:")
    objects = [
        {"class": "VEHICLE", "distance": "45m", "speed": "62 km/h", "confidence": 0.95},
        {"class": "VEHICLE", "distance": "72m", "speed": "58 km/h", "confidence": 0.88},
        {"class": "PEDESTRIAN", "distance": "28m", "speed": "5 km/h", "confidence": 0.92},
        {"class": "TRUCK", "distance": "35m", "speed": "55 km/h", "confidence": 0.87},
    ]
    for obj in objects:
        print(f"  • {obj['class']}: {obj['distance']} | {obj['speed']} | conf={obj['confidence']:.2f}")

    print("\n🚦 Traffic Lights:")
    traffic_lights = [
        {"state": "GREEN", "distance": "85m", "timer": "12s", "relevance": 0.98},
        {"state": "RED", "distance": "150m", "timer": "28s", "relevance": 0.45},
    ]
    for tl in traffic_lights:
        print(f"  • {tl['state']}: {tl['distance']} | {tl['timer']} remaining | relevance={tl['relevance']:.2f}")

    print("\n🛣️ Lane Detection:")
    print("  • Ego lane: Detected ✓")
    print("  • Left lane: Solid white line")
    print("  • Right lane: Dashed white line")
    print("  • Lane offset: 0.15m left of center")

    print("\n📊 System Status:")
    print("  • Neural Network:")
    print("    - Vision:     ████████░░ 85%")
    print("    - Planning:   ███████░░░ 72%")
    print("    - Control:    █████████░ 90%")
    print("    - Prediction: ██████░░░░ 68%")

    print("\n🚗 Ego Vehicle:")
    print("  • Speed: 65 km/h")
    print("  • Set Speed: 70 km/h")
    print("  • Lead Vehicle: 45m | TTC: 12.5s")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print(" TESLA FSD VISION - ARCHITECTURE DEMO")
    print(" HydraNet + BEV Transformer + Occupancy Network")
    print("=" * 60)

    print("\nThis demo shows the architecture of Tesla's FSD vision system")
    print("recreated in PyTorch. The system includes:")
    print("  1. RegNet backbone for feature extraction")
    print("  2. HydraNet for multi-task perception")
    print("  3. BEV Transformer for 3D projection")
    print("  4. Occupancy Network for 3D scene understanding")

    try:
        demo_backbone()
        demo_hydranet()
        demo_bev_transformer()
        demo_occupancy_network()
        demo_full_pipeline()
        demo_inference_results()

        print_header("✓ Demo Complete!")
        print("\nThe FSD Vision system successfully processed:")
        print("  • 8 camera inputs (simulating Tesla's camera array)")
        print("  • Multi-task 2D perception (objects, lanes, traffic lights)")
        print("  • 2D→3D BEV projection")
        print("  • 3D occupancy prediction")
        print("\nFor visualization, run: python demo/visualizer.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
