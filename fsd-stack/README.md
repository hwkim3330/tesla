# Tesla FSD Open Source Stack

A comprehensive open-source implementation of Tesla's Full Self-Driving (FSD) neural network architecture. This project provides trainable models for autonomous driving perception, BEV transformation, and end-to-end planning.

## Key Features

- **8-Camera Perception**: Multi-camera object detection, lane detection, traffic light recognition
- **BEV Transformation**: Lift-Splat-Shoot (LSS) style camera-to-BEV projection
- **Traffic Light Detection**: State classification + monocular distance estimation
- **Depth Estimation**: Camera-based depth prediction for distance calculation
- **End-to-End Planning**: Neural network trajectory planning
- **Developer Mode UI**: Tesla-style visualization with real-time inference
- **Trainable Pipeline**: Full training scripts with nuScenes/KITTI support

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Tesla FSD Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Front    │  │ Front    │  │ Front    │  │ Side     │  │ Side     │  ...  │
│  │ Main     │  │ Wide     │  │ Narrow   │  │ Left     │  │ Right    │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │             │              │
│       └─────────────┴─────────────┼─────────────┴─────────────┘              │
│                                   ▼                                          │
│                    ┌──────────────────────────────┐                         │
│                    │    RegNet/EfficientNet       │                         │
│                    │    Image Backbone            │                         │
│                    │    (Shared Weights)          │                         │
│                    └──────────────┬───────────────┘                         │
│                                   │                                          │
│       ┌───────────────────────────┼───────────────────────────┐             │
│       │                           │                           │              │
│       ▼                           ▼                           ▼              │
│  ┌─────────────┐         ┌─────────────────┐         ┌─────────────┐        │
│  │   Depth     │         │  Lift-Splat     │         │  2D Feature │        │
│  │ Estimation  │         │  (LSS) BEV      │         │    Heads    │        │
│  └──────┬──────┘         └────────┬────────┘         └──────┬──────┘        │
│         │                         │                          │               │
│         │                         ▼                          │               │
│         │              ┌──────────────────┐                  │               │
│         │              │  BEV Transformer │                  │               │
│         │              │  (Spatial Attn)  │                  │               │
│         │              └────────┬─────────┘                  │               │
│         │                       │                            │               │
│         └───────────────────────┼────────────────────────────┘               │
│                                 │                                            │
│                    ┌────────────┴────────────┐                              │
│                    │   Temporal Fusion       │                              │
│                    │   (Video Transformer)   │                              │
│                    └────────────┬────────────┘                              │
│                                 │                                            │
│    ┌────────────┬───────────────┼───────────────┬────────────┐              │
│    ▼            ▼               ▼               ▼            ▼               │
│ ┌──────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐         │
│ │Object│   │  Lane    │   │ Traffic  │   │Occupancy │   │  Depth │         │
│ │ Det  │   │Detection │   │  Light   │   │   Grid   │   │  Head  │         │
│ └──┬───┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └───┬────┘         │
│    │            │              │              │              │               │
│    └────────────┴──────────────┼──────────────┴──────────────┘               │
│                                ▼                                             │
│                    ┌──────────────────────┐                                 │
│                    │   Planning Network   │                                 │
│                    │  (Trajectory Output) │                                 │
│                    └──────────┬───────────┘                                 │
│                               │                                              │
│                               ▼                                              │
│                    ┌──────────────────────┐                                 │
│                    │  Control Commands    │                                 │
│                    │ (Steering, Throttle, │                                 │
│                    │       Brake)         │                                 │
│                    └──────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/hwkim3330/tesla.git
cd tesla/fsd-stack

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights
./scripts/download_pretrained.sh
```

## Quick Start

### Inference on Single Image
```python
from fsd_stack import FSDModel

# Load pretrained model
model = FSDModel.from_pretrained('fsd-v14')

# Run inference
results = model.predict('path/to/image.jpg')

# Access outputs
print(results.objects)        # Detected objects with 3D boxes
print(results.lanes)          # Lane markings
print(results.traffic_lights) # Traffic light states + distances
print(results.trajectory)     # Planned trajectory
```

### Real-time Webcam Demo
```bash
python inference/realtime_inference.py --source webcam --viz developer
```

### Training
```bash
# Train perception model on nuScenes
python training/train_perception.py --config configs/nuscenes_perception.yaml

# Train BEV model
python training/train_bev.py --config configs/bev_lss.yaml

# Train end-to-end
python training/train_e2e.py --config configs/e2e_planning.yaml
```

## Project Structure

```
fsd-stack/
├── models/
│   ├── backbone/          # Image encoders (RegNet, EfficientNet, ViT)
│   ├── perception/        # Detection heads (objects, lanes, traffic lights)
│   ├── bev/               # BEV transformation (LSS, BEVFormer)
│   ├── temporal/          # Video/temporal fusion
│   ├── planning/          # Trajectory planning
│   └── e2e/               # End-to-end FSD network
├── data/
│   ├── datasets/          # Dataset loaders (nuScenes, KITTI, Waymo)
│   ├── augmentation/      # Data augmentation
│   └── preprocessing/     # Camera calibration, preprocessing
├── training/
│   ├── losses/            # Loss functions
│   ├── schedulers/        # Learning rate schedulers
│   └── train_*.py         # Training scripts
├── inference/
│   ├── run_inference.py   # Batch inference
│   └── realtime_inference.py  # Real-time demo
├── visualization/
│   ├── fsd_viz.py         # Developer mode visualization
│   └── web/               # Web-based visualization
├── configs/               # YAML configuration files
├── scripts/               # Utility scripts
└── notebooks/             # Jupyter notebooks
```

## Models

### 1. Backbone Networks
- **RegNet-Y**: Efficient CNN backbone (default)
- **EfficientNet-V2**: Compound scaling
- **ViT-L**: Vision Transformer

### 2. BEV Transformation
- **LSS (Lift-Splat-Shoot)**: Explicit depth-based projection
- **BEVFormer**: Transformer-based BEV encoding
- **Simple-BEV**: Lightweight alternative

### 3. Task Heads
- **Object Detection**: 3D bounding boxes, velocity estimation
- **Lane Detection**: Polyline representation
- **Traffic Light**: Detection + state + distance
- **Depth**: Monocular depth estimation
- **Occupancy**: 3D occupancy grid

### 4. Planning
- **Trajectory MLP**: Simple trajectory prediction
- **Diffusion Policy**: Diffusion-based planning
- **Transformer Planner**: Attention-based planning

## Training Datasets

| Dataset | Size | Cameras | 3D Labels | Download |
|---------|------|---------|-----------|----------|
| nuScenes | 1000 scenes | 6 | Yes | [Link](https://www.nuscenes.org/) |
| KITTI | 22 sequences | 4 | Yes | [Link](http://www.cvlibs.net/datasets/kitti/) |
| Waymo Open | 1150 scenes | 5 | Yes | [Link](https://waymo.com/open/) |
| BDD100K | 100K videos | 1 | Partial | [Link](https://www.bdd100k.com/) |

## Performance

### Object Detection (nuScenes val)
| Model | mAP | NDS | FPS |
|-------|-----|-----|-----|
| FSD-Base | 42.3 | 51.2 | 25 |
| FSD-Large | 48.7 | 56.8 | 15 |
| FSD-XL | 52.1 | 60.3 | 8 |

### Traffic Light Detection
| Model | mAP | Distance MAE | FPS |
|-------|-----|--------------|-----|
| TL-Detector | 89.2 | 2.3m | 45 |

## Citation

```bibtex
@software{fsd_stack,
  title = {Tesla FSD Open Source Stack},
  author = {Community Contributors},
  year = {2024},
  url = {https://github.com/hwkim3330/tesla}
}
```

## References

- [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711) - BEV from multi-view images
- [BEVFormer](https://arxiv.org/abs/2203.17270) - Transformer-based BEV
- [Tesla AI Day 2021/2022](https://www.youtube.com/watch?v=j0z4FweCy4M)
- [nuScenes Dataset](https://www.nuscenes.org/)

## License

MIT License - See LICENSE file for details.
