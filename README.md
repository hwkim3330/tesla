# Tesla Autopilot - FSD Visualization

Interactive visualization of Tesla's Full Self-Driving (FSD) system with real-time neural network simulation, Bird's Eye View (BEV), and 3D road rendering.

**Live Demo**: https://hwkim3330.github.io/tesla/

![Tesla FSD Visualization](https://img.shields.io/badge/Tesla-FSD-red?style=for-the-badge&logo=tesla)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-black?style=for-the-badge&logo=three.js&logoColor=white)

## Features

### Web Visualization (JavaScript)
- **Real-time Steering Prediction**: NVIDIA-style CNN model using TensorFlow.js
- **3D Road Visualization**: Three.js based driving simulation
- **Bird's Eye View (BEV)**: Top-down view with detected objects and planned path
- **Neural Network Activity**: Live visualization of network layer activations
- **Detection Display**: Real-time object detection results (vehicles, lanes, signs)
- **Webcam Support**: Option to use real camera input for inference

### Python Models
Complete PyTorch implementation of Tesla FSD architecture:
- **HydraNet**: Multi-task learning with shared backbone
- **BEV Transformer**: 2D to 3D Bird's Eye View projection
- **Occupancy Network**: 3D voxel-based scene understanding
- **RegNet Backbone**: Feature extraction with Feature Pyramid Network

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    8 Camera Images                           │
│   (Front Main/Narrow/Wide, B-Pillars, Sides, Rear)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    RegNet Backbone                           │
│              (Feature Extraction + FPN)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  BEV Transformer                             │
│         (2D → 3D Projection, Temporal Fusion)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    HydraNet Heads                            │
│    (Detection, Lanes, Depth, Segmentation, Path Planning)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Control Output                            │
│            (Steering, Acceleration, Brake)                   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
tesla/
├── index.html              # Main web application
├── js/
│   ├── autopilot.js        # Neural network steering prediction (TF.js)
│   ├── neural-viz.js       # Network activity visualization
│   ├── bev-renderer.js     # Bird's Eye View renderer
│   └── main-view.js        # 3D road scene (Three.js)
├── python/
│   ├── models/
│   │   ├── backbone.py         # RegNet backbone + FPN
│   │   ├── bev_transformer.py  # BEV projection
│   │   ├── detection_heads.py  # Multi-task heads
│   │   ├── hydranet.py         # HydraNet architecture
│   │   ├── occupancy_network.py# 3D occupancy prediction
│   │   └── fsd_vision.py       # Complete FSD system
│   ├── utils/
│   │   ├── camera.py           # Camera models
│   │   └── transforms.py       # Image transforms
│   ├── demo/
│   │   └── run_demo.py         # Demo script
│   └── requirements.txt
└── README.md
```

## Quick Start

### Web Demo
Simply visit: https://hwkim3330.github.io/tesla/

Or run locally:
```bash
git clone https://github.com/hwkim3330/tesla.git
cd tesla
python -m http.server 8000
# Open http://localhost:8000
```

### Python Models
```bash
cd python
pip install -r requirements.txt
python demo/run_demo.py
```

## Technologies

### Web
- **TensorFlow.js**: Browser-based neural network inference
- **Three.js**: 3D graphics and road rendering
- **Canvas API**: 2D visualizations (BEV, network activity)
- **WebRTC**: Webcam access for real camera input

### Python
- **PyTorch**: Deep learning framework
- **timm**: Pre-trained vision models
- **NumPy**: Numerical computing

## Inspired By

- [NVIDIA End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [akshaybahadur21/Autopilot](https://github.com/akshaybahadur21/Autopilot)
- [Tesla AI Day Presentations](https://www.youtube.com/tesla)
- [Andrej Karpathy's Neural Networks Lectures](https://karpathy.ai/)

## Key Concepts

### End-to-End Learning
The neural network learns to map raw camera images directly to steering commands, bypassing traditional computer vision pipelines.

### Behavior Cloning
Training on millions of miles of human driving data to learn "what would a good driver do in this situation?"

### Multi-Task Learning (HydraNet)
Single backbone network with multiple task-specific heads for:
- Object Detection
- Lane Detection
- Depth Estimation
- Semantic Segmentation
- Path Prediction

### Bird's Eye View (BEV)
Transforming 2D camera images into a unified 3D representation for better spatial reasoning and planning.

## Screenshots

| Main View | BEV | Neural Network |
|-----------|-----|----------------|
| 3D road with detections | Top-down object view | Layer activations |

## Disclaimer

This is an educational visualization project. It is not affiliated with Tesla, Inc. and does not represent the actual FSD system implementation.

## License

MIT License
