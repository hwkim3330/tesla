# Tesla FSD Vision - PyTorch Implementation

테슬라 FSD(Full Self-Driving) 비전 시스템의 PyTorch 구현입니다.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        8 Camera Images                            │
│   (front_main, front_narrow, front_wide, front_left, front_right, │
│    side_left, side_right, rear)                                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    RegNet Backbone                                │
│              (Shared feature extraction)                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    Feature Pyramid Network                        │
│              (Multi-scale feature fusion)                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
┌──────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│    HydraNet     │ │    BEV      │ │  Temporal   │
│  (Multi-Task)   │ │ Transformer │ │   Fusion    │
│                 │ │  (2D→3D)    │ │             │
│  • Detection    │ └──────┬──────┘ └──────┬──────┘
│  • Traffic Light│        │               │
│  • Lane         │ ┌──────▼───────────────▼──────┐
│  • Depth        │ │      Occupancy Network      │
│  • Segmentation │ │    (3D Scene Understanding) │
│  • Path         │ └─────────────────────────────┘
└─────────────────┘
```

## Key Components

### 1. RegNet Backbone (`models/backbone.py`)
- Tesla에서 사용하는 주요 백본 네트워크
- SE (Squeeze-and-Excitation) 블록 포함
- 다중 스케일 특징 맵 출력 (1/4, 1/8, 1/16, 1/32)

### 2. HydraNet (`models/hydranet.py`)
- 여러 개의 태스크별 헤드를 가진 멀티태스크 네트워크
- 공유 백본 + 태스크별 전용 헤드

**Detection Heads** (`models/detection_heads.py`):
- `ObjectDetectionHead`: 차량, 보행자, 자전거 등 객체 감지
- `TrafficLightHead`: 신호등 감지 + 상태 분류 + 거리 추정
- `LaneDetectionHead`: 차선 세그멘테이션 + 인스턴스 임베딩
- `DepthEstimationHead`: 단안 깊이 추정
- `SemanticSegmentationHead`: 시맨틱 세그멘테이션
- `PathPredictionHead`: 자차 경로 예측

### 3. BEV Transformer (`models/bev_transformer.py`)
- 2D 이미지 특징을 3D BEV(Bird's Eye View)로 변환
- Deformable Attention 기반
- 카메라 내부/외부 파라미터 활용

### 4. Occupancy Network (`models/occupancy_network.py`)
- 3D 복셀 기반 점유 예측
- 시맨틱 클래스 + 모션 플로우 예측
- 충돌 검사 기능 포함

### 5. TeslaFSDVision (`models/fsd_vision.py`)
- 모든 컴포넌트를 통합한 완전한 시스템
- 8개 카메라 입력 처리
- 시간적 퓨전 지원

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tesla-fsd-vision.git
cd tesla-fsd-vision

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from models import TeslaFSDVision, create_fsd_vision

# Create model
model = create_fsd_vision(config='default')

# Prepare input (8 cameras)
images = torch.randn(1, 8, 3, 960, 1280)

# Inference
with torch.no_grad():
    outputs = model(images)

# Access outputs
print(outputs.keys())
# ['hydranet', 'bev_features', 'occupancy', 'detections',
#  'traffic_lights', 'lanes', 'depth', 'path']
```

## Demo

```bash
# Run architecture demo
python demo/run_demo.py

# Run visualization
python demo/visualizer.py
```

## Output Example

```
📦 Object Detections:
  • VEHICLE: 45m | 62 km/h | conf=0.95
  • PEDESTRIAN: 28m | 5 km/h | conf=0.92

🚦 Traffic Lights:
  • GREEN: 85m | 12s remaining | relevance=0.98

🛣️ Lane Detection:
  • Ego lane: Detected ✓
  • Lane offset: 0.15m left of center

📊 Neural Network Activity:
  - Vision:     ████████░░ 85%
  - Planning:   ███████░░░ 72%
  - Control:    █████████░ 90%
  - Prediction: ██████░░░░ 68%
```

## Model Configurations

| Config | Backbone | d_model | BEV Size | Params |
|--------|----------|---------|----------|--------|
| small  | RegNet-400MF | 128 | 100×100 | ~15M |
| default | Tesla Custom | 256 | 200×200 | ~50M |
| large  | RegNet-8GF | 512 | 400×400 | ~150M |

## Key Technologies

### Camera-to-BEV Projection
- 8개 카메라의 2D 이미지 특징을 통합 3D BEV 공간으로 변환
- Learnable BEV queries 사용
- Cross-attention으로 이미지 특징 참조

### Occupancy Network
- 3D 복셀 그리드로 장면 표현
- 객체 박스가 아닌 점유 확률로 표현
- 미지의 객체도 감지 가능 (long-tail problem 해결)

### Temporal Fusion
- 연속 프레임 BEV 특징 퓨전
- 자차 운동 보상
- 속도/가속도 추정 향상

## File Structure

```
tesla-fsd-vision/
├── models/
│   ├── __init__.py
│   ├── backbone.py         # RegNet, EfficientNet
│   ├── hydranet.py         # Multi-task network
│   ├── detection_heads.py  # Task-specific heads
│   ├── bev_transformer.py  # 2D→3D transformer
│   ├── occupancy_network.py # 3D occupancy
│   └── fsd_vision.py       # Complete system
├── utils/
│   ├── __init__.py
│   ├── camera.py           # Camera models
│   └── transforms.py       # Image transforms
├── demo/
│   ├── run_demo.py         # Architecture demo
│   └── visualizer.py       # Tesla-style visualization
├── configs/                # Configuration files
├── data/                   # Data handling
├── requirements.txt
└── README.md
```

## References

- [Tesla AI Day 2021](https://www.youtube.com/watch?v=j0z4FweCy4M)
- [Tesla AI Day 2022](https://www.youtube.com/watch?v=ODSJsviD_SU)
- [BEVFormer](https://arxiv.org/abs/2203.17270)
- [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711)
- [RegNet](https://arxiv.org/abs/2003.13678)

## Disclaimer

이 프로젝트는 교육 및 연구 목적으로 만들어진 Tesla FSD 비전 시스템의 재구현입니다.
Tesla의 공식 구현이 아니며, 실제 차량에 사용해서는 안 됩니다.

## License

MIT License
