# Tesla FSD V12 - End-to-End Neural Network

테슬라 FSD V12의 End-to-End 신경망 아키텍처를 시각화한 인터랙티브 웹 데모입니다.

## Live Demo

https://hwkim3330.github.io/tesla-fsd-v12/

## What is FSD V12?

FSD V12는 테슬라가 2024년부터 배포하기 시작한 완전히 새로운 자율주행 시스템입니다.

### 핵심 변화: "Photon to Control"

**기존 (V11 이하)**
- 30만 줄 이상의 C++ 코드
- 여러 개의 독립적인 신경망
- 명시적인 규칙 기반 로직
- 객체 → 분류 → 판단 → 제어

**V12 (End-to-End)**
- 단일 거대 신경망 (10억+ 파라미터)
- 카메라 영상 → 직접 제어 출력
- 규칙 코드 없음
- 인간 운전 데이터로 학습

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    8 Camera Images                       │
│   (Front Main/Narrow/Wide, B-Pillars, Sides, Rear)      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    RegNet Backbone                       │
│              (Feature Extraction)                        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Temporal Module                         │
│         (Video Memory, Motion Analysis)                  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            Transformer + World Model                     │
│    (Self-Attention, Future Prediction, Risk Assessment)  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Control Output                        │
│            (Steering, Acceleration, Brake)               │
└─────────────────────────────────────────────────────────┘
```

## Key Technologies

### 1. Video Transformer
- 정지 이미지가 아닌 비디오 시퀀스 처리
- 시간적 맥락을 통해 속도, 방향 변화 감지
- Self-Attention으로 중요한 영역에 집중

### 2. World Model
- 미래 상황 예측 (Imagination)
- "핸들을 돌리면 어떻게 될까?" 시뮬레이션
- 위험 상황 사전 감지

### 3. Behavior Cloning
- 수천만 건의 인간 운전 데이터로 학습
- "전문 운전자라면 이 상황에서 어떻게 할까?"
- 모범 운전만 선별하여 학습

## Features of This Demo

- **Hero Animation**: 신경망 연결 시각화
- **Architecture Diagram**: 파이프라인 구조 다이어그램
- **Neural Network Viz**: 레이어별 활성화 애니메이션
- **Live Simulation**: 실시간 주행 시뮬레이션
- **Data Flow**: 데이터 흐름 애니메이션
- **V11 vs V12 Comparison**: 아키텍처 비교

## Tech Stack

- Vanilla JavaScript (ES6+)
- HTML5 Canvas API
- CSS3 Animations
- No external frameworks

## Project Structure

```
tesla-fsd-v12/
├── index.html          # Main HTML
├── js/
│   ├── main.js         # Hero animation, interactions
│   ├── neural-network.js  # NN visualization
│   ├── simulation.js   # Driving simulation
│   └── dataflow.js     # Data flow animation
└── README.md
```

## Local Development

```bash
# Clone
git clone https://github.com/hwkim3330/tesla-fsd-v12.git
cd tesla-fsd-v12

# Serve locally (any static server)
python -m http.server 8000
# or
npx serve .
```

## References

- [Tesla AI Day 2024](https://www.youtube.com/tesla)
- [Andrej Karpathy - The spelled-out intro to neural networks](https://karpathy.ai/)
- [Ashok Elluswamy on FSD V12](https://twitter.com/aaboride)

## Disclaimer

이 프로젝트는 교육 목적으로 만들어진 Tesla FSD V12 아키텍처의 시각화입니다.
Tesla, Inc.와 관련이 없으며, 실제 FSD 시스템의 정확한 구현이 아닙니다.

## License

MIT License
