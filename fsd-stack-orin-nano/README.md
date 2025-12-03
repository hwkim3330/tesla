# Tesla FSD Lite - Jetson Orin Nano Optimized

Jetson Orin Nano에서 실시간 동작하는 경량화된 자율주행 신경망 스택.

## 왜 이 구조가 빠른가?

기존 FSD 스택의 무거운 컴포넌트를 Orin Nano에 최적화된 경량 버전으로 교체했습니다.

### 설계 원칙

| 기존 방식 | 최적화 방식 | 속도 향상 | 이유 |
|-----------|-------------|-----------|------|
| ResNet-50 / ViT | **MobileNetV3-Small** | ~5x | 모바일/엣지용 설계, 파라미터 80% 감소 |
| Cross-Attention Fusion | **Concatenation** | ~10x | 연산 비용 거의 0, 후속 레이어가 관계 학습 |
| Transformer (시계열) | **GRU** | ~3x | LSTM보다 단순, 비슷한 성능 |
| FP32 연산 | **FP16/INT8** | ~2-4x | Orin Nano의 Tensor Core 활용 |
| PyTorch 직접 실행 | **TensorRT 엔진** | ~3-5x | GPU 커널 최적화, 레이어 융합 |

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Orin Nano Optimized FSD Architecture                  │
│                        Target: 20+ FPS @ 8GB RAM                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐        ┌──────────────┐                               │
│  │   Camera     │        │    LiDAR     │                               │
│  │  (640x480)   │        │  (BEV Grid)  │                               │
│  └──────┬───────┘        └──────┬───────┘                               │
│         │                       │                                        │
│         ▼                       ▼                                        │
│  ┌──────────────┐        ┌──────────────┐                               │
│  │ MobileNetV3  │        │  PointPillar │                               │
│  │    Small     │        │    Lite      │                               │
│  │   (1.5M)     │        │   (0.5M)     │                               │
│  └──────┬───────┘        └──────┬───────┘                               │
│         │                       │                                        │
│         │    ┌──────────────────┘                                        │
│         │    │                                                           │
│         ▼    ▼                                                           │
│  ┌────────────────────┐                                                  │
│  │   Concatenation    │  ← 단순 연결 (연산 비용 ≈ 0)                     │
│  │   (No Attention)   │                                                  │
│  └─────────┬──────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│  ┌────────────────────┐                                                  │
│  │       GRU          │  ← 시계열 처리 (Transformer 대비 3x 빠름)        │
│  │  (Hidden: 256)     │                                                  │
│  └─────────┬──────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│  ┌────────────────────┐                                                  │
│  │   Lightweight MLP  │                                                  │
│  │    Decision Head   │                                                  │
│  └─────────┬──────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│  ┌────────────────────┐                                                  │
│  │  Control Output    │                                                  │
│  │ (Steering, Accel)  │                                                  │
│  └────────────────────┘                                                  │
│                                                                          │
│  Total Parameters: ~3M (FP16: 6MB, INT8: 3MB)                           │
│  Target Inference: 20-30 FPS on Orin Nano                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 최적화 상세 설명

### 1. 인코더 교체: ResNet/ViT → MobileNetV3-Small

```
문제: ResNet-50 (25M params), ViT-Base (86M params) → Orin Nano에 너무 무거움
해결: MobileNetV3-Small (1.5M params) → 파라미터 94% 감소

MobileNetV3의 핵심 기술:
├── Inverted Residual Blocks (메모리 효율적)
├── Squeeze-and-Excitation (채널 어텐션, 경량)
├── h-swish Activation (ReLU보다 약간 느리지만 정확도 향상)
└── Neural Architecture Search로 발견된 최적 구조
```

### 2. 퓨전 방식 단순화: Attention → Concatenation

```
문제: Cross-Attention 연산 복잡도 O(N²)
해결: Concatenation 연산 복잡도 O(N)

Cross-Attention (기존):
  Q, K, V = Linear(cam), Linear(lidar), Linear(lidar)
  Attention = Softmax(Q @ K.T / sqrt(d)) @ V  ← 무거움!

Concatenation (최적화):
  fused = torch.cat([cam_feat, lidar_feat], dim=-1)  ← 거의 공짜!
  output = MLP(fused)  ← 후속 레이어가 관계 학습
```

### 3. 시계열 모델 변경: Transformer → GRU

```
모델 비교 (시계열 처리):
┌─────────────┬────────────┬─────────────┬──────────────┐
│    Model    │  Params    │  Inference  │  특징        │
├─────────────┼────────────┼─────────────┼──────────────┤
│ Transformer │   많음     │    느림     │ 병렬화 가능   │
│ LSTM        │   보통     │    보통     │ 순차 처리    │
│ GRU         │   적음     │    빠름     │ LSTM 대비 단순│
└─────────────┴────────────┴─────────────┴──────────────┘

GRU가 LSTM보다 빠른 이유:
- Gate 개수: LSTM (3개) vs GRU (2개)
- Hidden state: LSTM (cell + hidden) vs GRU (hidden만)
- 파라미터: GRU가 ~25% 적음
- 성능: 대부분 태스크에서 비슷
```

### 4. 필수 최적화 기법

#### TensorRT 변환

```python
# PyTorch → ONNX → TensorRT
import torch
import tensorrt as trt

# 1. ONNX 변환
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17)

# 2. TensorRT 엔진 빌드 (FP16 활성화)
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=1024 \
        --buildOnly

# 속도 향상: PyTorch 대비 3-5x
```

#### FP16/INT8 양자화

```
정밀도별 비교:
┌────────┬───────────┬───────────┬───────────────┐
│ 정밀도 │ 메모리    │ 속도      │ 정확도 손실   │
├────────┼───────────┼───────────┼───────────────┤
│ FP32   │ 100%      │ 1x        │ 0%            │
│ FP16   │ 50%       │ 2x        │ <0.5%         │
│ INT8   │ 25%       │ 4x        │ 1-3%          │
└────────┴───────────┴───────────┴───────────────┘

Orin Nano는 FP16 연산을 하드웨어 레벨에서 지원 (Tensor Cores)
→ FP16 사용 시 정확도 손실 거의 없이 2배 속도 향상
```

## 성능 목표

| 항목 | 목표 | 비고 |
|------|------|------|
| 추론 속도 | 20-30 FPS | TensorRT FP16 |
| 메모리 사용 | < 4GB | Orin Nano 8GB 중 절반 |
| 모델 크기 | < 10MB | FP16 기준 |
| 지연 시간 | < 50ms | End-to-End |

## 사용법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. TensorRT 엔진 빌드
python inference/build_engine.py --fp16

# 3. 실시간 추론
python inference/run_realtime.py --camera 0 --engine model.trt
```

## 파일 구조

```
fsd-stack-orin-nano/
├── models/
│   ├── mobilenet_encoder.py    # MobileNetV3-Small 인코더
│   ├── pointpillar_lite.py     # 경량 LiDAR 인코더
│   ├── gru_temporal.py         # GRU 시계열 모듈
│   ├── fusion_concat.py        # Concatenation 퓨전
│   └── fsd_lite.py             # 통합 네트워크
├── inference/
│   ├── build_engine.py         # TensorRT 엔진 빌드
│   └── run_realtime.py         # 실시간 추론
├── configs/
│   └── orin_nano.yaml          # Orin Nano 최적화 설정
└── docs/
    └── optimization_guide.md   # 최적화 가이드
```

## 결론

이 구조는 **Jetson Orin Nano의 제한된 자원** (GPU 코어, 메모리 대역폭) 내에서:

1. **카메라 + LiDAR** 입력을 처리하고
2. **과거 상황을 고려**한 판단을 수행하며
3. **실시간 제어 출력**을 생성할 수 있는

**가장 현실적이고 빠른 마지노선 형태의 아키텍처**입니다.

더 무거운 모델은 Orin Nano에서 실시간 동작이 불가능하고,
더 가벼운 모델은 자율주행에 필요한 기능을 수행할 수 없습니다.
