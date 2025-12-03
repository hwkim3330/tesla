# Tesla FSD 3D Visualization

테슬라 FSD(Full Self-Driving) 개발자 모드 시각화 UI 클론 - Three.js 3D + Mapbox 실시간 지도 통합

## Features

### 3D Map View (왼쪽 패널)
- 🗺️ Mapbox GL JS 실시간 3D 지도
- 🏙️ 3D 건물 렌더링
- 🛣️ 실시간 경로 표시 (녹색 라인)
- 📍 자동 경로 추적 및 방향 전환
- 🚗 3D 차량 모델 오버레이

### Bird's Eye View (오른쪽 패널)
- 📊 Three.js 3D BEV 렌더링
- 🚙 자차 위치 (녹색 3D 모델)
- 🚗 타 차량 위치 (파란색 3D 모델)
- 🚶 보행자 위치 (노란색 3D 모델)
- 🚦 신호등 3D 모델 + 상태
- 📏 거리 마커 (20m ~ 100m)
- 🛤️ 차선 렌더링
- 🎯 예측 경로 (녹색 영역)

### HUD 정보
- ⚡ 현재 속도 / 설정 속도
- 📍 현재 도로명
- 📏 선행 차량 거리
- ⏱️ TTC (Time To Collision)
- 🚦 신호등 상태 + 거리 + 타이머
- 🧠 Neural Network 활성도 (Vision, Planning, Control, Prediction)
- 📍 Navigation 정보 (ETA, 남은 거리)

### View Modes
- **3D View**: 60도 피치 3D 뷰
- **Top View**: 수직 탑다운 뷰
- **Follow**: 75도 피치 밀착 추적

### Detection System
- 🚗 차량 감지 (VEHICLE)
- 🚶 보행자 감지 (PEDESTRIAN)
- 🚦 신호등 감지 + 상태

## Tech Stack

- **Three.js** - 3D WebGL 렌더링
- **Mapbox GL JS** - 실시간 3D 지도
- **Canvas API** - 오버레이 렌더링
- **CSS Grid/Flexbox** - 반응형 레이아웃
- **CSS Animations** - UI 애니메이션

## Demo

https://hwkim3330.github.io/tesla/

## Screenshots

### Main Interface
- 왼쪽: Mapbox 3D 지도 + 경로 + 3D 건물
- 오른쪽: Three.js BEV + 차량 정보 + NN 활성도

### BEV (Bird's Eye View)
- 3D 그리드 기반 탑다운 뷰
- 실시간 객체 추적
- 예측 경로 시각화

## Key Features

### Real-time Simulation
- 속도 변화 시뮬레이션
- 선행 차량 거리/속도 변화
- 신호등 상태 변화 (Red → Yellow → Green)
- 객체 위치 업데이트

### 3D Models
- **Ego Vehicle**: 녹색 Tesla 스타일 3D 모델
- **Other Vehicles**: Sedan, SUV, Truck 타입별 모델
- **Pedestrians**: 실린더 + 구체 조합 3D 모델
- **Traffic Lights**: 폴 + 신호등 박스 3D 모델

### Map Integration
- 서울 강남역 중심 시작
- 실시간 경로 추적
- 3D 건물 높이 기반 렌더링
- 부드러운 카메라 이동

## Related Projects

- [Dash](https://github.com/nicholaswmin/dash) - WebGL Self-driving car simulator (Three.js)
- [G3D](https://github.com/nicholaswmin/g3d) - Three.js + Mapbox integration
- [OpenDriveJS](https://github.com/nicholaswmin/opendrivejs) - Three.js ASAM OpenDrive visualizer
- [EinsteinVision](https://github.com/nicholaswmin/EinsteinVision) - Tesla-inspired visualization

## License

MIT
