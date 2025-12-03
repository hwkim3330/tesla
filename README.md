# Tesla FSD Visualization

테슬라 FSD(Full Self-Driving) 개발자 모드 시각화 UI 클론입니다.

## Features

### Camera View (왼쪽 패널)
- 🚗 차량 감지 (파란색 박스)
- 🚶 보행자 감지 (노란색 박스)
- 🚦 신호등 감지 + 거리/타이머
- 🛣️ 차선 인식 (파란색 라인)
- 📍 예측 경로 (녹색 영역)

### Bird's Eye View (오른쪽 패널)
- 📊 BEV 탑다운 뷰
- 🚙 자차 위치 (녹색)
- 🚗 타 차량 위치 (파란색)
- 🚶 보행자 위치 (노란색)
- 🚦 신호등 상태

### HUD 정보
- ⚡ 현재 속도 / 설정 속도
- 📏 선행 차량 거리
- ⏱️ TTC (Time To Collision)
- 🚦 다음 신호등 상태/타이머
- 🧠 Neural Network 활성도

## Tech Stack

- Pure HTML/CSS/JavaScript
- Canvas API for BEV rendering
- CSS Grid/Flexbox for layout
- CSS Animations

## Demo

https://hwkim3330.github.io/tesla/

## Related Projects

- [OpenPilot](https://github.com/commaai/openpilot) - 실제 동작하는 ADAS
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) - Camera→BEV 변환
- [OpenFSD](https://github.com/open-fsd/open-fsd) - Tesla FSD 재현 프로젝트

## License

MIT
