#!/usr/bin/env python3
"""
Real-time FSD Lite Inference on Jetson Orin Nano

카메라 입력을 실시간으로 처리하여 제어 명령 출력.
TensorRT 엔진 또는 PyTorch 모델 사용 가능.

사용법:
    # TensorRT 엔진 사용 (권장)
    python run_realtime.py --engine fsd_lite.trt --camera 0

    # PyTorch 모델 사용
    python run_realtime.py --checkpoint fsd_lite.pt --camera 0

    # 비디오 파일 사용
    python run_realtime.py --engine fsd_lite.trt --video driving.mp4
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import queue

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class InferenceResult:
    """추론 결과."""
    steering: float      # -1 (좌) ~ 1 (우)
    throttle: float      # 0 ~ 1
    brake: float         # 0 ~ 1
    confidence: float    # 0 ~ 1
    latency_ms: float    # 추론 시간
    fps: float           # 초당 프레임


class TensorRTInference:
    """TensorRT 기반 추론 엔진."""

    def __init__(self, engine_path: str):
        """
        TensorRT 엔진 로드.

        Args:
            engine_path: .trt 엔진 파일 경로
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT and PyCUDA required. "
                "Install on Jetson with: sudo apt-get install python3-libnvinfer-dev"
            )

        self.cuda = cuda
        self.trt = trt

        print(f"Loading TensorRT engine: {engine_path}")

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

        print(f"Engine loaded successfully")
        print(f"  Inputs: {[inp['name'] for inp in self.inputs]}")
        print(f"  Outputs: {[out['name'] for out in self.outputs]}")

    def _allocate_buffers(self):
        """GPU 메모리 할당."""
        self.inputs = []
        self.outputs = []
        self.stream = self.cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = int(np.prod(shape))

            # Allocate host and device buffers
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)

            buffer = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype,
            }

            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.inputs.append(buffer)
                self.context.set_tensor_address(name, int(device_mem))
            else:
                self.outputs.append(buffer)
                self.context.set_tensor_address(name, int(device_mem))

    def predict(
        self,
        camera_image: np.ndarray,
        lidar_bev: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        """
        추론 수행.

        Args:
            camera_image: 카메라 이미지 (H, W, 3) or (3, H, W)
            lidar_bev: LiDAR BEV 그리드 (선택)

        Returns:
            InferenceResult
        """
        start_time = time.time()

        # Preprocess camera image
        if camera_image.shape[-1] == 3:  # HWC -> CHW
            camera_image = camera_image.transpose(2, 0, 1)

        camera_image = camera_image.astype(np.float32) / 255.0
        camera_image = np.expand_dims(camera_image, 0)  # Add batch dim

        # Copy to input buffer
        np.copyto(self.inputs[0]['host'], camera_image.ravel())

        # Copy LiDAR if available
        if lidar_bev is not None and len(self.inputs) > 1:
            lidar_bev = np.expand_dims(lidar_bev, 0)
            np.copyto(self.inputs[1]['host'], lidar_bev.ravel())

        # Transfer to GPU
        for inp in self.inputs:
            self.cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer from GPU
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        # Parse outputs
        latency = (time.time() - start_time) * 1000

        # Assuming outputs are in order: steering, throttle, brake, confidence
        return InferenceResult(
            steering=float(self.outputs[0]['host'][0]),
            throttle=float(self.outputs[1]['host'][0]),
            brake=float(self.outputs[2]['host'][0]),
            confidence=float(self.outputs[3]['host'][0]),
            latency_ms=latency,
            fps=1000 / latency,
        )


class PyTorchInference:
    """PyTorch 기반 추론."""

    def __init__(self, checkpoint_path: str, use_lidar: bool = False):
        """
        PyTorch 모델 로드.

        Args:
            checkpoint_path: 체크포인트 경로
            use_lidar: LiDAR 사용 여부
        """
        import torch
        from models import FSDLite

        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading PyTorch model: {checkpoint_path}")
        print(f"Device: {self.device}")

        # Create model
        self.model = FSDLite(use_lidar=use_lidar)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Enable TensorFloat-32 on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("Model loaded successfully")

    @torch.no_grad()
    def predict(
        self,
        camera_image: np.ndarray,
        lidar_bev: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        """추론 수행."""
        start_time = time.time()

        # Preprocess
        if camera_image.shape[-1] == 3:
            camera_image = camera_image.transpose(2, 0, 1)

        camera_tensor = self.torch.from_numpy(camera_image).float() / 255.0
        camera_tensor = camera_tensor.unsqueeze(0).to(self.device)

        lidar_tensor = None
        if lidar_bev is not None:
            lidar_tensor = self.torch.from_numpy(lidar_bev).float()
            lidar_tensor = lidar_tensor.unsqueeze(0).to(self.device)

        # Inference
        outputs = self.model(camera_tensor, lidar_tensor)

        latency = (time.time() - start_time) * 1000

        return InferenceResult(
            steering=outputs['steering'][0].item(),
            throttle=outputs['throttle'][0].item(),
            brake=outputs['brake'][0].item(),
            confidence=outputs['confidence'][0].item(),
            latency_ms=latency,
            fps=1000 / latency,
        )


class CameraCapture:
    """비동기 카메라 캡처."""

    def __init__(
        self,
        source: int | str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        카메라 초기화.

        Args:
            source: 카메라 인덱스 또는 비디오 파일 경로
            width: 프레임 너비
            height: 프레임 높이
            fps: 목표 FPS
        """
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV required: pip install opencv-python")

        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps

        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None

        self._init_capture()

    def _init_capture(self):
        """캡처 장치 초기화."""
        self.cap = self.cv2.VideoCapture(self.source)

        if isinstance(self.source, int):
            # Camera settings
            self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(self.cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.source}")

        print(f"Camera opened: {self.source}")
        print(f"  Resolution: {int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {self.cap.get(self.cv2.CAP_PROP_FPS)}")

    def _capture_loop(self):
        """캡처 스레드 루프."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Drop old frame if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                if isinstance(self.source, str):
                    # Video file ended, restart
                    self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.001)

    def start(self):
        """캡처 시작."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """캡처 중지."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """최신 프레임 가져오기."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class RealtimeVisualizer:
    """실시간 시각화."""

    def __init__(self, width: int = 800, height: int = 600):
        """시각화 초기화."""
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV required")

        self.width = width
        self.height = height
        self.window_name = "FSD Lite - Real-time Inference"

    def draw(
        self,
        frame: np.ndarray,
        result: InferenceResult,
    ) -> np.ndarray:
        """
        결과 시각화.

        Args:
            frame: 입력 프레임
            result: 추론 결과

        Returns:
            시각화된 프레임
        """
        # Resize frame
        vis_frame = self.cv2.resize(frame, (self.width, self.height))

        # Draw control indicators
        self._draw_steering(vis_frame, result.steering)
        self._draw_pedals(vis_frame, result.throttle, result.brake)
        self._draw_info(vis_frame, result)

        return vis_frame

    def _draw_steering(self, frame: np.ndarray, steering: float):
        """스티어링 휠 표시."""
        center_x = self.width // 2
        center_y = self.height - 100

        # Wheel circle
        self.cv2.circle(frame, (center_x, center_y), 60, (100, 100, 100), 2)

        # Steering indicator
        angle = steering * 45  # Max 45 degrees
        rad = np.radians(angle - 90)
        end_x = int(center_x + 50 * np.cos(rad))
        end_y = int(center_y + 50 * np.sin(rad))

        color = (0, 255, 0) if abs(steering) < 0.3 else (0, 165, 255)
        self.cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 3)

        # Value text
        text = f"Steering: {steering:+.2f}"
        self.cv2.putText(frame, text, (center_x - 60, center_y + 80),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_pedals(self, frame: np.ndarray, throttle: float, brake: float):
        """페달 표시."""
        bar_width = 30
        bar_height = 100

        # Throttle (right)
        x = self.width - 100
        y = self.height - 150

        self.cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                          (100, 100, 100), 2)
        fill_height = int(bar_height * throttle)
        self.cv2.rectangle(frame, (x, y + bar_height - fill_height),
                          (x + bar_width, y + bar_height), (0, 255, 0), -1)
        self.cv2.putText(frame, f"T:{throttle:.2f}", (x - 10, y + bar_height + 20),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Brake (left)
        x = self.width - 150
        self.cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                          (100, 100, 100), 2)
        fill_height = int(bar_height * brake)
        self.cv2.rectangle(frame, (x, y + bar_height - fill_height),
                          (x + bar_width, y + bar_height), (0, 0, 255), -1)
        self.cv2.putText(frame, f"B:{brake:.2f}", (x - 10, y + bar_height + 20),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_info(self, frame: np.ndarray, result: InferenceResult):
        """정보 표시."""
        info_lines = [
            f"Latency: {result.latency_ms:.1f}ms",
            f"FPS: {result.fps:.1f}",
            f"Confidence: {result.confidence:.2f}",
        ]

        y = 30
        for line in info_lines:
            color = (0, 255, 0) if result.latency_ms < 50 else (0, 165, 255)
            self.cv2.putText(frame, line, (20, y),
                            self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

    def show(self, frame: np.ndarray) -> bool:
        """
        프레임 표시.

        Returns:
            False if window closed
        """
        self.cv2.imshow(self.window_name, frame)
        key = self.cv2.waitKey(1) & 0xFF
        return key != 27 and key != ord('q')  # ESC or Q to quit

    def close(self):
        """윈도우 닫기."""
        self.cv2.destroyAllWindows()


def run_realtime(
    inference_engine,
    camera: CameraCapture,
    visualizer: Optional[RealtimeVisualizer] = None,
    target_size: Tuple[int, int] = (640, 480),
):
    """
    실시간 추론 루프.

    Args:
        inference_engine: 추론 엔진 (TensorRT 또는 PyTorch)
        camera: 카메라 캡처
        visualizer: 시각화 (선택)
        target_size: 입력 이미지 크기
    """
    import cv2

    print("\nStarting real-time inference...")
    print("Press 'Q' or ESC to quit\n")

    camera.start()

    fps_counter = []
    try:
        while True:
            # Get frame
            frame = camera.get_frame(timeout=1.0)
            if frame is None:
                continue

            # Resize for inference
            input_frame = cv2.resize(frame, target_size)

            # Run inference
            result = inference_engine.predict(input_frame)

            # Track FPS
            fps_counter.append(result.fps)
            if len(fps_counter) > 100:
                fps_counter.pop(0)

            avg_fps = np.mean(fps_counter)

            # Print results
            print(f"\rSteering: {result.steering:+.2f} | "
                  f"Throttle: {result.throttle:.2f} | "
                  f"Brake: {result.brake:.2f} | "
                  f"Conf: {result.confidence:.2f} | "
                  f"Latency: {result.latency_ms:.1f}ms | "
                  f"FPS: {avg_fps:.1f}", end='')

            # Visualize
            if visualizer:
                vis_frame = visualizer.draw(frame, result)
                if not visualizer.show(vis_frame):
                    break

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        camera.stop()
        if visualizer:
            visualizer.close()


def main():
    parser = argparse.ArgumentParser(description='Real-time FSD Lite Inference')

    # Model options
    parser.add_argument('--engine', type=str, default=None,
                        help='TensorRT engine path (.trt)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='PyTorch checkpoint path (.pt)')
    parser.add_argument('--use-lidar', action='store_true',
                        help='Enable LiDAR input')

    # Input options
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    parser.add_argument('--video', type=str, default=None,
                        help='Video file path')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width')
    parser.add_argument('--height', type=int, default=480,
                        help='Input height')

    # Display options
    parser.add_argument('--no-display', action='store_true',
                        help='Disable visualization')

    args = parser.parse_args()

    # Initialize inference engine
    if args.engine:
        inference = TensorRTInference(args.engine)
    elif args.checkpoint:
        inference = PyTorchInference(args.checkpoint, use_lidar=args.use_lidar)
    else:
        print("Creating model with random weights (no checkpoint specified)")
        inference = PyTorchInference(None, use_lidar=args.use_lidar)

    # Initialize camera
    source = args.video if args.video else args.camera
    camera = CameraCapture(source, args.width, args.height)

    # Initialize visualizer
    visualizer = None if args.no_display else RealtimeVisualizer()

    # Run
    run_realtime(
        inference_engine=inference,
        camera=camera,
        visualizer=visualizer,
        target_size=(args.width, args.height),
    )


if __name__ == '__main__':
    main()
