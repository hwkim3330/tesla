"""
TensorRT Engine Builder for FSD Lite

PyTorch 모델을 TensorRT 엔진으로 변환.
Orin Nano에서 최대 성능 달성을 위한 최적화.

사용법:
    python build_engine.py --model fsd_lite.pt --fp16
    python build_engine.py --model fsd_lite.pt --int8 --calib-data calib/
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import FSDLite


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: dict,
    opset_version: int = 17,
    dynamic_axes: dict = None,
):
    """
    PyTorch 모델을 ONNX로 변환.

    Args:
        model: PyTorch 모델
        output_path: ONNX 파일 경로
        input_shapes: 입력 텐서 형태
        opset_version: ONNX opset 버전
        dynamic_axes: 동적 축 설정
    """
    print(f"Exporting to ONNX: {output_path}")

    model.eval()

    # Create dummy inputs
    dummy_inputs = {}
    for name, shape in input_shapes.items():
        dummy_inputs[name] = torch.randn(*shape)

    # Export
    torch.onnx.export(
        model,
        (dummy_inputs['camera'], dummy_inputs.get('lidar')),
        output_path,
        input_names=list(input_shapes.keys()),
        output_names=['steering', 'throttle', 'brake', 'confidence'],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    print(f"ONNX model saved: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    return output_path


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    calib_data_path: str = None,
    workspace_size: int = 1 << 30,  # 1GB
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
):
    """
    ONNX 모델을 TensorRT 엔진으로 변환.

    Args:
        onnx_path: ONNX 파일 경로
        engine_path: TensorRT 엔진 저장 경로
        fp16: FP16 모드 활성화
        int8: INT8 모드 활성화
        calib_data_path: INT8 캘리브레이션 데이터 경로
        workspace_size: GPU 작업 메모리 크기
        min_batch: 최소 배치 크기
        opt_batch: 최적 배치 크기
        max_batch: 최대 배치 크기
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not found. Install on Jetson with:")
        print("  sudo apt-get install python3-libnvinfer-dev")
        return None

    print(f"\nBuilding TensorRT engine...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Engine: {engine_path}")
    print(f"  FP16: {fp16}")
    print(f"  INT8: {int8}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # Create network with explicit batch
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(error)}")
            return None

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()

    # Camera input shape
    profile.set_shape(
        'camera',
        min=(min_batch, 3, 480, 640),
        opt=(opt_batch, 3, 480, 640),
        max=(max_batch, 3, 480, 640),
    )

    # LiDAR input shape (optional)
    if network.num_inputs > 1:
        profile.set_shape(
            'lidar',
            min=(min_batch, 4, 200, 200),
            opt=(opt_batch, 4, 200, 200),
            max=(max_batch, 4, 200, 200),
        )

    config.add_optimization_profile(profile)

    # Precision settings
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

        if calib_data_path:
            # INT8 calibration
            calibrator = Int8Calibrator(calib_data_path)
            config.int8_calibrator = calibrator
        else:
            print("  Warning: INT8 without calibration data")

    # Build engine
    print("\nBuilding engine (this may take several minutes)...")
    start_time = time.time()

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None

    build_time = time.time() - start_time
    print(f"Engine built in {build_time:.1f} seconds")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved: {engine_path}")
    print(f"Engine size: {os.path.getsize(engine_path) / 1e6:.2f} MB")

    return engine_path


class Int8Calibrator:
    """
    INT8 양자화를 위한 캘리브레이터.

    실제 데이터를 사용하여 양자화 범위를 결정.
    """

    def __init__(
        self,
        data_path: str,
        cache_file: str = 'int8_calib.cache',
        batch_size: int = 8,
        num_batches: int = 100,
    ):
        try:
            import tensorrt as trt
            self.trt = trt
        except ImportError:
            raise ImportError("TensorRT required for INT8 calibration")

        self.data_path = data_path
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0

        # Load calibration data
        self._load_data()

    def _load_data(self):
        """캘리브레이션 데이터 로드."""
        import numpy as np
        from glob import glob

        # Find image files
        image_files = sorted(glob(os.path.join(self.data_path, '*.npy')))

        if not image_files:
            print(f"No calibration data found in {self.data_path}")
            self.data = None
            return

        # Load and preprocess
        self.data = []
        for f in image_files[:self.num_batches * self.batch_size]:
            arr = np.load(f).astype(np.float32)
            self.data.append(arr)

        print(f"Loaded {len(self.data)} calibration samples")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """다음 배치 반환."""
        if self.data is None:
            return None

        if self.current_batch >= self.num_batches:
            return None

        import numpy as np
        import pycuda.driver as cuda

        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size

        batch = np.stack(self.data[start_idx:end_idx])
        self.current_batch += 1

        # Allocate GPU memory
        d_input = cuda.mem_alloc(batch.nbytes)
        cuda.memcpy_htod(d_input, batch.ravel())

        return [int(d_input)]

    def read_calibration_cache(self):
        """캐시에서 캘리브레이션 데이터 읽기."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """캘리브레이션 데이터 캐시에 저장."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def benchmark_engine(engine_path: str, num_iterations: int = 100):
    """
    TensorRT 엔진 벤치마크.

    Args:
        engine_path: 엔진 파일 경로
        num_iterations: 반복 횟수
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
    except ImportError:
        print("TensorRT/PyCUDA required for benchmarking")
        return

    print(f"\nBenchmarking: {engine_path}")

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = context.get_tensor_shape(name)
        size = int(np.prod(shape))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            context.set_tensor_address(name, int(device_mem))
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            context.set_tensor_address(name, int(device_mem))

    # Warmup
    print("Warming up...")
    for _ in range(10):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()

    # Benchmark
    print(f"Running {num_iterations} iterations...")

    times = []
    for _ in range(num_iterations):
        start = time.time()

        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()

        times.append(time.time() - start)

    # Results
    times = np.array(times) * 1000  # ms

    print(f"\nResults:")
    print(f"  Mean latency: {np.mean(times):.2f} ms")
    print(f"  Std latency:  {np.std(times):.2f} ms")
    print(f"  Min latency:  {np.min(times):.2f} ms")
    print(f"  Max latency:  {np.max(times):.2f} ms")
    print(f"  FPS:          {1000 / np.mean(times):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT Engine for FSD Lite')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='PyTorch checkpoint path')
    parser.add_argument('--onnx', type=str, default='fsd_lite.onnx',
                        help='ONNX output path')
    parser.add_argument('--engine', type=str, default='fsd_lite.trt',
                        help='TensorRT engine output path')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 mode')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 mode')
    parser.add_argument('--calib-data', type=str, default=None,
                        help='INT8 calibration data path')
    parser.add_argument('--workspace', type=int, default=1024,
                        help='Workspace size in MB')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark the engine')
    parser.add_argument('--use-lidar', action='store_true',
                        help='Include LiDAR input')

    args = parser.parse_args()

    # Create model
    print("Creating FSD Lite model...")
    model = FSDLite(use_lidar=args.use_lidar)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)

    model.eval()

    # Input shapes
    input_shapes = {
        'camera': (1, 3, 480, 640),
    }
    if args.use_lidar:
        input_shapes['lidar'] = (1, 4, 200, 200)

    # Export to ONNX
    onnx_path = export_onnx(
        model=model,
        output_path=args.onnx,
        input_shapes=input_shapes,
    )

    # Build TensorRT engine
    engine_path = build_tensorrt_engine(
        onnx_path=onnx_path,
        engine_path=args.engine,
        fp16=args.fp16,
        int8=args.int8,
        calib_data_path=args.calib_data,
        workspace_size=args.workspace * (1 << 20),
    )

    # Benchmark
    if args.benchmark and engine_path:
        benchmark_engine(engine_path)


if __name__ == '__main__':
    main()
