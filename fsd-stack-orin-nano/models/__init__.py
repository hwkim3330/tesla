"""
FSD Lite Models for Jetson Orin Nano

Orin Nano에서 실시간 동작하는 경량 자율주행 모델 컬렉션.
"""

from .mobilenet_encoder import (
    MobileNetV3Small,
    CameraEncoder,
    HardSwish,
    HardSigmoid,
    SqueezeExcite,
    InvertedResidual,
)

from .fusion_concat import (
    ConcatFusion,
    SpatialConcatFusion,
    MultiModalFusion,
)

from .gru_temporal import (
    GRUTemporal,
    TemporalFusion,
    ConvGRU,
)

from .fsd_lite import (
    FSDLite,
    FSDLiteOutput,
    FSDLiteTRT,
    PointPillarLite,
    DecisionHead,
)

__all__ = [
    # Encoder
    'MobileNetV3Small',
    'CameraEncoder',
    'HardSwish',
    'HardSigmoid',
    'SqueezeExcite',
    'InvertedResidual',
    # Fusion
    'ConcatFusion',
    'SpatialConcatFusion',
    'MultiModalFusion',
    # Temporal
    'GRUTemporal',
    'TemporalFusion',
    'ConvGRU',
    # FSD Lite
    'FSDLite',
    'FSDLiteOutput',
    'FSDLiteTRT',
    'PointPillarLite',
    'DecisionHead',
]

__version__ = '1.0.0'
