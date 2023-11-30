from .fusion import HardSwishFusionPass, SwishFusionPass, ComputingFusionPass
from .refine import (
    BiasParameterBakingPass,
    AsymmetricaUnsignlAlignSign,
    QuantizeFusionPass,
    ActivationClipRefine,
    PassiveParameterBakingPass,
)
from .calibration import RuntimeBlockWiseCalibrationPass
from .observer import TorchXQuantObserver
from .training import CustomTrainingBasedPass
from .legalized import GraphLegalized
from .equalization import CustomLayerwiseEqualizationPass
