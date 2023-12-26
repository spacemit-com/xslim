#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from .fusion import HardSwishFusionPass, SwishFusionPass, ComputingFusionPass
from .refine import (
    BiasParameterBakingPass,
    AsymmetricaUnsignlAlignSign,
    QuantizeFusionPass,
    ActivationClipRefine,
    PassiveParameterBakingPass,
    QuantizeConfigRefinePass,
)
from .calibration import RuntimeBlockWiseCalibrationPass
from .observer import TorchXQuantObserver, TorchXQuantKLObserver, TorchXQuantMSEObserver
from .training import CustomTrainingBasedPass
from .legalized import GraphLegalized
from .equalization import CustomLayerwiseEqualizationPass
