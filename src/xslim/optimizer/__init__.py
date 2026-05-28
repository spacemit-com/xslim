#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from .calibration import RuntimeBlockWiseCalibrationPass
from .equalization import XSlimLayerwiseEqualizationPass
from .fusion import (
    ComputingFusionPass,
    FlattenGemmFusionPass,
    FormatBatchNormalizationPass,
    HardSwishFusionPass,
    SwishFusionPass,
)
from .legalized import GraphLegalized
from .observer import TorchXSlimKLObserver, TorchXSlimMSEObserver, TorchXSlimObserver
from .refine import (
    ActivationClipRefine,
    AsymmetricaUnsignlAlignSign,
    PassiveParameterBakingPass,
    QuantizeConfigRefinePass,
    QuantizeFusionPass,
)
from .training import LearnedStepSizePassDecorator, LSQDelegatorDecorator
