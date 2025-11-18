#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import logging
from enum import Enum

from .logger import (xslim_debug, xslim_error, xslim_info, xslim_trace,
                     xslim_warning)


class XQUANT_GLOBAL_CONFIGURATION:
    def __init__(self) -> None:
        import torch

        self.cuda_support = torch.cuda.is_available()

        self.min_block_size = 10

        self.max_block_size = 20

        self.merge_block_step = 4

        self.fine_tune_epoch = 2

        self.equalization_iterations = 10

        self.max_bits = 12

        self.analyse_steps = 16

        self.version = "2.0.6"


PASSIVE_OPERATIONS = {
    "MaxPool",
    "GlobalMaxPool",
    "Reshape",
    "Flatten",
    "Identity",
    "Dropout",
    "Slice",
    "Pad",
    "Split",
    "Transpose",
    "Interp",
    "Squeeze",
    "Unsqueeze",
    "Gather",
}

COMPUTING_OP = {"Conv", "Gemm", "ConvTranspose", "MatMul", "BatchMatMul"}

BIAS_CORRECTION_INTERST_TYPE = {
    "Conv",
    "Gemm",
    "ConvTranspose",
    # "LayerNormalization",
    # "InstanceNormalization",
    # "GroupNormalization",
}

OBSERVER_FLOATING_MSE_FETCHES = 4096

OBSERVER_MIN_SCALE_THRESHOLD = 2**-23

OBSERVER_MAX_SCALE_THRESHOLD = 2**8

OBSERVER_MAX_BIAS_VAL = 2**20

OBSERVER_PERCENTILE = 0.9999

OBSERVER_SIGMOID_MAX_VALUE = 10

MIN_ONNX_OPSET_VERSION = 17

GLOBAL_FUNCTIONS_MAPPING = "GLOBAL_FUNCTIONS_MAPPING"


class AutoFinetuneLevel(Enum):
    DO_NOTHING = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


class PrecisionLevel(Enum):
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_None = 100


XQUANT_CONFIG = XQUANT_GLOBAL_CONFIGURATION()
