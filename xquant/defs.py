#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from enum import Enum
import logging
from .logger import xquant_info, xquant_warning, xquant_debug, xquant_error, xquant_trace


class XQUANT_GLOBAL_CONFIGURATION:
    def __init__(self) -> None:
        import torch

        self.cuda_support = torch.cuda.is_available()

        self.default_block_size = 8

        self.fine_tune_epoch = 2

        self.equalization_iterations = 10

        self.max_bits = 12

        self.version = "1.0.3"


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

COMPUTING_OP = {"Conv", "Gemm", "ConvTranspose", "MatMul"}

BIAS_CORRECTION_INTERST_TYPE = {"Conv", "Gemm", "ConvTranspose"}

OBSERVER_FLOATING_MSE_FETCHES = 4096


class AutoFinetuneLevel(Enum):
    DO_NOTHING = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


class PrecisionLevel(Enum):
    BIT_8 = 0
    BIT_8_16 = 1
    GEMM_16 = 2


XQUANT_CONFIG = XQUANT_GLOBAL_CONFIGURATION()
