#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from enum import Enum
import logging
from .logger import xquant_info, xquant_warning, xquant_debug, xquant_error, xquant_trace


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

        self.max_bias_val = 2**22

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

BIAS_CORRECTION_INTERST_TYPE = {
    "Conv",
    "Gemm",
    "ConvTranspose",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
}

OBSERVER_FLOATING_MSE_FETCHES = 4096

OBSERVER_MIN_SCALE_THRESHOLD = 2**-22

OBSERVER_MAX_SCALE_THRESHOLD = 2**8

OBSERVER_SIGMOID_MAX_VALUE = 10


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
