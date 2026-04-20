#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import logging
import os
from enum import Enum
from typing import Optional

import onnx

from .logger import (xslim_debug, xslim_error, xslim_info, xslim_trace,
                     xslim_warning)


def _get_version():
    # Prefer the version from the local source tree (VERSION_NUMBER) when available.
    try:
        version_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'VERSION_NUMBER'
        )
        with open(version_file, encoding='utf-8') as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        pass
    # Fall back to the installed package metadata when VERSION_NUMBER is not present.
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version('xslim')
    except (ImportError, PackageNotFoundError):
        return "unknown"


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

        self.version = _get_version()


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

MIN_ONNX_OPSET_VERSION = 24


def is_ai_onnx_operator_supported(
    op_type: str, opset_version: int = MIN_ONNX_OPSET_VERSION
) -> bool:
    """Return whether ai.onnx defines the operator at or before the target opset."""
    try:
        onnx.defs.get_schema(
            op_type,
            max_inclusive_version=opset_version,
            domain="",
        )
        return True
    except onnx.defs.SchemaError:
        return False


def resolve_operator_domain(
    op_type: str,
    opset_version: int = MIN_ONNX_OPSET_VERSION,
    fallback_domain: str = "com.microsoft",
) -> Optional[str]:
    """Return None for standard ai.onnx operators, else the required custom domain."""
    if is_ai_onnx_operator_supported(op_type, opset_version):
        return None
    return fallback_domain

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
