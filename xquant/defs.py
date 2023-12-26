#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
from enum import Enum


class XQUANT_GLOBAL_CONFIGURATION:
    def __init__(self) -> None:
        import torch

        self.cuda_support = torch.cuda.is_available()

        self.default_block_size = 8

        self.fine_tune_epoch = 2

        self.version = "1.0.0"


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


def xquant_warning(info: str):
    print(f"\033[33m[Warning] {info}\033[0m")


def xquant_info(info: str):
    print(f"\033[34m[Info] {info}\033[0m")
