#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import argparse
from . import quantize_onnx_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to the Xquant Config.")

if __name__ == "__main__":
    args = parser.parse_args()
    quantize_onnx_model(args.config)
