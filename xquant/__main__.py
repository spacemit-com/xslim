#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import argparse
from . import quantize_onnx_model

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", "--config", required=True, help="Path to the Xquant Config.")
parser.add_argument("-i", "--input_path", required=False, default=None, help="Path to the Input ONNX Model.")
parser.add_argument("-o", "--output_path", required=False, default=None, help="Path to the Output ONNX Model.")

if __name__ == "__main__":
    args = parser.parse_args()
    quantize_onnx_model(args.config, args.input_path, args.output_path)
