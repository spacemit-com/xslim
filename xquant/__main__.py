#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import argparse

from xquant.logger import logger

from . import quantize_onnx_model

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", "--config", required=False, default=None, help="Path to the Xquant Config.")
parser.add_argument("-i", "--input_path", required=False, default=None, help="Path to the Input ONNX Model.")
parser.add_argument("-o", "--output_path", required=False, default=None, help="Path to the Output ONNX Model.")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.config is None and (args.input_path is None or args.output_path is None):
        parser.print_help()
        exit(1)

    if args.config is None:
        logger.info("No config provided, using default config, dynamic quantization...")
        args.config = {
            "quantization_parameters": {
                "precision_level": 3,
            },
        }

    quantize_onnx_model(args.config, args.input_path, args.output_path)
