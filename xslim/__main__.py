#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import argparse

from xslim.logger import logger

from . import quantize_onnx_model

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", "--config", required=False, default=None, help="Path to the Xquant Config.")
parser.add_argument("-i", "--input_path", required=False, default=None, help="Path to the Input ONNX Model.")
parser.add_argument("-o", "--output_path", required=False, default=None, help="Path to the Output ONNX Model.")
parser.add_argument("--fp16", required=False, action="store_true", help="convert onnx model to fp16.")
parser.add_argument("--dynq", required=False, action="store_true", help="convert onnx model to dynq.")
parser.add_argument("--ignore_op_types", required=False, default="", help="Ignore op types.")
parser.add_argument("--ignore_op_names", required=False, default="", help="Ignore op names.")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.config is None and (args.input_path is None or args.output_path is None):
        parser.print_help()
        exit(1)

    if args.config is None:
        precesion_level = 100

        if args.fp16:
            precesion_level = 4
        elif args.dynq:
            precesion_level = 3

        if precesion_level == 3:
            logger.info("No config provided, using default config, dynamic quantization...")
        elif precesion_level == 4:
            logger.info("No config provided, using default config, convert onnx model to fp16...")
        elif precesion_level >= 100:
            logger.info("No config provided, using default config, only simplify onnx model...")

        args.config = {
            "quantization_parameters": {
                "precision_level": precesion_level,
                "ignore_op_types": [i for i in args.ignore_op_types.split(",") if i != ""],
                "ignore_op_names": [i for i in args.ignore_op_names.split(",") if i != ""],
            },
        }

    quantize_onnx_model(args.config, args.input_path, args.output_path)
