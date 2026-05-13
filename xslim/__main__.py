#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xslim", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--config", required=False, default=None, help="Path to the Xquant Config.")
    parser.add_argument("-i", "--input_path", required=False, default=None, help="Path to the Input ONNX Model.")
    parser.add_argument("-o", "--output_path", required=False, default=None, help="Path to the Output ONNX Model.")
    parser.add_argument("--fp16", required=False, action="store_true", help="convert onnx model to fp16.")
    parser.add_argument("--dynq", required=False, action="store_true", help="convert onnx model to dynq.")
    parser.add_argument("--ignore_op_types", required=False, default="", help="Ignore op types.")
    parser.add_argument("--ignore_op_names", required=False, default="", help="Ignore op names.")
    parser.add_argument(
        "--opset",
        required=False,
        type=int,
        default=None,
        help="Convert the default ai.onnx opset to the target version.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config is None and (args.input_path is None or args.output_path is None):
        parser.print_help()
        return 1

    if args.config is None:
        from .logger import logger

        precision_level = 100
        if args.fp16:
            precision_level = 4
        elif args.dynq:
            precision_level = 3

        if precision_level == 3:
            logger.info("No config provided, using default config, dynamic quantization...")
        elif precision_level == 4:
            logger.info("No config provided, using default config, convert onnx model to fp16...")
        elif precision_level >= 100:
            logger.info("No config provided, using default config, only simplify onnx model...")

        args.config = {
            "quantization_parameters": {
                "precision_level": precision_level,
                "ignore_op_types": [item for item in args.ignore_op_types.split(",") if item != ""],
                "ignore_op_names": [item for item in args.ignore_op_names.split(",") if item != ""],
            },
        }
        if args.opset is not None:
            args.config["model_parameters"] = {"opset": args.opset}
    elif args.opset is not None:
        with open(args.config, "r", encoding="utf-8") as fp:
            args.config = json.load(fp)
        args.config.setdefault("model_parameters", {})["opset"] = args.opset

    from . import quantize_onnx_model

    quantize_onnx_model(args.config, args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
