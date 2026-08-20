# XSlim

[中文版](README_zh.md) | English

[![Version](https://img.shields.io/badge/version-2.1.1-blue.svg)](https://github.com/spacemit-com/xslim/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/)

**XSlim** is a Post-Training Quantization (PTQ) tool developed by [SpacemiT](https://www.spacemit.com). It integrates chip-optimized quantization strategies and provides a unified interface for ONNX model quantization via JSON configuration files.

---

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Samples](#samples)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

## Features

- **INT8 / FP16 / Dynamic Quantization** – multiple precision levels for different deployment scenarios
- **JSON-driven configuration** – simple, declarative quantization setup
- **Python API & CLI** – use as a library or from the command line
- **Custom preprocessing** – plug in your own preprocessing functions
- **Expanded ONNX operator coverage** – run Graphwise Analysis and quantization on models that use common arithmetic, activation, comparison, reduction, dropout, and opset-24 `Pad` patterns
- **Automatic YOLO decode fusion** – fuse supported YOLO decode subgraphs into a single `spacemit_functions.YoloDecode` node
- **ONNX Function-aware export** – preserve embedded FunctionProto definitions and emit required custom-domain imports automatically
- **ONNX-based workflow** – built on the ONNX ecosystem

## Installation

```bash
python -m pip install xslim
```

Or install from source:

```bash
git clone https://github.com/spacemit-com/xslim.git
cd xslim
python -m pip install .
```

For local development, use an editable install:

```bash
python -m pip install -e .
```

Build metadata is defined in `pyproject.toml`, and the import package lives under the standard `src/` layout. To build source and wheel distributions locally:

```bash
python -m pip install --upgrade build
python -m build
```

## Quick Start

### Python API

```python
import xslim

# Using a JSON config file
xslim.quantize_onnx_model("config.json")

# Using a dict
config = {
    "model_parameters": {
        "onnx_model": "model.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "input_parameters": [{
            "mean_value": [123.675, 116.28, 103.53],
            "std_value": [58.395, 57.12, 57.375],
            "color_format": "rgb",
            "preprocess_file": "PT_IMAGENET",
            "data_list_path": "./calib_img_list.txt"
        }]
    }
}
xslim.quantize_onnx_model(config)

# You can also pass the model path and output path directly
xslim.quantize_onnx_model("config.json", "input.onnx", "output.onnx")
```

### Command Line

```bash
# Installed CLI entry point
xslim --config config.json

# Module entry point also remains available
python -m xslim --config config.json

# Specify input and output model paths
xslim -c config.json -i input.onnx -o output.onnx

# Dynamic quantization (no config file needed)
xslim -i input.onnx -o output.onnx --dynq

# FP16 conversion (no config file needed)
xslim -i input.onnx -o output.onnx --fp16

# Convert the default ai.onnx opset to a target version
xslim -i input.onnx -o output.onnx --opset 20

# ONNX simplification only (no config file needed)
xslim -i input.onnx -o output.onnx
```

For config-free dynamic quantization and FP16 conversion, you can exclude operators with comma-separated names or types:

```bash
xslim -i input.onnx -o output.onnx --dynq --ignore_op_types Softmax,LayerNormalization
xslim -i input.onnx -o output.onnx --fp16 --ignore_op_names /model/head/MatMul
```

Static INT8 quantization expects a floating-point input model. If the model already contains `QuantizeLinear` or `DequantizeLinear`, XSlim stops with a clear error instead of quantizing an already-quantized graph again.

For supported YOLO exports, no extra switch is required: XSlim will try to fuse decode-heavy post-processing into `spacemit_functions.YoloDecode` during simplification and keep the corresponding ONNX `FunctionProto` in the exported model.

## Documentation

- [Configuration Reference](doc/configuration.md) – Full description of all JSON configuration options
- [Examples](doc/examples.md) – Step-by-step guides for INT8, FP16, dynamic quantization, custom preprocessing, and more
- [Accuracy Tuning Guide](doc/accuracy_tuning.md) – How to diagnose and improve quantization accuracy

## Samples

See the [samples](samples/) directory for ready-to-run examples covering ResNet-18, MobileNet V3, BERT, and more. YOLO-specific usage notes are documented in the examples and accuracy-tuning guides.

## Changelog

For a full list of published versions, see the [Releases](https://github.com/spacemit-com/xslim/releases) page.

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/spacemit-com/xslim/issues) or submit a [pull request](https://github.com/spacemit-com/xslim/pulls).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
