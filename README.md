# XSlim

[中文版](README_zh.md) | English

[![Version](https://img.shields.io/badge/version-2.0.13-blue.svg)](https://github.com/spacemit-com/xslim/releases)
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
- **ONNX-based workflow** – built on the ONNX ecosystem

## Installation

```bash
pip install xslim
```

Or install from source:

```bash
git clone https://github.com/spacemit-com/xslim.git
cd xslim
pip install -r requirements.txt
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
# INT8 quantization with a JSON config
python -m xslim --config config.json

# Specify input and output model paths
python -m xslim -c config.json -i input.onnx -o output.onnx

# Dynamic quantization (no config file needed)
python -m xslim -i input.onnx -o output.onnx --dynq

# FP16 conversion (no config file needed)
python -m xslim -i input.onnx -o output.onnx --fp16

# Convert the default ai.onnx opset to a target version
python -m xslim -i input.onnx -o output.onnx --opset 20

# ONNX simplification only (no config file needed)
python -m xslim -i input.onnx -o output.onnx
```

## Documentation

- [Configuration Reference](doc/configuration.md) – Full description of all JSON configuration options
- [Examples](doc/examples.md) – Step-by-step guides for INT8, FP16, dynamic quantization, custom preprocessing, and more
- [Accuracy Tuning Guide](doc/accuracy_tuning.md) – How to diagnose and improve quantization accuracy

## Samples

See the [samples](samples/) directory for ready-to-run examples covering ResNet-18, MobileNet V3, BERT, and more.

## Changelog

For a full list of changes, see the [Releases](https://github.com/spacemit-com/xslim/releases) page.

| Version | Highlights |
|---|---|
| 2.0.13 | Current development version |
| [2.0.12](https://github.com/spacemit-com/xslim/releases/tag/2.0.12) | Latest release; complete README changelog/release metadata, add accuracy-tuning docs and README links, introduce the xslim-accuracy-tuning GitHub skill, add YOLO truncation guidance, and rename input parameters for consistency |
| [2.0.11](https://github.com/spacemit-com/xslim/releases/tag/2.0.11) | Fix Pad/missing-input handling, add Or/Einsum/Selu support, normalize Conv/ConvTranspose kernel shapes, and raise minimum Python to 3.9 |
| [2.0.10](https://github.com/spacemit-com/xslim/releases/tag/2.0.10) | Align release metadata, improve CI/test coverage, normalize missing default ONNX opset before dynamic quantization, and refine shape inference handling |
| [2.0.9](https://github.com/spacemit-com/xslim/releases/tag/2.0.9) | Add documentation, preserve tensor dtype metadata during FP16 conversion, and restore compatibility with onnxslim 0.1.87 |
| [2.0.8](https://github.com/spacemit-com/xslim/releases/tag/2.0.8) | Improve packaging/CI, add torch executor operator coverage, add PyPI publish workflow, and centralize version metadata |
| [2.0.7](https://github.com/spacemit-com/xslim/releases/tag/2.0.7) | Fix FP16 conversion bug on complex models |
| [2.0.6](https://github.com/spacemit-com/xslim/releases/tag/2.0.6) | Fix metadata props deletion; default CLI behavior changed to model simplification (use `--dynq` for dynamic quantization) |

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/spacemit-com/xslim/issues) or submit a [pull request](https://github.com/spacemit-com/xslim/pulls).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
