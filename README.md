# XSlim

[中文版](README_zh.md) | English

XSlim is a Post-Training Quantization (PTQ) tool developed by [SpacemiT](https://www.spacemit.com). It integrates chip-optimized quantization strategies and provides a unified interface for model quantization via JSON configuration files.

## Features

- **INT8 / FP16 / Dynamic Quantization** – multiple precision levels for different deployment scenarios
- **JSON-driven configuration** – simple, declarative quantization setup
- **Python API & CLI** – use as a library or from the command line
- **Custom preprocessing** – plug in your own preprocessing functions
- **ONNX-based workflow** – built on the ONNX ecosystem

## Installation

```bash
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
        "input_parametres": [{
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
python -m xslim --config config.json

# FP16 conversion
python -m xslim -i input.onnx -o output.onnx --fp16

# Dynamic quantization
python -m xslim -i input.onnx -o output.onnx --dynq
```

## Configuration

Quantization is configured through a JSON file with three main sections:

| Section | Description |
|---|---|
| `model_parameters` | Input/output model paths and working directory |
| `calibration_parameters` | Calibration dataset, preprocessing, and observer settings |
| `quantization_parameters` | Precision level, fine-tune level, and per-subgraph overrides |

### Precision Levels

| Level | Description |
|---|---|
| 0 | INT8 quantization (default) |
| 1 | INT8 quantization with higher precision |
| 2 | INT8 quantization with highest precision |
| 3 | Dynamic quantization |
| 4 | FP16 conversion |
| 100 | ONNX simplification only (no quantization) |

For a complete configuration reference, see the [XSlim documentation](https://github.com/spacemit-com/docs-ai/blob/main/zh/compute_stack/ai_compute_stack/xslim.md).

## Samples

See the [samples](samples/) directory for ready-to-run examples covering ResNet-18, MobileNet V3, BERT, and more.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
