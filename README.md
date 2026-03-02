# XSlim

[中文版](README_zh.md) | English

[![Version](https://img.shields.io/badge/version-2.0.8-blue.svg)](https://github.com/spacemit-com/xslim/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue.svg)](https://www.python.org/)

**XSlim** is a Post-Training Quantization (PTQ) tool developed by [SpacemiT](https://www.spacemit.com). It integrates chip-optimized quantization strategies and provides a unified interface for ONNX model quantization via JSON configuration files.

---

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
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
# INT8 quantization with a JSON config
python -m xslim --config config.json

# Specify input and output model paths
python -m xslim -c config.json -i input.onnx -o output.onnx

# Dynamic quantization (no config file needed)
python -m xslim -i input.onnx -o output.onnx --dynq

# FP16 conversion (no config file needed)
python -m xslim -i input.onnx -o output.onnx --fp16

# ONNX simplification only (no config file needed)
python -m xslim -i input.onnx -o output.onnx
```

## Configuration Reference

Quantization is configured through a JSON file with three main sections: `model_parameters`, `calibration_parameters`, and `quantization_parameters`. All fields below are optional unless noted otherwise.

### `model_parameters`

| Field | Default | Description |
|---|---|---|
| `onnx_model` | — | Path to the input ONNX model |
| `output_prefix` | Model filename (output ends with `.q.onnx`) | Output file prefix |
| `working_dir` | Directory of `onnx_model` | Output and working directory |
| `skip_onnxsim` | `false` | Skip ONNX simplification |

### `calibration_parameters`

| Field | Default | Options | Description |
|---|---|---|---|
| `calibration_step` | `100` | — | Max number of calibration samples (recommended 100–1000) |
| `calibration_device` | `cuda` | `cuda`, `cpu` | Auto-detected; falls back to `cpu` |
| `calibration_type` | `default` | `default`, `kl`, `minmax`, `percentile`, `mse` | Calibration observer algorithm |
| `input_parametres` | — | — | List of per-input settings (see below) |

#### `input_parametres` (per input)

| Field | Default | Options | Description |
|---|---|---|---|
| `input_name` | Read from ONNX model | — | Input tensor name |
| `input_shape` | Read from ONNX model | — | Input shape (symbolic batch dim defaults to 1) |
| `dtype` | Read from ONNX model | `float32`, `int8`, `uint8`, `int16` | Data type |
| `file_type` | `img` | `img`, `npy`, `raw` | Calibration file type |
| `color_format` | `bgr` | `rgb`, `bgr` | Image color format |
| `mean_value` | `None` | — | Per-channel mean for normalization |
| `std_value` | `None` | — | Per-channel std for normalization |
| `preprocess_file` | `None` | `PT_IMAGENET`, `IMAGENET`, or custom path | Preprocessing function (see below) |
| `data_list_path` | **required** | — | Path to calibration file list |

### `quantization_parameters`

| Field | Default | Options | Description |
|---|---|---|---|
| `precision_level` | `0` | `0`, `1`, `2`, `3`, `4` | See precision levels below |
| `finetune_level` | `1` | `0`, `1`, `2`, `3` | See fine-tune levels below |
| `analysis_enable` | `true` | — | Enable post-quantization analysis |
| `max_percentile` | `0.9999` | ≥ `0.99` | Percentile clipping range |
| `custom_setting` | `None` | — | Per-subgraph overrides (list) |
| `truncate_var_names` | `[]` | — | Tensor names to split the graph |
| `ignore_op_types` | `[]` | — | Op types to skip during quantization |
| `ignore_op_names` | `[]` | — | Op names to skip during quantization |

#### Precision Levels

| Level | Description |
|---|---|
| 0 | Full INT8 quantization (default) |
| 1 | Partial INT8, suitable for general Transformer models |
| 2 | Partial INT8 with highest precision |
| 3 | Dynamic quantization |
| 4 | FP16 conversion |

#### Fine-tune Levels

| Level | Description |
|---|---|
| 0 | No calibration parameter tuning |
| 1 | May apply static calibration parameter tuning |
| 2+ | Block-wise calibration parameter tuning based on quantization loss |

### Calibration Data List

The `data_list_path` file should list one calibration file per line. Paths can be absolute or relative to the directory containing the list file. For multi-input models, ensure the file order is consistent across inputs.

```text
data/calib/image_001.JPEG
data/calib/image_002.JPEG
data/calib/image_003.JPEG
```

### Custom Preprocessing

Set `preprocess_file` to `"path/to/script.py:function_name"` to use a custom preprocessing function:

```python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Read files from path_list, preprocess using input_parametr, and return a torch.Tensor.

    Args:
        path_list: List of file paths for one calibration batch.
        input_parametr: Equivalent to calibration_parameters.input_parametres[idx].

    Returns:
        A batched torch.Tensor of calibration data.
    """
    batch_list = []
    mean_value = input_parametr["mean_value"]
    std_value = input_parametr["std_value"]
    input_shape = input_parametr["input_shape"]
    for file_path in path_list:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (input_shape[-1], input_shape[-2]))
        img = img.astype(np.float32)
        img = (img - mean_value) / std_value
        img = np.transpose(img, (2, 0, 1))
        img = torch.unsqueeze(torch.from_numpy(img), 0)
        batch_list.append(img)
    return torch.cat(batch_list, dim=0)
```

## Samples

See the [samples](samples/) directory for ready-to-run examples covering ResNet-18, MobileNet V3, BERT, and more.

## Changelog

For a full list of changes, see the [Releases](https://github.com/spacemit-com/xslim/releases) page.

| Version | Highlights |
|---|---|
| 2.0.8 | Latest development version |
| [2.0.7](https://github.com/spacemit-com/xslim/releases/tag/2.0.7) | Fix FP16 conversion bug on complex models |
| [2.0.6](https://github.com/spacemit-com/xslim/releases/tag/2.0.6) | Fix metadata props deletion; default CLI behavior changed to model simplification (use `--dynq` for dynamic quantization) |

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/spacemit-com/xslim/issues) or submit a [pull request](https://github.com/spacemit-com/xslim/pulls).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
