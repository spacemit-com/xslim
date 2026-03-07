# Configuration Reference

[中文版](configuration_zh.md) | English

This page documents every field in the XSlim JSON configuration file. All fields are optional unless marked **required**.

## Top-level Structure

```json
{
    "model_parameters": { ... },
    "calibration_parameters": { ... },
    "quantization_parameters": { ... }
}
```

---

## `model_parameters`

Controls input/output paths and pre-processing of the ONNX model.

| Field | Type | Default | Description |
|---|---|---|---|
| `onnx_model` | `string` | — | Path to the input ONNX model file |
| `output_prefix` | `string` | Model filename (output ends with `.q.onnx`) | Prefix for the output file name |
| `working_dir` | `string` | Directory of `onnx_model` | Directory where the output model and intermediate files are written |
| `skip_onnxsim` | `bool` | `false` | Set to `true` to skip ONNX simplification before quantization |

### Example

```json
"model_parameters": {
    "onnx_model": "models/resnet18.onnx",
    "output_prefix": "resnet18_int8",
    "working_dir": "./output",
    "skip_onnxsim": false
}
```

---

## `calibration_parameters`

Controls how calibration data is loaded and used to compute quantization ranges.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `calibration_step` | `int` | `100` | — | Maximum number of calibration samples. Recommended range: 100–1000 |
| `calibration_device` | `string` | `cuda` | `cuda`, `cpu` | Device for calibration inference. Auto-detected; falls back to `cpu` if no GPU is available |
| `calibration_type` | `string` | `default` | `default`, `kl`, `minmax`, `percentile`, `mse` | Observer algorithm used to compute activation ranges |
| `input_parametres` | `list` | — | — | Per-input calibration settings. One entry per model input (see below) |

### Calibration Type Details

| Value | Description |
|---|---|
| `default` | Uses the chip-recommended algorithm (usually `kl` or `percentile`) |
| `kl` | KL-divergence minimization |
| `minmax` | Uses the observed minimum and maximum values |
| `percentile` | Clips the range at `max_percentile` to reduce outlier influence |
| `mse` | Minimizes mean squared error between original and quantized activations |

### `input_parametres` (per input)

Each entry in the list corresponds to one model input, in the same order as the ONNX model's inputs.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `input_name` | `string` | Read from ONNX model | — | Name of the input tensor |
| `input_shape` | `list[int]` | Read from ONNX model | — | Shape of the input. Symbolic batch dimensions default to `1` |
| `dtype` | `string` | Read from ONNX model | `float32`, `int8`, `uint8`, `int16` | Data type expected by the model input |
| `file_type` | `string` | `img` | `img`, `npy`, `raw` | Format of the calibration files |
| `color_format` | `string` | `bgr` | `rgb`, `bgr` | Color channel order for image inputs |
| `mean_value` | `list[float]` | `null` | — | Per-channel mean subtracted during normalization |
| `std_value` | `list[float]` | `null` | — | Per-channel standard deviation used to scale during normalization |
| `preprocess_file` | `string` | `null` | `PT_IMAGENET`, `IMAGENET`, or a custom path | Preprocessing function (see [Custom Preprocessing](#custom-preprocessing)) |
| `data_list_path` | `string` | **required** | — | Path to a text file listing calibration data files, one per line |

#### `file_type` Details

| Value | Description |
|---|---|
| `img` | Standard image file (JPEG, PNG, BMP, etc.) read with OpenCV |
| `npy` | NumPy `.npy` file containing a single array |
| `raw` | Raw binary file read directly as bytes |

### Calibration Data List File

`data_list_path` points to a plain-text file with one calibration file path per line. Paths may be absolute or relative to the directory containing the list file.

```text
data/calib/image_001.JPEG
data/calib/image_002.JPEG
data/calib/image_003.JPEG
```

For multi-input models, each `input_parametres` entry has its own list file. The file at position *N* across all lists forms one calibration batch, so the lists must have the same length and consistent ordering.

### Custom Preprocessing

Set `preprocess_file` to one of:

- `"PT_IMAGENET"` – Built-in PyTorch-style ImageNet preprocessing (resize → center-crop → normalize)
- `"IMAGENET"` – Built-in standard ImageNet preprocessing
- `"path/to/script.py:function_name"` – A function in a user-provided Python file

Custom preprocessing functions must have the following signature:

```python
from typing import Sequence
import torch

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Args:
        path_list: List of file paths forming one calibration batch.
        input_parametr: The corresponding entry from calibration_parameters.input_parametres.

    Returns:
        A batched torch.Tensor of shape [batch, ...].
    """
    ...
```

Full example:

```python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
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

---

## `quantization_parameters`

Controls the quantization strategy and precision.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `precision_level` | `int` | `0` | `0`–`4` | Global precision level (see below) |
| `finetune_level` | `int` | `1` | `0`–`3` | Aggressiveness of calibration parameter tuning (see below) |
| `analysis_enable` | `bool` | `true` | — | Run post-quantization accuracy analysis |
| `max_percentile` | `float` | `0.9999` | ≥ `0.99` | Percentile clipping threshold when `calibration_type` is `percentile` |
| `custom_setting` | `list` | `null` | — | Per-subgraph precision overrides (see below) |
| `truncate_var_names` | `list[string]` | `[]` | — | Tensor names used to split the graph into separate subgraphs |
| `ignore_op_types` | `list[string]` | `[]` | — | ONNX operator types to exclude from quantization |
| `ignore_op_names` | `list[string]` | `[]` | — | Specific operator names to exclude from quantization |

### Precision Levels

| Level | Description |
|---|---|
| `0` | Full INT8 quantization (default). Best compression, may impact accuracy |
| `1` | Partial INT8 – sensitive layers kept at higher precision. Suitable for general Transformer models |
| `2` | Partial INT8 with the highest preserved precision. Use for accuracy-sensitive models |
| `3` | Dynamic quantization – weights are quantized, activations are quantized at runtime |
| `4` | FP16 conversion – all floating-point ops are cast to FP16 (no calibration data required) |

### Fine-tune Levels

| Level | Description |
|---|---|
| `0` | No calibration parameter adjustment |
| `1` | May apply lightweight static calibration parameter tuning |
| `2` | Block-wise calibration parameter tuning based on quantization loss |
| `3` | More aggressive block-wise tuning; higher quality but slower |

### `custom_setting`

A list of per-subgraph precision overrides. Each entry selects a subgraph by its boundary tensors and applies a local `precision_level`.

| Field | Type | Description |
|---|---|---|
| `input_names` | `list[string]` | Tensor names that mark the start of the subgraph |
| `output_names` | `list[string]` | Tensor names that mark the end of the subgraph |
| `precision_level` | `int` | Precision level to apply to this subgraph (same values as the global `precision_level`) |

```json
"custom_setting": [
    {
        "input_names": ["input"],
        "output_names": ["input.12"],
        "precision_level": 2
    }
]
```

### `truncate_var_names`

A list of tensor names where the graph is cut. Each tensor becomes both an output of the preceding subgraph and an input of the following subgraph. This is useful when the beginning or end of the model should be handled differently.

```json
"truncate_var_names": ["features.0.0.weight", "classifier.1.weight"]
```

### `ignore_op_types` and `ignore_op_names`

Operators that match any entry in `ignore_op_types` (by ONNX op type) or `ignore_op_names` (by node name) are excluded from quantization and kept at their original precision.

```json
"ignore_op_types": ["LayerNormalization", "Softmax"],
"ignore_op_names": ["/model/encoder/layer.0/attention/MatMul"]
```

---

## Complete Configuration Example

```json
{
    "model_parameters": {
        "onnx_model": "models/my_model.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "calibration_step": 200,
        "calibration_device": "cuda",
        "calibration_type": "kl",
        "input_parametres": [
            {
                "mean_value": [123.675, 116.28, 103.53],
                "std_value": [58.395, 57.12, 57.375],
                "color_format": "rgb",
                "preprocess_file": "PT_IMAGENET",
                "data_list_path": "./calib_data/img_list.txt"
            }
        ]
    },
    "quantization_parameters": {
        "precision_level": 1,
        "finetune_level": 2,
        "analysis_enable": true,
        "max_percentile": 0.9999,
        "ignore_op_types": ["Softmax"],
        "custom_setting": [
            {
                "input_names": ["input"],
                "output_names": ["stem_output"],
                "precision_level": 0
            }
        ]
    }
}
```
