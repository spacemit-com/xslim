# Configuration Reference

[中文版](configuration_zh.md) | English

XSlim is configured through a JSON file with three top-level sections. All fields are **optional** unless marked otherwise.

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

Controls input/output paths and model pre-processing.

| Field | Type | Default | Description |
|---|---|---|---|
| `onnx_model` | `string` | — | Path to the input ONNX model file |
| `output_prefix` | `string` | Model filename stem (output appends `.q.onnx`) | Prefix for the output file name |
| `working_dir` | `string` | Directory containing `onnx_model` | Directory for the output model and intermediate files |
| `skip_onnxsim` | `bool` | `false` | Set `true` to skip ONNX model simplification before quantization |

### Example

```json
"model_parameters": {
    "onnx_model": "models/resnet18.onnx",
    "output_prefix": "resnet18_int8",
    "working_dir": "./output"
}
```

---

## `calibration_parameters`

Controls how calibration data is loaded and how quantization ranges are computed.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `calibration_step` | `int` | `500` | 10–1000 | Maximum number of calibration steps (dataloader iterations/batches); effective sample count ≈ `calibration_step × batch_size` |
| `calibration_device` | `string` | `cuda` | `cuda`, `cpu` | Inference device for calibration; auto-detected, falls back to `cpu` |
| `calibration_type` | `string` | `default` | `default`, `kl`, `minmax`, `percentile`, `mse` | Observer algorithm for computing activation ranges |
| `input_parameters` | `list` | **required** | — | Per-input settings, one entry per model input (see below) |

### Calibration Type Details

| Value | Description |
|---|---|
| `default` | Chip-recommended algorithm (typically `kl` or `percentile`) |
| `kl` | KL-divergence minimization |
| `minmax` | Uses the observed minimum and maximum values |
| `percentile` | Clips activations at `max_percentile` to suppress outliers |
| `mse` | Minimizes mean squared error between original and quantized activations |

> **Tip:** Start with `default`. If accuracy is insufficient, try `percentile` or `minmax`.

### `input_parameters` (per input)

Each list entry corresponds to one model input **in the same order as the ONNX model's input list**.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `input_name` | `string` | Read from model | — | Input tensor name |
| `input_shape` | `list[int]` | Read from model | — | Input shape; symbolic batch dimension defaults to `1` |
| `dtype` | `string` | Read from model | Any ONNX tensor dtype | Input data type; by default read from the ONNX model (non-float types such as `int64` are supported; `img`/`raw` loaders interpret data as `float32`) |
| `file_type` | `string` | `img` | `img`, `npy`, `raw` | Calibration file format (see below) |
| `color_format` | `string` | `bgr` | `rgb`, `bgr` | Color channel order for image inputs |
| `mean_value` | `list[float]` | `null` | — | Per-channel mean subtracted during normalization |
| `std_value` | `list[float]` | `null` | — | Per-channel standard deviation used to divide during normalization |
| `preprocess_file` | `string` | `null` | `PT_IMAGENET`, `IMAGENET`, or a custom path | Preprocessing function (see [Custom Preprocessing](#custom-preprocessing)) |
| `data_list_path` | `string` | **required** | — | Path to a text file listing calibration data files, one path per line |

#### `file_type` Details

| Value | Description |
|---|---|
| `img` | Standard image file (JPEG, PNG, BMP, etc.) read with OpenCV |
| `npy` | NumPy `.npy` file containing a single array |
| `raw` | Raw binary file; must contain `float32` data matching `input_shape` |

### Calibration Data List File

`data_list_path` points to a plain-text file with **one file path per line**. Paths may be absolute or relative to the directory containing the list file.

```text
data/calib/ILSVRC2012_val_00002138.JPEG
data/calib/ILSVRC2012_val_00000994.JPEG
data/calib/ILSVRC2012_val_00014467.JPEG
```

For multi-input models, each `input_parameters` entry has its own list file. Files at the same line number across all lists form one calibration batch — the lists must therefore have equal length and consistent ordering.

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
        path_list: List of file paths for one calibration batch.
        input_parametr: The corresponding entry from calibration_parameters.input_parameters.

    Returns:
        A batched torch.Tensor of shape [batch, C, H, W].
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
        img = cv2.resize(img, (input_shape[-1], input_shape[-2]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = (img - mean_value) / std_value
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        batch_list.append(img)
    return torch.cat(batch_list, dim=0)
```

For multi-input models where preprocessing is similar across inputs, the same function can be reused for each entry.

---

## `quantization_parameters`

Controls the quantization strategy and precision. All fields in this section are optional.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `precision_level` | `int` | `0` | `0`–`4` | Global precision level (see below) |
| `finetune_level` | `int` | `1` | `0`–`3` | Aggressiveness of calibration parameter tuning (see below) |
| `analysis_enable` | `bool` | `true` | — | Run post-quantization accuracy analysis |
| `max_percentile` | `float` | `0.9999` | ≥ `0.99` | Percentile clipping threshold (applies when `calibration_type` is `percentile`) |
| `custom_setting` | `list` | `null` | — | Per-subgraph precision overrides (see below) |
| `truncate_var_names` | `list[string]` | `[]` | — | Tensor names used to split the graph into two parts (see below) |
| `ignore_op_types` | `list[string]` | `[]` | — | ONNX operator types to exclude from quantization |
| `ignore_op_names` | `list[string]` | `[]` | — | Specific operator names to exclude from quantization |

### Precision Levels

| Level | Description |
|---|---|
| `0` | Full INT8 quantization (default). Maximum compression; quantization tuning stays within INT8 |
| `1` | Partial INT8 – some sensitive operators are kept at higher precision. Suitable for general Transformer models |
| `2` | Partial INT8 with the most operators kept at higher precision. Best for accuracy-sensitive models |
| `3` | Dynamic quantization – weights are statically quantized; activations are quantized at runtime |
| `4` | FP16 conversion – all floating-point ops are cast to FP16 (no calibration data required) |

### Fine-tune Levels

| Level | Description |
|---|---|
| `0` | No calibration parameter adjustment |
| `1` | May apply lightweight static calibration parameter tuning |
| `2` | Block-wise calibration parameter tuning based on quantization loss |
| `3` | More aggressive block-wise tuning; higher quality but slower |

### `custom_setting`

A list of per-subgraph overrides. Each entry selects a contiguous subgraph by its boundary tensors and applies local quantization settings. All tensors between the boundary input operators and the boundary output operators (inclusive) are covered by the override.

| Field | Type | Description |
|---|---|---|
| `input_names` | `list[string]` | Input tensor names that mark the entry edge of the subgraph (constants may be omitted) |
| `output_names` | `list[string]` | Output tensor names that mark the exit edge of the subgraph |
| `precision_level` | `int` | Precision level to apply to this subgraph |
| `calibration_type` | `string` | Calibration algorithm for this subgraph (same values as the global `calibration_type`) |
| `max_percentile` | `float` | Percentile threshold for this subgraph (same semantics as the global `max_percentile`) |

```json
"custom_setting": [
    {
        "input_names": ["input"],
        "output_names": ["input.12"],
        "precision_level": 2,
        "calibration_type": "default"
    }
]
```

### `truncate_var_names`

Specifies a list of tensor names at which the computation graph is **split into exactly two parts**. The split tensors become output tensors of the first subgraph and input tensors of the second. The tool validates the binary split and raises an error if the result is invalid.

```json
"truncate_var_names": ["/Concat_5_output_0", "/Transpose_6_output_0"]
```

### `ignore_op_types` and `ignore_op_names`

Operators matching any entry in `ignore_op_types` (by ONNX op type) or `ignore_op_names` (by node name) are excluded from quantization and kept at their original precision.

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
        "calibration_type": "default",
        "input_parameters": [
            {
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57.0, 57.0, 57.0],
                "color_format": "bgr",
                "preprocess_file": "PT_IMAGENET",
                "data_list_path": "./calib_data/img_list.txt"
            }
        ]
    },
    "quantization_parameters": {
        "precision_level": 1,
        "finetune_level": 2,
        "analysis_enable": true,
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
