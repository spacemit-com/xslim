# Examples

[中文版](examples_zh.md) | English

This page provides step-by-step examples for the most common XSlim quantization scenarios. All examples assume you have already installed XSlim (`pip install xslim`) and have an ONNX model available.

Ready-to-run versions of these examples can be found in the [samples](../samples/) directory.

---

## 1. Basic INT8 Quantization (ResNet-18)

The simplest way to quantize an image classification model to INT8.

**Config file** (`resnet18.json`):

```json
{
    "model_parameters": {
        "onnx_model": "models/resnet18.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "input_parametres": [
            {
                "mean_value": [123.675, 116.28, 103.53],
                "std_value": [58.395, 57.12, 57.375],
                "color_format": "rgb",
                "preprocess_file": "PT_IMAGENET",
                "data_list_path": "./calib_data/img_list.txt"
            }
        ]
    }
}
```

**Run via CLI:**

```bash
python -m xslim -c resnet18.json
```

**Run via Python API:**

```python
import xslim

xslim.quantize_onnx_model("resnet18.json")
```

The quantized model is written to `./output/resnet18.q.onnx`.

---

## 2. Per-subgraph Precision Control (MobileNet V3)

Use `custom_setting` to apply a different precision level to a specific subgraph. This is helpful when the first few layers of the model are sensitive to quantization errors.

**Config file** (`mobilenet_v3_small.json`):

```json
{
    "model_parameters": {
        "onnx_model": "models/mobilenet_v3_small.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
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
        "custom_setting": [
            {
                "input_names": ["input"],
                "output_names": ["input.12"],
                "precision_level": 2
            }
        ]
    }
}
```

**Key points:**
- The subgraph from tensor `input` to tensor `input.12` is quantized at `precision_level: 2` (partial INT8, highest precision).
- The rest of the model uses the default `precision_level: 0` (full INT8).
- Use a tool like Netron to inspect tensor names in your model.

---

## 3. NLP Model with NumPy Inputs (BERT-SQuAD)

For models with multiple non-image inputs (e.g., NLP models), set `file_type` to `npy` and provide a separate calibration list for each input.

**Config file** (`bertsquad.json`):

```json
{
    "model_parameters": {
        "onnx_model": "models/bertsquad.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "input_parametres": [
            {
                "file_type": "npy",
                "data_list_path": "quant_dataset/unique_ids_raw_output.txt"
            },
            {
                "file_type": "npy",
                "data_list_path": "quant_dataset/segment_ids.txt"
            },
            {
                "file_type": "npy",
                "data_list_path": "quant_dataset/input_mask.txt"
            },
            {
                "file_type": "npy",
                "data_list_path": "quant_dataset/input_ids.txt"
            }
        ]
    },
    "quantization_parameters": {
        "finetune_level": 2,
        "precision_level": 2
    }
}
```

**Key points:**
- Each `input_parametres` entry corresponds to one model input in ONNX order.
- `file_type: "npy"` loads calibration data from `.npy` files.
- `precision_level: 2` keeps more layers at higher precision, which is recommended for Transformer models.
- `finetune_level: 2` enables block-wise calibration parameter tuning.

---

## 4. FP16 Conversion

Convert all floating-point operations to FP16. No calibration data is needed.

**Config file** (`mobilenet_v3_small_fp16.json`):

```json
{
    "model_parameters": {
        "onnx_model": "models/mobilenet_v3_small.onnx",
        "working_dir": "./output"
    },
    "quantization_parameters": {
        "precision_level": 4
    }
}
```

**Or use the CLI without a config file:**

```bash
python -m xslim -i models/mobilenet_v3_small.onnx -o output/mobilenet_v3_small_fp16.onnx --fp16
```

---

## 5. Dynamic Quantization

Weights are statically quantized; activations are quantized at runtime. This avoids the need for a calibration dataset.

**Config file** (`mobilenet_v3_small_dyn_quantize.json`):

```json
{
    "model_parameters": {
        "onnx_model": "models/mobilenet_v3_small.onnx",
        "working_dir": "./output"
    },
    "quantization_parameters": {
        "precision_level": 3
    }
}
```

**Or use the CLI without a config file:**

```bash
python -m xslim -i models/mobilenet_v3_small.onnx -o output/mobilenet_v3_small_dyn.onnx --dynq
```

---

## 6. Custom Preprocessing

Use your own preprocessing function when the built-in `PT_IMAGENET` or `IMAGENET` presets don't match your pipeline.

**Preprocessing script** (`preprocess.py`):

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

**Config file** (`resnet18_custom_preprocess.json`):

```json
{
    "model_parameters": {
        "onnx_model": "models/resnet18.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "input_parametres": [
            {
                "mean_value": [123.675, 116.28, 103.53],
                "std_value": [58.395, 57.12, 57.375],
                "color_format": "bgr",
                "preprocess_file": "./preprocess.py:preprocess_impl",
                "data_list_path": "./calib_data/img_list.txt"
            }
        ]
    }
}
```

**Key points:**
- `preprocess_file` follows the format `"path/to/script.py:function_name"`.
- The function receives a list of file paths and the full `input_parametres` entry as a dict.
- It must return a batched `torch.Tensor`.

---

## 7. Using the Python API

All examples above can also be run with the Python API instead of the CLI.

```python
import xslim

# From a JSON config file
xslim.quantize_onnx_model("resnet18.json")

# From a Python dict
config = {
    "model_parameters": {
        "onnx_model": "models/resnet18.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "input_parametres": [{
            "mean_value": [123.675, 116.28, 103.53],
            "std_value": [58.395, 57.12, 57.375],
            "color_format": "rgb",
            "preprocess_file": "PT_IMAGENET",
            "data_list_path": "./calib_data/img_list.txt"
        }]
    }
}
xslim.quantize_onnx_model(config)

# Override model paths at call time
xslim.quantize_onnx_model("resnet18.json", "input.onnx", "output.onnx")
```

---

## Tips

- **Calibration sample count**: 100–300 samples are usually enough. More samples improve calibration quality but increase runtime.
- **Choosing precision level**: Start with `precision_level: 0`. If accuracy drops, try `1` or `2`. Use `4` (FP16) only when INT8 quality is insufficient.
- **Transformer models**: Use `precision_level: 1` or `2` combined with `finetune_level: 2` for best results.
- **Inspecting tensor names**: Use [Netron](https://netron.app) to visualize the ONNX graph and find tensor names for `custom_setting` or `truncate_var_names`.
