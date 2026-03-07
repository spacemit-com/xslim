# Examples

[中文版](examples_zh.md) | English

Step-by-step examples for the most common XSlim quantization scenarios. All examples assume XSlim is installed (`pip install xslim`) and an ONNX model is available.

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
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57.0, 57.0, 57.0],
                "color_format": "bgr",
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

Use `custom_setting` to apply a different precision level to a specific subgraph. This is helpful when the first few layers are sensitive to quantization errors.

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
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57.0, 57.0, 57.0],
                "color_format": "bgr",
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
- Use a tool like [Netron](https://netron.app) to inspect tensor names in your model.

---

## 3. Multi-input NLP Model (BERT-SQuAD)

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
- `precision_level: 2` keeps more layers at higher precision — recommended for Transformer models.
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
python -m xslim -i models/mobilenet_v3_small.onnx -o output/mobilenet_fp16.onnx --fp16
```

---

## 5. Dynamic Quantization

Weights are statically quantized; activations are quantized at runtime. No calibration dataset is required.

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
python -m xslim -i models/mobilenet_v3_small.onnx -o output/mobilenet_dynq.onnx --dynq
```

---

## 6. Custom Preprocessing

Use your own preprocessing function when the built-in `PT_IMAGENET` / `IMAGENET` presets don't match your pipeline.

**Preprocessing script** (`preprocess.py`):

```python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Args:
        path_list: List of file paths for one calibration batch.
        input_parametr: The corresponding entry from calibration_parameters.input_parametres.
    Returns:
        A batched torch.Tensor of shape [batch, C, H, W].
    """
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
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57.0, 57.0, 57.0],
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
- It must return a batched `torch.Tensor` of shape `[batch, C, H, W]`.
- For multi-input models with similar preprocessing, the same function can be reused across entries.

---

## 7. Python API

All scenarios above can be driven through the Python API instead of the CLI.

```python
import xslim
import onnx

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
            "mean_value": [103.94, 116.78, 123.68],
            "std_value": [57.0, 57.0, 57.0],
            "color_format": "bgr",
            "preprocess_file": "PT_IMAGENET",
            "data_list_path": "./calib_data/img_list.txt"
        }]
    }
}
xslim.quantize_onnx_model(config)

# Override model paths at call time (string path)
xslim.quantize_onnx_model("resnet18.json", "input.onnx", "output.onnx")

# Pass an already-loaded onnx.ModelProto; returns the quantized ModelProto
onnx_model = onnx.load("models/resnet18.onnx")
quantized_model = xslim.quantize_onnx_model("resnet18.json", onnx_model)
```

---

## Tips

- **Calibration sample count**: 100–300 samples are usually sufficient. More samples improve calibration quality but increase runtime.
- **Choosing precision level**: Start with `precision_level: 0`. If accuracy drops, try `1` or `2`. Use `4` (FP16) only when INT8 quality is insufficient.
- **Transformer models**: Use `precision_level: 1` or `2` combined with `finetune_level: 2` for best results.
- **Inspecting tensor names**: Use [Netron](https://netron.app) to visualize the ONNX graph and find tensor names for `custom_setting` or `truncate_var_names`.
