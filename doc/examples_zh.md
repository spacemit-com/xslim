# 使用示例

中文 | [English](examples.md)

常见 XSlim 量化场景的分步示例。所有示例均假设您已安装 XSlim（`pip install xslim`）并已准备好 ONNX 模型。

可运行版本的示例请参阅 [samples](../samples/) 目录。

---

## 1. 基础 INT8 量化（ResNet-18）

对图像分类模型进行 INT8 量化的最简方式。

**配置文件**（`resnet18.json`）：

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

**命令行运行：**

```bash
python -m xslim -c resnet18.json
```

**Python API 运行：**

```python
import xslim

xslim.quantize_onnx_model("resnet18.json")
```

量化后的模型将写入 `./output/resnet18.q.onnx`。

---

## 2. 子图精度控制（MobileNet V3）

使用 `custom_setting` 对特定子图应用不同的精度级别。当模型的前几层对量化误差敏感时，此方法非常有效。

**配置文件**（`mobilenet_v3_small.json`）：

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

**要点：**
- 从 Tensor `input` 到 `input.12` 的子图使用 `precision_level: 2`（部分 INT8，最高精度）。
- 模型其余部分使用默认的 `precision_level: 0`（全 INT8）。
- 可使用 [Netron](https://netron.app) 查看模型中的 Tensor 名称。

---

## 3. 多输入 NLP 模型（BERT-SQuAD）

对于具有多个非图像输入的模型（如 NLP 模型），将 `file_type` 设置为 `npy`，并为每个输入提供独立的校准数据列表。

**配置文件**（`bertsquad.json`）：

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

**要点：**
- 每个 `input_parametres` 条目按 ONNX 模型输入顺序对应一个模型输入。
- `file_type: "npy"` 从 `.npy` 文件加载校准数据。
- `precision_level: 2` 保留更多层的高精度，推荐用于 Transformer 模型。
- `finetune_level: 2` 启用逐块量化参数校准。

---

## 4. FP16 转换

将所有浮点运算转换为 FP16，无需校准数据。

**配置文件**（`mobilenet_v3_small_fp16.json`）：

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

**或通过命令行直接运行（无需配置文件）：**

```bash
python -m xslim -i models/mobilenet_v3_small.onnx -o output/mobilenet_fp16.onnx --fp16
```

---

## 5. 动态量化

权重静态量化，激活值在运行时量化，无需准备校准数据集。

**配置文件**（`mobilenet_v3_small_dyn_quantize.json`）：

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

**或通过命令行直接运行（无需配置文件）：**

```bash
python -m xslim -i models/mobilenet_v3_small.onnx -o output/mobilenet_dynq.onnx --dynq
```

---

## 6. 自定义预处理

当内置的 `PT_IMAGENET` 或 `IMAGENET` 预设不符合您的处理流程时，可使用自定义预处理函数。

**预处理脚本**（`preprocess.py`）：

```python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Args:
        path_list: 一个校准 batch 的文件路径列表。
        input_parametr: calibration_parameters.input_parametres 中对应的条目。
    Returns:
        形状为 [batch, C, H, W] 的批量 torch.Tensor。
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

**配置文件**（`resnet18_custom_preprocess.json`）：

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

**要点：**
- `preprocess_file` 遵循 `"path/to/script.py:function_name"` 格式。
- 函数接收文件路径列表和完整的 `input_parametres` 条目（dict 类型）。
- 函数必须返回形状为 `[batch, C, H, W]` 的批量 `torch.Tensor`。
- 对于多输入模型，若各输入的预处理逻辑相近，可直接复用同一函数。

---

## 7. Python API

上述所有场景均可通过 Python API 代替命令行驱动。

```python
import xslim
import onnx

# 通过 JSON 配置文件
xslim.quantize_onnx_model("resnet18.json")

# 通过 Python 字典
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

# 在调用时覆盖模型路径（字符串路径）
xslim.quantize_onnx_model("resnet18.json", "input.onnx", "output.onnx")

# 传入已加载的 onnx.ModelProto；函数返回量化后的 ModelProto
onnx_model = onnx.load("models/resnet18.onnx")
quantized_model = xslim.quantize_onnx_model("resnet18.json", onnx_model)
```

---

## 使用建议

- **校准样本数量**：通常 100–300 个样本即可。更多样本可提高校准质量，但会增加耗时。
- **精度级别选择**：从 `precision_level: 0` 开始。若精度下降，依次尝试 `1` 或 `2`。仅当 INT8 质量不满足要求时才使用 `4`（FP16）。
- **Transformer 模型**：建议使用 `precision_level: 1` 或 `2`，结合 `finetune_level: 2` 以获得最佳效果。
- **查看 Tensor 名称**：使用 [Netron](https://netron.app) 可视化 ONNX 计算图，找到用于 `custom_setting` 或 `truncate_var_names` 的 Tensor 名称。
