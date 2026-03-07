# 配置参考

中文 | [English](configuration.md)

本页面记录了 XSlim JSON 配置文件中的所有字段。除特殊标注外，所有字段均为**可选**。

## 顶层结构

```json
{
    "model_parameters": { ... },
    "calibration_parameters": { ... },
    "quantization_parameters": { ... }
}
```

---

## `model_parameters`

控制 ONNX 模型的输入/输出路径及预处理。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `onnx_model` | `string` | — | 输入 ONNX 模型文件路径 |
| `output_prefix` | `string` | 模型文件名（输出以 `.q.onnx` 结尾） | 输出文件名前缀 |
| `working_dir` | `string` | `onnx_model` 所在目录 | 输出模型及中间文件的写入目录 |
| `skip_onnxsim` | `bool` | `false` | 设为 `true` 可跳过量化前的 ONNX 模型简化 |

### 示例

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

控制校准数据的加载方式及量化范围的计算。

| 字段 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `calibration_step` | `int` | `100` | — | 最大校准样本数，建议范围：100–1000 |
| `calibration_device` | `string` | `cuda` | `cuda`、`cpu` | 校准推理设备，自动检测，无 GPU 时回退到 `cpu` |
| `calibration_type` | `string` | `default` | `default`、`kl`、`minmax`、`percentile`、`mse` | 激活值范围计算所用的观测算法 |
| `input_parametres` | `list` | — | — | 每个输入的校准设置，每个模型输入对应一项（见下方） |

### 校准类型说明

| 值 | 说明 |
|---|---|
| `default` | 使用芯片推荐算法（通常为 `kl` 或 `percentile`） |
| `kl` | KL 散度最小化 |
| `minmax` | 使用观测到的最小值和最大值 |
| `percentile` | 按 `max_percentile` 截断范围，减少异常值影响 |
| `mse` | 最小化原始激活值与量化激活值之间的均方误差 |

### `input_parametres`（每个输入）

列表中每项对应一个模型输入，顺序与 ONNX 模型的输入顺序一致。

| 字段 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `input_name` | `string` | 从 ONNX 模型读取 | — | 输入 Tensor 名称 |
| `input_shape` | `list[int]` | 从 ONNX 模型读取 | — | 输入的 shape，符号 batch 维默认为 `1` |
| `dtype` | `string` | 从 ONNX 模型读取 | `float32`、`int8`、`uint8`、`int16` | 模型输入的数据类型 |
| `file_type` | `string` | `img` | `img`、`npy`、`raw` | 校准文件格式 |
| `color_format` | `string` | `bgr` | `rgb`、`bgr` | 图片输入的色彩通道顺序 |
| `mean_value` | `list[float]` | `null` | — | 归一化时逐通道减去的均值 |
| `std_value` | `list[float]` | `null` | — | 归一化时逐通道除以的标准差 |
| `preprocess_file` | `string` | `null` | `PT_IMAGENET`、`IMAGENET` 或自定义路径 | 预处理函数（见[自定义预处理](#自定义预处理)） |
| `data_list_path` | `string` | **必填** | — | 列出校准数据文件路径的文本文件路径 |

#### `file_type` 说明

| 值 | 说明 |
|---|---|
| `img` | 标准图像文件（JPEG、PNG、BMP 等），使用 OpenCV 读取 |
| `npy` | 包含单个数组的 NumPy `.npy` 文件 |
| `raw` | 直接以字节流读取的原始二进制文件 |

### 校准数据列表文件

`data_list_path` 指向一个纯文本文件，每行一个校准文件路径。路径可以是绝对路径，也可以是相对于列表文件所在目录的相对路径。

```text
data/calib/image_001.JPEG
data/calib/image_002.JPEG
data/calib/image_003.JPEG
```

对于多输入模型，每个 `input_parametres` 条目都有各自的列表文件。所有列表中第 *N* 个文件共同构成一个校准 batch，因此各列表的长度和顺序必须保持一致。

### 自定义预处理

`preprocess_file` 可设置为以下值之一：

- `"PT_IMAGENET"` — 内置 PyTorch 风格的 ImageNet 预处理（resize → center-crop → normalize）
- `"IMAGENET"` — 内置标准 ImageNet 预处理
- `"path/to/script.py:function_name"` — 用户自定义 Python 文件中的函数

自定义预处理函数必须符合以下签名：

```python
from typing import Sequence
import torch

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Args:
        path_list: 一个校准 batch 的文件路径列表。
        input_parametr: calibration_parameters.input_parametres 中对应的条目。

    Returns:
        形状为 [batch, ...] 的批量 torch.Tensor。
    """
    ...
```

完整示例：

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

控制量化策略和精度。

| 字段 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `precision_level` | `int` | `0` | `0`–`4` | 全局精度级别（见下方） |
| `finetune_level` | `int` | `1` | `0`–`3` | 量化参数校准的激进程度（见下方） |
| `analysis_enable` | `bool` | `true` | — | 是否运行量化后精度分析 |
| `max_percentile` | `float` | `0.9999` | ≥ `0.99` | `calibration_type` 为 `percentile` 时的百分位截断阈值 |
| `custom_setting` | `list` | `null` | — | 子图精度覆盖设置（见下方） |
| `truncate_var_names` | `list[string]` | `[]` | — | 用于将计算图切分为子图的 Tensor 名称 |
| `ignore_op_types` | `list[string]` | `[]` | — | 不参与量化的 ONNX 算子类型 |
| `ignore_op_names` | `list[string]` | `[]` | — | 不参与量化的特定算子名称 |

### 精度级别

| 级别 | 说明 |
|---|---|
| `0` | 全 INT8 量化（默认）。压缩率最高，可能影响精度 |
| `1` | 部分 INT8 — 敏感层保持较高精度。适用于一般 Transformer 模型 |
| `2` | 部分 INT8，精度保留最高。适用于对精度要求严格的模型 |
| `3` | 动态量化 — 权重静态量化，激活值在运行时量化 |
| `4` | FP16 转换 — 所有浮点算子转换为 FP16（无需校准数据） |

### 微调级别

| 级别 | 说明 |
|---|---|
| `0` | 不进行任何量化参数调整 |
| `1` | 可能进行轻量级静态量化参数校准 |
| `2` | 根据量化损失进行逐块量化参数校准 |
| `3` | 更激进的逐块校准，质量更高但耗时更长 |

### `custom_setting`

子图精度覆盖列表，每项通过边界 Tensor 选定子图并设置局部精度级别。

| 字段 | 类型 | 说明 |
|---|---|---|
| `input_names` | `list[string]` | 标记子图起始位置的 Tensor 名称 |
| `output_names` | `list[string]` | 标记子图结束位置的 Tensor 名称 |
| `precision_level` | `int` | 应用于该子图的精度级别（与全局 `precision_level` 取值相同） |

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

指定用于切分计算图的 Tensor 名称列表。每个 Tensor 同时成为前一子图的输出和后一子图的输入。当模型首尾部分需要差异化处理时非常有用。

```json
"truncate_var_names": ["features.0.0.weight", "classifier.1.weight"]
```

### `ignore_op_types` 与 `ignore_op_names`

匹配 `ignore_op_types`（按 ONNX 算子类型）或 `ignore_op_names`（按节点名称）的算子将被排除在量化范围之外，保持原始精度。

```json
"ignore_op_types": ["LayerNormalization", "Softmax"],
"ignore_op_names": ["/model/encoder/layer.0/attention/MatMul"]
```

---

## 完整配置示例

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
