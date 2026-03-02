# XSlim

中文 | [English](README.md)

[![版本](https://img.shields.io/badge/版本-2.0.8-blue.svg)](https://github.com/spacemit-com/xslim/releases)
[![许可证](https://img.shields.io/badge/许可证-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue.svg)](https://www.python.org/)

**XSlim** 是 [SpacemiT](https://www.spacemit.com) 推出的离线（Post-Training）量化工具，集成了已经调整好的适配芯片的量化策略，使用 JSON 配置文件调用统一接口实现 ONNX 模型量化。

---

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [示例](#示例)
- [更新日志](#更新日志)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 特性

- **INT8 / FP16 / 动态量化** — 多种精度级别满足不同部署场景
- **JSON 驱动配置** — 简洁的声明式量化设定
- **Python API 与命令行** — 可作为库调用或通过命令行使用
- **自定义预处理** — 支持自定义预处理函数
- **基于 ONNX** — 构建在 ONNX 生态系统之上

## 安装

```bash
pip install xslim
```

或从源码安装：

```bash
git clone https://github.com/spacemit-com/xslim.git
cd xslim
pip install -r requirements.txt
```

## 快速开始

### Python API

```python
import xslim

# 使用 JSON 配置文件
xslim.quantize_onnx_model("config.json")

# 使用字典
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

# 也可以直接传入模型路径和输出路径
xslim.quantize_onnx_model("config.json", "input.onnx", "output.onnx")
```

### 命令行

```bash
# 使用 JSON 配置进行 INT8 量化
python -m xslim --config config.json

# 指定输入和输出模型路径
python -m xslim -c config.json -i input.onnx -o output.onnx

# 动态量化（无需配置文件）
python -m xslim -i input.onnx -o output.onnx --dynq

# FP16 转换（无需配置文件）
python -m xslim -i input.onnx -o output.onnx --fp16

# 仅模型精简（无需配置文件）
python -m xslim -i input.onnx -o output.onnx
```

## 配置参考

量化通过 JSON 配置文件进行设置，包含三个主要部分：`model_parameters`、`calibration_parameters` 和 `quantization_parameters`。除特殊标注外，以下字段均为可选。

### `model_parameters`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `onnx_model` | — | 输入 ONNX 模型路径 |
| `output_prefix` | 模型文件名（输出以 `.q.onnx` 结尾） | 输出文件前缀 |
| `working_dir` | `onnx_model` 所在目录 | 输出及工作目录 |
| `skip_onnxsim` | `false` | 跳过 ONNX 模型简化 |

### `calibration_parameters`

| 字段 | 默认值 | 可选值 | 说明 |
|---|---|---|---|
| `calibration_step` | `100` | — | 最大校准样本数（建议 100–1000） |
| `calibration_device` | `cuda` | `cuda`、`cpu` | 自动检测，无 GPU 则回退到 `cpu` |
| `calibration_type` | `default` | `default`、`kl`、`minmax`、`percentile`、`mse` | 校准观测器算法 |
| `input_parametres` | — | — | 每个输入的设置列表（见下方） |

#### `input_parametres`（每个输入）

| 字段 | 默认值 | 可选值 | 说明 |
|---|---|---|---|
| `input_name` | 从 ONNX 模型读取 | — | 输入 Tensor 名称 |
| `input_shape` | 从 ONNX 模型读取 | — | 输入 shape（符号 batch 维默认为 1） |
| `dtype` | 从 ONNX 模型读取 | `float32`、`int8`、`uint8`、`int16` | 数据类型 |
| `file_type` | `img` | `img`、`npy`、`raw` | 校准文件类型 |
| `color_format` | `bgr` | `rgb`、`bgr` | 图片色彩格式 |
| `mean_value` | `None` | — | 逐通道均值 |
| `std_value` | `None` | — | 逐通道标准差 |
| `preprocess_file` | `None` | `PT_IMAGENET`、`IMAGENET` 或自定义路径 | 预处理函数（见下方） |
| `data_list_path` | **必填** | — | 校准数据列表文件路径 |

### `quantization_parameters`

| 字段 | 默认值 | 可选值 | 说明 |
|---|---|---|---|
| `precision_level` | `0` | `0`、`1`、`2`、`3`、`4` | 见下方精度级别说明 |
| `finetune_level` | `1` | `0`、`1`、`2`、`3` | 见下方微调级别说明 |
| `analysis_enable` | `true` | — | 是否开启量化后分析 |
| `max_percentile` | `0.9999` | ≥ `0.99` | percentile 截断范围 |
| `custom_setting` | `None` | — | 子图自定义设置列表 |
| `truncate_var_names` | `[]` | — | 用于截断计算图的 Tensor 名称 |
| `ignore_op_types` | `[]` | — | 跳过量化的算子类型 |
| `ignore_op_names` | `[]` | — | 跳过量化的算子名称 |

#### 精度级别

| 级别 | 说明 |
|---|---|
| 0 | 全 INT8 量化（默认） |
| 1 | 部分 INT8 量化，适用于一般 Transformer 模型 |
| 2 | 部分 INT8 量化，最高精度 |
| 3 | 动态量化 |
| 4 | FP16 转换 |

#### 微调级别

| 级别 | 说明 |
|---|---|
| 0 | 不进行任何激进的参数校准 |
| 1 | 可能进行一些静态量化参数校准 |
| 2+ | 将根据逐块量化的损失情况进行量化参数校准 |

### 校准数据列表

`data_list_path` 文件每行表示一个校准数据文件路径，可以写绝对路径，也可以写相对于列表文件所在目录的相对路径。如果模型是多输入的，请确保每个输入的文件列表顺序一致。

```text
data/calib/image_001.JPEG
data/calib/image_002.JPEG
data/calib/image_003.JPEG
```

### 自定义预处理

将 `preprocess_file` 设置为 `"path/to/script.py:function_name"` 即可使用自定义预处理函数：

```python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    读取 path_list 中的文件，依据 input_parametr 中的参数进行预处理，返回一个 torch.Tensor。

    Args:
        path_list: 一个校准 batch 的文件列表。
        input_parametr: 等同于 calibration_parameters.input_parametres[idx]。

    Returns:
        一个 batch 的校准数据 torch.Tensor。
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

## 示例

请查看 [samples](samples/) 目录，包含 ResNet-18、MobileNet V3、BERT 等模型的可运行示例。

## 更新日志

完整的更新记录请查阅 [Releases](https://github.com/spacemit-com/xslim/releases) 页面。

| 版本 | 主要更新 |
|---|---|
| 2.0.8 | 最新开发版本 |
| [2.0.7](https://github.com/spacemit-com/xslim/releases/tag/2.0.7) | 修复复杂模型转换 FP16 的 bug |
| [2.0.6](https://github.com/spacemit-com/xslim/releases/tag/2.0.6) | 修复 metadata props 被删除的问题；默认 CLI 行为调整为模型结构简化（需显式使用 `--dynq` 进行动态量化） |

## 参与贡献

欢迎贡献！请提交 [Issue](https://github.com/spacemit-com/xslim/issues) 或发起 [Pull Request](https://github.com/spacemit-com/xslim/pulls)。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
