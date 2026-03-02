# XSlim

中文 | [English](README.md)

XSlim 是 [SpacemiT](https://www.spacemit.com) 推出的离线（Post-Training）量化工具，集成了已经调整好的适配芯片的量化策略，使用 JSON 配置文件调用统一接口实现模型量化。

## 特性

- **INT8 / FP16 / 动态量化** — 多种精度级别满足不同部署场景
- **JSON 驱动配置** — 简洁的声明式量化设定
- **Python API 与命令行** — 可作为库调用或通过命令行使用
- **自定义预处理** — 支持自定义预处理函数
- **基于 ONNX** — 构建在 ONNX 生态系统之上

## 安装

```bash
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
python -m xslim --config config.json

# FP16 转换
python -m xslim -i input.onnx -o output.onnx --fp16

# 动态量化
python -m xslim -i input.onnx -o output.onnx --dynq
```

## 配置说明

量化通过 JSON 配置文件进行设置，包含以下三个主要部分：

| 配置项 | 说明 |
|---|---|
| `model_parameters` | 输入/输出模型路径及工作目录 |
| `calibration_parameters` | 校准数据集、预处理及观测器设置 |
| `quantization_parameters` | 精度级别、微调级别及子图自定义设置 |

### 精度级别

| 级别 | 说明 |
|---|---|
| 0 | INT8 量化（默认） |
| 1 | INT8 量化（较高精度） |
| 2 | INT8 量化（最高精度） |
| 3 | 动态量化 |
| 4 | FP16 转换 |
| 100 | 仅 ONNX 简化（不进行量化） |

完整的配置参考请查阅 [XSlim 详细说明](https://github.com/spacemit-com/docs-ai/blob/main/zh/compute_stack/ai_compute_stack/xslim.md)。

## 示例

请查看 [samples](samples/) 目录，包含 ResNet-18、MobileNet V3、BERT 等模型的可运行示例。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
