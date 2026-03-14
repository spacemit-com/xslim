# XSlim

中文 | [English](README.md)

[![版本](https://img.shields.io/badge/版本-2.0.10-blue.svg)](https://github.com/spacemit-com/xslim/releases)
[![许可证](https://img.shields.io/badge/许可证-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.6-blue.svg)](https://www.python.org/)

**XSlim** 是 [SpacemiT](https://www.spacemit.com) 推出的离线（Post-Training）量化工具，集成了已经调整好的适配芯片的量化策略，使用 JSON 配置文件调用统一接口实现 ONNX 模型量化。

---

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [文档](#文档)
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

## 文档

- [配置参考](doc/configuration_zh.md) — 所有 JSON 配置选项的完整说明
- [使用示例](doc/examples_zh.md) — INT8、FP16、动态量化、自定义预处理等场景的分步指南

## 示例

请查看 [samples](samples/) 目录，包含 ResNet-18、MobileNet V3、BERT 等模型的可运行示例。

## 更新日志

完整的更新记录请查阅 [Releases](https://github.com/spacemit-com/xslim/releases) 页面。

| 版本 | 主要更新 |
|---|---|
| 2.0.10 | 当前开发版本 |
| [2.0.9](https://github.com/spacemit-com/xslim/releases/tag/2.0.9) | 最新发布版本 |
| [2.0.7](https://github.com/spacemit-com/xslim/releases/tag/2.0.7) | 修复复杂模型转换 FP16 的 bug |
| [2.0.6](https://github.com/spacemit-com/xslim/releases/tag/2.0.6) | 修复 metadata props 被删除的问题；默认 CLI 行为调整为模型结构简化（需显式使用 `--dynq` 进行动态量化） |

## 参与贡献

欢迎贡献！请提交 [Issue](https://github.com/spacemit-com/xslim/issues) 或发起 [Pull Request](https://github.com/spacemit-com/xslim/pulls)。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
