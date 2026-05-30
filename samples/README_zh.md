# 量化示例

中文 | [English](README.md)

本目录包含 XSlim 的可运行量化示例。

## 准备工作

下载示例模型和校准数据：

```bash
bash download_data.sh
```

## 使用方法

通过命令行运行量化：

```bash
python -m xslim -c resnet18.json
python -m xslim -c mobilenet_v3_small.json
python -m xslim -c resnet18_custom_preprocess.json
python -m xslim -c bertsquad.json
python -m xslim -c mobilenet_v3_small_fp16.json
python -m xslim -c mobilenet_v3_small_dyn_quantize.json
```

或使用 Python API（参见 `demo.py`）：

```python
import xslim

xslim.quantize_onnx_model("resnet18.json")
```

## 示例配置说明

| 配置文件 | 说明 |
|---|---|
| `resnet18.json` | ResNet-18 INT8 量化 |
| `mobilenet_v3_small.json` | MobileNet V3 Small INT8 量化（含子图自定义设置） |
| `resnet18_custom_preprocess.json` | 使用自定义预处理函数的 INT8 量化 |
| `bertsquad.json` | BERT-SQuAD INT8 量化（numpy 输入） |
| `mobilenet_v3_small_fp16.json` | MobileNet V3 Small FP16 转换 |
| `mobilenet_v3_small_dyn_quantize.json` | MobileNet V3 Small 动态量化 |

## 说明

- 静态 INT8 示例要求输入为浮点 ONNX 模型。若模型已包含 `QuantizeLinear` 或 `DequantizeLinear`，请改用原始浮点导出模型。
- XSlim 2.1.0 提升了对现代 ONNX 导出模型的兼容性，包括 opset-24 `Pad`、标量规约 Tensor、axes 输入形式的规约算子、更多比较/逻辑算子以及常见激活/一元算子。
