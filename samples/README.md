# Quantization Samples

[中文版](README_zh.md) | English

This directory contains ready-to-run quantization examples for XSlim.

## Setup

Download the sample models and calibration data:

```bash
bash download_data.sh
```

## Usage

Run quantization via the command line:

```bash
python -m xslim -c resnet18.json
python -m xslim -c mobilenet_v3_small.json
python -m xslim -c resnet18_custom_preprocess.json
python -m xslim -c bertsquad.json
python -m xslim -c mobilenet_v3_small_fp16.json
```

Or use the Python API (see `demo.py`):

```python
import xslim

xslim.quantize_onnx_model("resnet18.json")
```

## Sample Configurations

| Config File | Description |
|---|---|
| `resnet18.json` | INT8 quantization for ResNet-18 |
| `mobilenet_v3_small.json` | INT8 quantization for MobileNet V3 Small with per-subgraph settings |
| `resnet18_custom_preprocess.json` | INT8 quantization with a custom preprocessing function |
| `bertsquad.json` | INT8 quantization for BERT-SQuAD (numpy inputs) |
| `mobilenet_v3_small_fp16.json` | FP16 conversion for MobileNet V3 Small |
| `mobilenet_v3_small_dyn_quantize.json` | Dynamic quantization for MobileNet V3 Small |
