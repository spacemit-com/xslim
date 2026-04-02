# Quantization Accuracy Tuning Guide

[中文版](accuracy_tuning_zh.md) | English

This guide explains how to systematically diagnose and improve model accuracy after quantization with XSlim.

---

## Table of Contents

1. [Choosing a Quantization Strategy](#1-choosing-a-quantization-strategy)
2. [Tuning Workflow Overview](#2-tuning-workflow-overview)
3. [Step 1: Check Input Preprocessing](#3-step-1-check-input-preprocessing)
4. [Step 2: Calibration Dataset Size](#4-step-2-calibration-dataset-size)
5. [Step 3: Adjust precision_level and finetune_level](#5-step-3-adjust-precision_level-and-finetune_level)
6. [Step 4: Read the Graphwise Analysis Report](#6-step-4-read-the-graphwise-analysis-report)
7. [Step 5: Targeted Tuning Based on Analysis Results](#7-step-5-targeted-tuning-based-on-analysis-results)
8. [Quick Reference](#8-quick-reference)

---

## 1. Choosing a Quantization Strategy

XSlim supports three quantization strategies. Try them in the following priority order:

| Priority | Strategy | `precision_level` | Notes |
|:---:|---|:---:|---|
| 1 | **Static INT8 quantization** | `0`–`2` | Weights and activations are both quantized statically. Highest compression, fastest inference. **Start here.** |
| 2 | **Dynamic quantization** | `3` | Weights are quantized statically; activations are quantized at runtime. No calibration data required. Suitable for models with unstable activation distributions (e.g., certain NLP models). |
| 3 | **FP16 conversion** | `4` | All floating-point operators are converted to FP16. No calibration data required. Minimal accuracy loss but lower compression and speedup than INT8. |

> **Recommendation:** Fully tune static INT8 quantization first. Only consider dynamic quantization or FP16 when INT8 accuracy cannot meet your requirements.

### Quick Comparison

| Strategy | Calibration Data | Accuracy Risk | Inference Speedup | Typical Use Case |
|---|:---:|:---:|:---:|---|
| Static INT8 | ✅ Required | Medium | ⭐⭐⭐⭐ | Image classification, detection, most CNNs |
| Dynamic quantization | ❌ Not required | Low–Medium | ⭐⭐⭐ | LSTM, some Transformers |
| FP16 | ❌ Not required | Very low | ⭐⭐ | Accuracy-critical models, no calibration data available |

---

## 2. Tuning Workflow Overview

```
Prepare model and calibration data
              │
              ▼
   [Step 1] Check input preprocessing
              │ Preprocessing matches inference?
              ▼
   [Step 2] Is the calibration dataset large enough?
              │
              ▼
   [Step 3] Adjust precision_level / finetune_level
              │
              ▼
   [Step 4] Enable analysis_enable, read Graphwise report
              │ High-error layers found?
              ▼
   [Step 5] Use custom_setting to target problem layers
              │
              ▼
         Accuracy acceptable? ──Yes──▶ Done
                   │
                  No
                   ▼
         Try higher precision_level or switch strategy
```

---

## 3. Step 1: Check Input Preprocessing

**Inconsistent input preprocessing is one of the most common causes of quantization accuracy loss.** The preprocessing applied to calibration data must exactly match what the model expects at inference time. A mismatch causes the calibrated quantization parameters to misrepresent the actual activation distribution, leading to significant accuracy degradation.

### Checklist

| Item | Description |
|---|---|
| **Pixel value range** | The range of values in calibration data (e.g., `[0, 255]` vs. `[0.0, 1.0]`) must match what the model was trained with |
| **Normalization parameters** | `mean_value` and `std_value` must exactly match the training normalization |
| **Color channel order** | `color_format` must match training (OpenCV reads images as BGR by default; PyTorch typically uses RGB) |
| **Resize / crop strategy** | Resize interpolation method and crop type (center crop vs. resize-only) must match training |
| **Data type** | Usually `float32`; integer inputs (e.g., NLP token IDs as `int64`) do not require normalization |

### Built-in Preprocessing Presets

| `preprocess_file` value | Description |
|---|---|
| `"PT_IMAGENET"` | PyTorch-style ImageNet preprocessing: resize(256) → center-crop(224) → normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| `"IMAGENET"` | Standard ImageNet preprocessing (BGR channels, suitable for Caffe-style models) |
| `"path/to/script.py:function_name"` | User-defined preprocessing function (see [Custom Preprocessing](configuration.md#custom-preprocessing)) |

> **Tip:** For PyTorch models using `torchvision.transforms`, `"PT_IMAGENET"` is usually the right choice. For models from other frameworks, write a custom preprocessing function and double-check all parameters.

### Common Mistakes

```json
// ❌ Wrong: model trained with RGB, but calibration uses BGR (default)
"color_format": "bgr"

// ✅ Correct: matches training
"color_format": "rgb"
```

```json
// ❌ Wrong: mean is in [0,1] range but images are in [0,255]
"mean_value": [0.485, 0.456, 0.406],
"std_value": [0.229, 0.224, 0.225]

// ✅ Correct: mean matches the actual pixel value range [0,255]
"mean_value": [123.675, 116.28, 103.53],
"std_value": [58.395, 57.12, 57.375]
```

---

## 4. Step 2: Calibration Dataset Size

The number of calibration samples directly affects the quality of activation range statistics and therefore quantization accuracy.

### Recommended Counts

| Scenario | Recommended samples | `calibration_step` reference (batch_size=1) |
|---|:---:|:---:|
| Quick validation | 50–100 | 50–100 |
| General (CNN, classification) | 100–500 | 100–500 |
| Complex models (Transformer, detection) | 500–1000 | 500–1000 |
| High-accuracy requirements | ≥ 1000 | ≥ 1000 |

> **Note:** `calibration_step` counts dataloader iterations, so effective sample count ≈ `calibration_step × batch_size`. The default is `500`.

### Calibration Data Quality Requirements

- **Diversity:** Calibration data should cover the typical input distribution the model will encounter at inference, not just a single category or scene.
- **Representativeness:** Sample randomly from a validation or test set. Avoid using duplicate samples from the training set.
- **Realism:** Use real data that resembles the actual deployment scenario rather than synthetic images.

### Symptoms of Insufficient Calibration Data

- Accuracy varies noticeably across runs (due to random sampling of calibration data)
- Graphwise report shows extremely uneven `F.Hist` distributions (values concentrated in only a few bins)

---

## 5. Step 3: Adjust precision_level and finetune_level

Once preprocessing is verified and calibration data is adequate, systematically improve accuracy by adjusting these two parameters.

### `precision_level`

| Level | Quantization Strategy | Typical Use Case |
|:---:|---|---|
| `0` | Full INT8 (default) | Most CNNs (ResNet, MobileNet, EfficientNet, etc.) |
| `1` | Partial INT8 — a few sensitive layers kept at higher precision | General Transformer models, models with Attention |
| `2` | Partial INT8 — more layers kept at higher precision | Accuracy-sensitive models, complex Transformers |
| `3` | Dynamic quantization | Models with highly variable activation distributions, no calibration data available |
| `4` | FP16 conversion | Extremely accuracy-sensitive models, or when calibration data is unavailable |

**Tuning sequence:**
1. Start with `precision_level: 0`;
2. If accuracy is insufficient, try `precision_level: 1`;
3. Still insufficient: try `precision_level: 2`;
4. When INT8 cannot meet requirements: try `precision_level: 3` (dynamic) or `4` (FP16).

### `finetune_level`

| Level | Behavior | Runtime | Use Case |
|:---:|---|:---:|---|
| `0` | No quantization parameter adjustment | Fastest | Quick validation, relaxed accuracy requirements |
| `1` | Lightweight static calibration (default) | Short | Default for most scenarios |
| `2` | Block-wise calibration | Medium | Accuracy-sensitive models, Transformers |
| `3` | More aggressive block-wise calibration | Long | Highest quality, maximum accuracy |

**Tuning advice:**
- General models: use default `finetune_level: 1`;
- Transformer / NLP models: recommend `finetune_level: 2`;
- Highest accuracy requirement: try `finetune_level: 3` (note significantly longer runtime).

### Recommended Combinations

| Model Type | Recommended Settings |
|---|---|
| CNN image classification (ResNet, MobileNet, etc.) | `precision_level: 0`, `finetune_level: 1` |
| Object detection (YOLO, SSD, etc.) | `precision_level: 1`, `finetune_level: 2` |
| Transformer / BERT / ViT | `precision_level: 1` or `2`, `finetune_level: 2` |
| LSTM / RNN | `precision_level: 3` (dynamic quantization) |
| Extreme accuracy sensitivity | `precision_level: 4` (FP16) |

---

## 6. Step 4: Read the Graphwise Analysis Report

After quantization, XSlim automatically runs a Graphwise Analysis (controlled by `analysis_enable: true`) and writes a report to the working directory as `<output_prefix>_report.md`.

### Report Structure

The report is a Markdown table **sorted by quantization error from highest to lowest**. Each row corresponds to a single tensor (input or output) of one operator:

| Column | Meaning | Healthy Reference |
|---|---|:---:|
| `Op` | Operator name and type in the format `name[type]` | — |
| `Var` | Tensor name (`[Constant]` suffix means it is a weight parameter) | — |
| `SNR` | Signal-to-noise ratio (quantization noise / signal; lower is better). **Red when > 0.1** | `< 0.1` |
| `MSE` | Mean squared error (lower is better) | As small as possible |
| `Cosine` | Cosine similarity between quantized and float values (higher is better). **Red when < 0.99** | `> 0.99` |
| `Q.MinMax` | Min and max values of the quantized tensor | — |
| `F.MinMax` | Min and max values of the float (original) tensor | — |
| `F.Hist` | Float value distribution histogram (32 bins, comma-separated counts) | — |

### Key Metric Interpretation

#### SNR (Signal-to-Noise Ratio)

SNR = energy of quantization error / energy of original signal. **Higher SNR means larger quantization error.**

- `SNR < 0.01`: Negligible error — no action needed;
- `0.01 ≤ SNR < 0.1`: Acceptable error range;
- `SNR ≥ 0.1` (highlighted red): High error — this operator is a priority tuning target.

#### Cosine Similarity

Measures directional alignment between quantized and float values.

- `Cosine > 0.999`: Excellent alignment — quantization quality is good;
- `0.99 ≤ Cosine ≤ 0.999`: Slight deviation;
- `Cosine < 0.99` (highlighted red): Significant deviation — this operator requires attention.

#### F.MinMax vs. Q.MinMax

- A large gap (e.g., float range `[-100, 100]` but quantized range `[-5, 5]`) indicates **range clipping**, which causes information loss and accuracy degradation;
- Try adjusting `calibration_type` (e.g., switch to `minmax`) or increasing `max_percentile` (e.g., from `0.9999` to `0.99999`).

#### F.Hist (Value Distribution Histogram)

- Heavily skewed histogram (most values in just a few bins) indicates outliers in activation values — try `percentile` calibration with a lower `max_percentile`;
- Very sparse histogram (many bins are zero) may indicate insufficient calibration data — increase `calibration_step`.

### Example Report Interpretation

```markdown
| Op | Var | SNR | MSE | Cosine | Q.MinMax | F.MinMax | F.Hist |
|---|---|---|---|---|---|---|---|
| /model/layer4/conv[Conv] | /layer4/conv/output | <font color="red">0.1523</font> | 0.0312 | <font color="red">0.9743</font> | -3.21, 3.18 | -15.32, 12.87 | 0,0,2,8,45,... |
| /model/layer1/conv[Conv] | /layer1/conv/output | 0.0045 | 0.0003 | 0.9998 | -1.02, 1.05 | -1.05, 1.08 | 3,12,55,... |
```

Reading this example:
- `layer4/conv` has SNR `0.1523` (red) and Cosine `0.9743` (red) — **this is the primary source of accuracy loss**;
- The float range `[-15.32, 12.87]` is far wider than the quantized range `[-3.21, 3.18]`, indicating severe range clipping;
- `layer1/conv` has healthy metrics — no action needed.

---

## 7. Step 5: Targeted Tuning Based on Analysis Results

Once you have identified high-error operators from the Graphwise report, use the following methods for targeted tuning.

### Method 1: Increase Local Precision (`custom_setting`)

For subgraphs containing operators with abnormal SNR/Cosine, raise the `precision_level` for that region:

```json
"quantization_parameters": {
    "precision_level": 0,
    "custom_setting": [
        {
            "input_names": ["problem_layer_input_tensor"],
            "output_names": ["problem_layer_output_tensor"],
            "precision_level": 2
        }
    ]
}
```

> **Tip:** Use [Netron](https://netron.app) to open the ONNX model and find the input/output tensor names of the problem operator.

### Method 2: Adjust Calibration Type (`calibration_type` / `max_percentile`)

When the report shows a large gap between F.MinMax and Q.MinMax (range clipping):

- If activations have outliers (long-tail F.Hist), use `percentile` and lower `max_percentile`:

```json
"calibration_parameters": {
    "calibration_type": "percentile"
},
"quantization_parameters": {
    "max_percentile": 0.9995
}
```

- If activations are approximately uniform or Gaussian, try `minmax`:

```json
"calibration_parameters": {
    "calibration_type": "minmax"
}
```

You can also apply a different calibration strategy only to the problem subgraph via `custom_setting`:

```json
"quantization_parameters": {
    "custom_setting": [
        {
            "input_names": ["problem_layer_input"],
            "output_names": ["problem_layer_output"],
            "calibration_type": "percentile",
            "max_percentile": 0.9995
        }
    ]
}
```

### Method 3: Exclude Sensitive Operators (`ignore_op_types` / `ignore_op_names`)

When certain operator types (e.g., `Softmax`, `LayerNormalization`) consistently show high error in the report, exclude them from quantization:

```json
"quantization_parameters": {
    "ignore_op_types": ["Softmax", "LayerNormalization"]
}
```

Or exclude specific operators by name:

```json
"quantization_parameters": {
    "ignore_op_names": ["/model/encoder/layer.0/attention/MatMul"]
}
```

### Method 4: Increase finetune_level

When the report shows moderate error across many layers (`0.01 < SNR < 0.1`) without a single dominant problem, increasing `finetune_level` often provides an overall accuracy improvement:

```json
"quantization_parameters": {
    "finetune_level": 2
}
```

### Combined Tuning Example

The following configuration combines multiple methods for a CNN model with high accuracy requirements:

```json
{
    "model_parameters": {
        "onnx_model": "models/my_model.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "calibration_step": 500,
        "calibration_type": "percentile",
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
        "precision_level": 0,
        "finetune_level": 2,
        "max_percentile": 0.9999,
        "analysis_enable": true,
        "ignore_op_types": ["Softmax"],
        "custom_setting": [
            {
                "input_names": ["problem_layer_input"],
                "output_names": ["problem_layer_output"],
                "precision_level": 2,
                "calibration_type": "minmax"
            }
        ]
    }
}
```

---

## 8. Quick Reference

### Accuracy Issue Lookup

| Symptom | Likely Cause | Recommended Action |
|---|---|---|
| Large accuracy drop after quantization (>5%) | Preprocessing mismatch | Verify `mean_value`, `std_value`, `color_format` match training exactly |
| Small accuracy drop (1–5%) | Insufficient or non-representative calibration data | Increase `calibration_step`; ensure data diversity |
| Small accuracy drop | `precision_level` too low | Try `precision_level: 1` or `2` |
| Small accuracy drop | Inaccurate calibration range | Try `calibration_type: "percentile"` or `"kl"` |
| Specific layer has very high SNR in report | Abnormal activation range in that layer | Use `custom_setting` for that subgraph with `precision_level: 2` or `minmax` calibration |
| Many layers show Cosine < 0.99 | High global quantization error | Increase `finetune_level` to `2` or `3` |
| INT8 accuracy unacceptably low regardless of tuning | Model is highly sensitive to quantization | Use `precision_level: 3` (dynamic) or `4` (FP16) |

### Parameter Selection Quick Reference

| Goal | Recommended Settings |
|---|---|
| Maximum compression | `precision_level: 0`, `finetune_level: 1` |
| Best INT8 accuracy | `precision_level: 2`, `finetune_level: 3`, `calibration_step: 1000` |
| Transformer INT8 | `precision_level: 1`, `finetune_level: 2`, `calibration_type: "default"` |
| No calibration data | `precision_level: 3` (dynamic) or `precision_level: 4` (FP16) |
| Fastest tuning iteration | `precision_level: 0`, `finetune_level: 0`, `calibration_step: 100` |

---

## Reference Documents

- [Configuration Reference](configuration.md) — Complete description of all JSON configuration fields
- [Examples](examples.md) — Step-by-step examples for common quantization scenarios
