# Step 6: Targeted Tuning Based on Analysis Results

[← Step 5](step5.md) | [Back to README →](README.md)

---

## Overview

Once you have identified high-error operators from the Graphwise report, use the following methods for targeted tuning.

---

## Method 1: Increase Local Precision (`custom_setting`)

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

---

## Method 2: Adjust Calibration Type (`calibration_type` / `max_percentile`)

When the report shows a large gap between F.MinMax and Q.MinMax (range clipping):

**If activations have outliers (long-tail F.Hist)**, use `percentile` and lower `max_percentile`:

```json
"calibration_parameters": {
    "calibration_type": "percentile"
},
"quantization_parameters": {
    "max_percentile": 0.9995
}
```

**If activations are approximately uniform or Gaussian**, try `minmax`:

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

---

## Method 3: Exclude Sensitive Operators (`ignore_op_types` / `ignore_op_names`)

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

---

## Method 4: Increase `finetune_level`

When the report shows moderate error across many layers (`0.01 < SNR < 0.1`) without a single dominant problem, increasing `finetune_level` often provides an overall accuracy improvement:

```json
"quantization_parameters": {
    "finetune_level": 2
}
```

---

## Combined Tuning Example

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

## Quick Reference: Accuracy Issue Lookup

| Symptom | Likely Cause | Recommended Action |
|---|---|---|
| Large accuracy drop after quantization (>5%) | Preprocessing mismatch | Verify `mean_value`, `std_value`, `color_format` match training exactly |
| Small accuracy drop (1–5%) | Insufficient or non-representative calibration data | Increase `calibration_step`; ensure data diversity |
| Small accuracy drop | `precision_level` too low | Try `precision_level: 1` or `2` |
| Small accuracy drop | Inaccurate calibration range | Try `calibration_type: "percentile"` or `"kl"` |
| Specific layer has very high SNR in report | Abnormal activation range in that layer | Use `custom_setting` for that subgraph with `precision_level: 2` or `minmax` calibration |
| Many layers show Cosine < 0.99 | High global quantization error | Increase `finetune_level` to `2` or `3` |
| INT8 accuracy unacceptably low regardless of tuning | Model is highly sensitive to quantization | Use `precision_level: 3` (dynamic) or `4` (FP16) |

## Quick Reference: Parameter Selection

| Goal | Recommended Settings |
|---|---|
| Maximum compression | `precision_level: 0`, `finetune_level: 1` |
| Best INT8 accuracy | `precision_level: 2`, `finetune_level: 3`, `calibration_step: 1000` |
| Transformer INT8 | `precision_level: 1`, `finetune_level: 2`, `calibration_type: "default"` |
| No calibration data | `precision_level: 3` (dynamic) or `precision_level: 4` (FP16) |
| Fastest tuning iteration | `precision_level: 0`, `finetune_level: 0`, `calibration_step: 100` |

---

## ✅ Skill Check

1. The Graphwise report shows one Conv layer with `SNR = 0.21` and `Cosine = 0.96`, and the float range `[-80, 75]` is much wider than quantized range `[-4, 4]`. What methods would you apply?
2. After applying all tuning methods, INT8 accuracy is still 8% below the float baseline. What is your next step?

<details>
<summary>Answers</summary>

1. Multiple approaches:
   - Use `custom_setting` with `precision_level: 2` for that layer
   - Set `calibration_type: "percentile"` with a lower `max_percentile` to reduce range clipping
   - Or use `calibration_type: "minmax"` to capture the full range

2. The model is highly sensitive to INT8 quantization. Switch to `precision_level: 3` (dynamic quantization) or `precision_level: 4` (FP16 conversion).

</details>

---

## 🎉 Congratulations!

You have completed the **XSlim Quantization Accuracy Tuning** skill. You are now able to:

- ✅ Choose the right quantization strategy
- ✅ Verify and fix preprocessing mismatches
- ✅ Configure an appropriate calibration dataset
- ✅ Tune `precision_level` and `finetune_level`
- ✅ Interpret the Graphwise Analysis report
- ✅ Apply targeted tuning with `custom_setting`, calibration type adjustment, and operator exclusion

### Next Steps

- Read the full [Accuracy Tuning Guide](../../../doc/accuracy_tuning.md) for additional details
- Explore the [Configuration Reference](../../../doc/configuration.md) for all available options
- Try the [Samples](../../../samples/) for ready-to-run quantization examples

---

[← Step 5 – Read the Graphwise Analysis Report](step5.md) | [Back to README](README.md)
