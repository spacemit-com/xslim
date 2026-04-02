# Step 2: Check Input Preprocessing

[← Step 1](step1.md) | [Next: Step 3 →](step3.md)

---

## Why Preprocessing Matters

**Inconsistent input preprocessing is one of the most common causes of quantization accuracy loss.** The preprocessing applied to calibration data must exactly match what the model expects at inference time. A mismatch causes the calibrated quantization parameters to misrepresent the actual activation distribution, leading to significant accuracy degradation.

## Checklist

| Item | Description |
|---|---|
| **Pixel value range** | The range of values in calibration data (e.g., `[0, 255]` vs. `[0.0, 1.0]`) must match what the model was trained with |
| **Normalization parameters** | `mean_value` and `std_value` must exactly match the training normalization |
| **Color channel order** | `color_format` must match training (OpenCV reads images as BGR by default; PyTorch typically uses RGB) |
| **Resize / crop strategy** | Resize interpolation method and crop type (center crop vs. resize-only) must match training |
| **Data type** | Usually `float32`; integer inputs (e.g., NLP token IDs as `int64`) do not require normalization |

## Built-in Preprocessing Presets

| `preprocess_file` value | Description |
|---|---|
| `"PT_IMAGENET"` | PyTorch-style ImageNet preprocessing: resize(256) → center-crop(224) → normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| `"IMAGENET"` | Standard ImageNet preprocessing (BGR channels, suitable for Caffe-style models) |
| `"path/to/script.py:function_name"` | User-defined preprocessing function |

> **Tip:** For PyTorch models using `torchvision.transforms`, `"PT_IMAGENET"` is usually the right choice. For models from other frameworks, write a custom preprocessing function and double-check all parameters.

## Common Mistakes

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

## Example Configuration

```json
{
    "calibration_parameters": {
        "input_parametres": [
            {
                "mean_value": [123.675, 116.28, 103.53],
                "std_value": [58.395, 57.12, 57.375],
                "color_format": "rgb",
                "preprocess_file": "PT_IMAGENET",
                "data_list_path": "./calib_img_list.txt"
            }
        ]
    }
}
```

## ✅ Skill Check

1. Your PyTorch model was trained on ImageNet with standard `torchvision.transforms`. Which `preprocess_file` preset should you use?
2. If `mean_value` is `[0.485, 0.456, 0.406]` but your images are in the `[0, 255]` range, what is the problem and how do you fix it?

<details>
<summary>Answers</summary>

1. `"PT_IMAGENET"` — this matches PyTorch's standard ImageNet preprocessing.
2. The `mean_value` is specified in `[0, 1]` scale but applied to `[0, 255]` images. Fix: use `mean_value: [123.675, 116.28, 103.53]` (multiply by 255) and `std_value: [58.395, 57.12, 57.375]`.

</details>

---

[← Step 1 – Choosing a Quantization Strategy](step1.md) | [Next: Step 3 – Calibration Dataset Size →](step3.md)
