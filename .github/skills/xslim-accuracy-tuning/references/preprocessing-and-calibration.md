# Preprocessing and Calibration Reference

Use this file when the task involves calibration data, input mismatch, or unexplained accuracy loss.

## Preprocessing Checklist

Confirm that calibration preprocessing matches inference and training exactly.

| Item | What to verify |
|---|---|
| Pixel value range | `[0, 255]` vs `[0, 1]` must match model expectations |
| Normalization | `mean_value` and `std_value` must match training |
| Color order | `color_format` must match training (`rgb` vs `bgr`) |
| Resize / crop | interpolation and crop behavior must match training |
| Data type | image inputs usually `float32`; token IDs may remain integer |

## Built-in Preprocessing Presets

| `preprocess_file` | Meaning |
|---|---|
| `"PT_IMAGENET"` | PyTorch-style ImageNet preprocessing |
| `"IMAGENET"` | Standard ImageNet preprocessing for BGR/Caffe-style flows |
| `"path/to/script.py:function_name"` | Custom preprocessing function |

## Common Mistakes

### Color order mismatch

```json
"color_format": "bgr"
```

Use this only if training also used BGR. Many PyTorch pipelines require:

```json
"color_format": "rgb"
```

### Mean/std scale mismatch

If images are in `[0, 255]`, avoid `[0, 1]`-scale normalization values:

```json
"mean_value": [0.485, 0.456, 0.406],
"std_value": [0.229, 0.224, 0.225]
```

Use the `[0, 255]` equivalent instead:

```json
"mean_value": [123.675, 116.28, 103.53],
"std_value": [58.395, 57.12, 57.375]
```

## Calibration Sample Guidance

| Scenario | Recommended samples | `calibration_step` reference with batch size 1 |
|---|:---:|:---:|
| Quick validation | 50-100 | 50-100 |
| General CNN | 100-500 | 100-500 |
| Detection / Transformer | 500-1000 | 500-1000 |
| Highest accuracy requirement | 1000+ | 1000+ |

Remember: effective sample count is approximately `calibration_step × batch_size`.

## Calibration Quality Requirements

- Prefer real deployment-like samples.
- Prefer representative diversity over narrow category coverage.
- Avoid duplicate-heavy or synthetic-only calibration sets unless the deployment data is similar.

## Warning Signs of Weak Calibration

- accuracy varies noticeably across runs
- Graphwise histogram bins are extremely sparse
- many layers show moderate error without one dominant hotspot

## Config Reminder

XSlim uses the key:

```json
"input_parametres"
```

Keep that spelling when writing or editing examples.
