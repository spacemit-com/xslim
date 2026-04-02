# Step 4: Adjust `precision_level` and `finetune_level`

[← Step 3](step3.md) | [Next: Step 5 →](step5.md)

---

## Overview

Once preprocessing is verified and calibration data is adequate, systematically improve accuracy by adjusting these two parameters.

## `precision_level`

| Level | Quantization Strategy | Typical Use Case |
|:---:|---|---|
| `0` | Full INT8 (default) | Most CNNs (ResNet, MobileNet, EfficientNet, etc.) |
| `1` | Partial INT8 — a few sensitive layers kept at higher precision | General Transformer models, models with Attention |
| `2` | Partial INT8 — more layers kept at higher precision | Accuracy-sensitive models, complex Transformers |
| `3` | Dynamic quantization | Models with highly variable activation distributions, no calibration data available |
| `4` | FP16 conversion | Extremely accuracy-sensitive models, or when calibration data is unavailable |

**Tuning sequence:**

1. Start with `precision_level: 0`
2. If accuracy is insufficient, try `precision_level: 1`
3. Still insufficient: try `precision_level: 2`
4. When INT8 cannot meet requirements: try `precision_level: 3` (dynamic) or `4` (FP16)

## `finetune_level`

| Level | Behavior | Runtime | Use Case |
|:---:|---|:---:|---|
| `0` | No quantization parameter adjustment | Fastest | Quick validation, relaxed accuracy requirements |
| `1` | Lightweight static calibration (default) | Short | Default for most scenarios |
| `2` | Block-wise calibration | Medium | Accuracy-sensitive models, Transformers |
| `3` | More aggressive block-wise calibration | Long | Highest quality, maximum accuracy |

**Tuning advice:**

- General models: use default `finetune_level: 1`
- Transformer / NLP models: recommend `finetune_level: 2`
- Highest accuracy requirement: try `finetune_level: 3` (note significantly longer runtime)

## Recommended Combinations

| Model Type | Recommended Settings |
|---|---|
| CNN image classification (ResNet, MobileNet, etc.) | `precision_level: 0`, `finetune_level: 1` |
| Object detection (YOLO, SSD, etc.) | `precision_level: 1`, `finetune_level: 2` |
| Transformer / BERT / ViT | `precision_level: 1` or `2`, `finetune_level: 2` |
| LSTM / RNN | `precision_level: 3` (dynamic quantization) |
| Extreme accuracy sensitivity | `precision_level: 4` (FP16) |

## Example Configuration

```json
{
    "quantization_parameters": {
        "precision_level": 1,
        "finetune_level": 2
    }
}
```

## ✅ Skill Check

1. You have a ViT (Vision Transformer) model with moderate accuracy requirements. Which `precision_level` and `finetune_level` would you start with?
2. You need the fastest possible quantization iteration for debugging. Which `finetune_level` do you choose?

<details>
<summary>Answers</summary>

1. `precision_level: 1`, `finetune_level: 2` — recommended for Transformer models with Attention.
2. `finetune_level: 0` — no parameter adjustment, fastest runtime.

</details>

---

[← Step 3 – Calibration Dataset Size](step3.md) | [Next: Step 5 – Read the Graphwise Analysis Report →](step5.md)
