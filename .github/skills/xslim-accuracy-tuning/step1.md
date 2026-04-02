# Step 1: Choosing a Quantization Strategy

[← Back to README](README.md) | [Next: Step 2 →](step2.md)

---

## Overview

XSlim supports three quantization strategies. Try them in the following priority order:

| Priority | Strategy | `precision_level` | Notes |
|:---:|---|:---:|---|
| 1 | **Static INT8 quantization** | `0`–`2` | Weights and activations are both quantized statically. Highest compression, fastest inference. **Start here.** |
| 2 | **Dynamic quantization** | `3` | Weights are quantized statically; activations are quantized at runtime. No calibration data required. Suitable for models with unstable activation distributions (e.g., certain NLP models). |
| 3 | **FP16 conversion** | `4` | All floating-point operators are converted to FP16. No calibration data required. Minimal accuracy loss but lower compression and speedup than INT8. |

> **Recommendation:** Fully tune static INT8 quantization first. Only consider dynamic quantization or FP16 when INT8 accuracy cannot meet your requirements.

## Quick Comparison

| Strategy | Calibration Data | Accuracy Risk | Inference Speedup | Typical Use Case |
|---|:---:|:---:|:---:|---|
| Static INT8 | ✅ Required | Medium | ⭐⭐⭐⭐ | Image classification, detection, most CNNs |
| Dynamic quantization | ❌ Not required | Low–Medium | ⭐⭐⭐ | LSTM, some Transformers |
| FP16 | ❌ Not required | Very low | ⭐⭐ | Accuracy-critical models, no calibration data available |

## Tuning Workflow Overview

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

## ✅ Skill Check

Before moving on, answer these questions:

1. Which `precision_level` value enables FP16 conversion?
2. Which strategy requires no calibration data?
3. For a standard ResNet image classifier, which strategy should you try first?

<details>
<summary>Answers</summary>

1. `precision_level: 4`
2. Dynamic quantization (`precision_level: 3`) and FP16 (`precision_level: 4`)
3. Static INT8 (`precision_level: 0`)

</details>

---

[← Back to README](README.md) | [Next: Step 2 – Check Input Preprocessing →](step2.md)
