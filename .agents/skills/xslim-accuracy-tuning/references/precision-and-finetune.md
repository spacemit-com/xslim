# Precision and Finetune Reference

Use this file when the task is selecting or adjusting `precision_level` and `finetune_level`.

## `precision_level`

| Level | Meaning | Typical Use Case |
|:---:|---|---|
| `0` | Full INT8 | Most CNNs |
| `1` | Partial INT8 with a few sensitive regions kept higher precision | Detection, attention-based models |
| `2` | More conservative partial INT8 | Accuracy-sensitive or complex Transformer models |
| `3` | Dynamic quantization | Unstable activations or no calibration data |
| `4` | FP16 conversion | Highest accuracy priority |

## `finetune_level`

| Level | Meaning | Runtime | Typical Use Case |
|:---:|---|:---:|---|
| `0` | No parameter adjustment | Fastest | Quick iteration |
| `1` | Lightweight static calibration | Short | General default |
| `2` | Block-wise calibration | Medium | Accuracy-sensitive models, many Transformers |
| `3` | More aggressive block-wise calibration | Long | Best accuracy-oriented tuning |

## Recommended Combinations

| Goal / Model Type | Recommended Settings |
|---|---|
| Maximum compression | `precision_level: 0`, `finetune_level: 1` |
| CNN classification | `precision_level: 0`, `finetune_level: 1` |
| Detection | `precision_level: 1`, `finetune_level: 2` |
| Transformer / BERT / ViT | `precision_level: 1` or `2`, `finetune_level: 2` |
| Best INT8 accuracy | `precision_level: 2`, `finetune_level: 3`, higher `calibration_step` |
| No calibration data | `precision_level: 3` or `4` |
| Fastest debug iteration | `precision_level: 0`, `finetune_level: 0`, low `calibration_step` |

## Tuning Order

Apply these changes incrementally:

1. keep `precision_level: 0`
2. if needed, raise to `1`
3. if still needed, raise to `2`
4. independently raise `finetune_level` when many layers show broad moderate error
5. only then choose dynamic or FP16 fallback

## Interpretation Rule

- If a few layers are bad: prefer local fixes first.
- If many layers are mildly degraded: raise `finetune_level`.
- If the entire model remains too sensitive at higher INT8 settings: move to dynamic or FP16.
