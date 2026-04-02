# Strategy Selection Reference

Use this file when the task is choosing between static INT8, dynamic quantization, and FP16.

## Priority Order

Try XSlim strategies in this order unless the deployment scenario already rules one out:

| Priority | Strategy | `precision_level` | Notes |
|:---:|---|:---:|---|
| 1 | Static INT8 quantization | `0`-`2` | Highest compression and best acceleration. Start here for most models. |
| 2 | Dynamic quantization | `3` | No calibration data required. Useful for unstable activation distributions. |
| 3 | FP16 conversion | `4` | Lowest accuracy risk among fallback options, but weaker compression/speedup than INT8. |

## Quick Comparison

| Strategy | Calibration Data | Accuracy Risk | Typical Use Case |
|---|:---:|:---:|---|
| Static INT8 | Required | Medium | CNN classification, detection, most standard deployment cases |
| Dynamic quantization | Not required | Low-Medium | LSTM, some Transformer/NLP workloads |
| FP16 | Not required | Very low | Accuracy-critical workloads or no calibration data |

## Recommended Starting Points

| Model Type | Recommended Start |
|---|---|
| CNN image classification | `precision_level: 0` |
| Detection model | `precision_level: 1` |
| Transformer / BERT / ViT | `precision_level: 1` or `2` |
| LSTM / RNN | `precision_level: 3` |
| Extremely accuracy-sensitive model | `precision_level: 4` |

## Escalation Rule

Escalate in this order:

1. fully verify preprocessing
2. improve calibration quality
3. try `precision_level: 1`
4. try `precision_level: 2`
5. switch to `precision_level: 3` or `4`

Avoid jumping directly to FP16 before checking the more common INT8 failure modes.
