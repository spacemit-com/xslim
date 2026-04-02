# Step 5: Read the Graphwise Analysis Report

[← Step 4](step4.md) | [Next: Step 6 →](step6.md)

---

## Overview

After quantization, XSlim automatically runs a Graphwise Analysis (controlled by `analysis_enable: true`) and writes a report to the working directory as `<output_prefix>_report.md`. The report is a Markdown table **sorted by quantization error from highest to lowest**.

## Enable the Analysis Report

```json
{
    "quantization_parameters": {
        "analysis_enable": true
    }
}
```

## Report Structure

Each row corresponds to a single tensor (input or output) of one operator:

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

## Key Metric Interpretation

### SNR (Signal-to-Noise Ratio)

SNR = energy of quantization error / energy of original signal. **Higher SNR means larger quantization error.**

- `SNR < 0.01`: Negligible error — no action needed
- `0.01 ≤ SNR < 0.1`: Acceptable error range
- `SNR ≥ 0.1` (highlighted red): High error — this operator is a priority tuning target

### Cosine Similarity

Measures directional alignment between quantized and float values.

- `Cosine > 0.999`: Excellent alignment — quantization quality is good
- `0.99 ≤ Cosine ≤ 0.999`: Slight deviation
- `Cosine < 0.99` (highlighted red): Significant deviation — this operator requires attention

### F.MinMax vs. Q.MinMax

A large gap (e.g., float range `[-100, 100]` but quantized range `[-5, 5]`) indicates **range clipping**, which causes information loss and accuracy degradation. Try adjusting `calibration_type` (e.g., switch to `minmax`) or increasing `max_percentile` (e.g., from `0.9999` to `0.99999`).

### F.Hist (Value Distribution Histogram)

- Heavily skewed histogram (most values in just a few bins) → outliers in activation values → try `percentile` calibration with a lower `max_percentile`
- Very sparse histogram (many bins are zero) → insufficient calibration data → increase `calibration_step`

## Example Report Interpretation

```markdown
| Op | Var | SNR | MSE | Cosine | Q.MinMax | F.MinMax | F.Hist |
|---|---|---|---|---|---|---|---|
| /model/layer4/conv[Conv] | /layer4/conv/output | <font color="red">0.1523</font> | 0.0312 | <font color="red">0.9743</font> | -3.21, 3.18 | -15.32, 12.87 | 0,0,2,8,45,... |
| /model/layer1/conv[Conv] | /layer1/conv/output | 0.0045 | 0.0003 | 0.9998 | -1.02, 1.05 | -1.05, 1.08 | 3,12,55,... |
```

Reading this example:
- `layer4/conv` has SNR `0.1523` (red) and Cosine `0.9743` (red) — **this is the primary source of accuracy loss**
- The float range `[-15.32, 12.87]` is far wider than the quantized range `[-3.21, 3.18]`, indicating severe range clipping
- `layer1/conv` has healthy metrics — no action needed

## ✅ Skill Check

1. An operator in the report has `SNR = 0.15`. Is this acceptable? What action should you take?
2. A layer shows `F.MinMax = [-50, 48]` but `Q.MinMax = [-3.2, 3.1]`. What problem does this indicate?

<details>
<summary>Answers</summary>

1. No, `SNR ≥ 0.1` is highlighted red and means high quantization error. This operator is a priority tuning target — proceed to targeted tuning in Step 6.
2. Severe range clipping — the quantized range covers only a small fraction of the actual float range. Try adjusting `calibration_type` to `"minmax"` or increasing `max_percentile`.

</details>

---

[← Step 4 – Adjust precision_level and finetune_level](step4.md) | [Next: Step 6 – Targeted Tuning →](step6.md)
