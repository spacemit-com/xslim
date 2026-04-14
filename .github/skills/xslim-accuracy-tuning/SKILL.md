---
name: XSlim Accuracy Tuning
description: This skill should be used when the user asks to "调优 xslim 精度", "优化量化精度", "分析量化后精度下降", "解读 Graphwise Analysis 报告", "设置 precision_level", "设置 finetune_level", "improve XSlim quantization accuracy", "tune XSlim accuracy", "interpret XSlim Graphwise Analysis", or "fix accuracy drop after quantization".
version: 0.1.0
---

# XSlim Accuracy Tuning

Use this skill to diagnose and improve accuracy after XSlim quantization. Focus on reusable tuning workflow, parameter selection, and report interpretation. Keep the main reasoning in this file and load reference files only when a task needs deeper detail.

When the user asks to bootstrap work from an ONNX file, prefer the bundled script `scripts/bootstrap_xslim_project.py` instead of manually drafting the config and demo files. Use the script to inspect model inputs, choose a starter quantization profile, and scaffold a runnable demo project.

## Overview

XSlim accuracy tuning usually follows the same sequence:

1. Confirm the quantization strategy is appropriate.
2. Verify calibration preprocessing matches training and inference exactly.
3. Check whether calibration data is representative and large enough.
4. Adjust `precision_level` and `finetune_level` in a controlled order.
5. Read the Graphwise Analysis report to locate high-error operators.
6. Apply targeted fixes such as `custom_setting`, calibration changes, or operator exclusion.

Treat this as a narrowing workflow. Start from global causes that affect the whole model, then move to local fixes for specific layers or subgraphs.

## When to Use This Skill

Use this skill when working on any XSlim task involving:

- accuracy drop after quantization
- reading a user's ONNX file and generating a starter XSlim project
- automatic generation of quantization configs or demo scaffolding
- strategy selection between static INT8, dynamic quantization, and FP16
- `precision_level` / `finetune_level` tuning
- preprocessing mismatch checks
- calibration dataset sizing
- Graphwise Analysis report interpretation
- local tuning with `custom_setting`, `ignore_op_types`, or `ignore_op_names`

Do not use this skill for generic ONNX graph editing, packaging changes, or unrelated repository documentation work.

## Core Workflow

Follow this sequence unless the user already narrowed the problem to a later step.

### 0. Bootstrap repetitive setup work with the bundled script

If the user needs a starting project rather than a diagnosis only, run:

`scripts/bootstrap_xslim_project.py`

Use it to:

- inspect the ONNX model inputs
- infer a starter task profile (image-style vs token/npy-style)
- generate a quantization config
- scaffold `demo.py`, `demo.sh`, `README.md`, and calibration list placeholders

Load `references/scripted-automation.md` when the task is about auto-generating configs, project scaffolding, or reducing repeated manual setup.

### 1. Confirm the quantization strategy

Start from static INT8 unless there is a strong reason not to. Escalate only when the lower-precision option cannot meet the accuracy requirement.

- Use `precision_level: 0` as the default baseline for common CNN models.
- Move to `precision_level: 1` or `2` for more accuracy-sensitive models or Transformer-style workloads.
- Use `precision_level: 3` for dynamic quantization when activation distributions are unstable or calibration data is unavailable.
- Use `precision_level: 4` for FP16 when the model is highly accuracy-sensitive.

Load `references/strategy-selection.md` when the task needs a strategy comparison or model-type recommendations.

### 2. Verify preprocessing before changing quantization knobs

Treat preprocessing mismatch as the first likely root cause when the accuracy drop is unexpectedly large. Confirm:

- pixel value range
- `mean_value` and `std_value`
- `color_format`
- resize / crop behavior
- input dtype
- preprocessing preset or custom preprocessing function

Load `references/preprocessing-and-calibration.md` when the task needs concrete checks or examples for `input_parameters`.

### 3. Validate calibration data quantity and quality

Check whether the calibration dataset is both large enough and representative enough. Prefer real deployment-like samples over synthetic or repetitive samples.

- For quick validation, use roughly 50-100 samples.
- For general CNN tuning, use roughly 100-500 samples.
- For complex models such as detection or Transformers, use roughly 500-1000 samples.
- For high-accuracy tuning, use 1000+ samples when practical.

Remember that `calibration_step` counts dataloader iterations rather than raw file count, so effective sample count depends on batch size.

Load `references/preprocessing-and-calibration.md` for calibration sizing and warning signs of insufficient data.

### 4. Tune global quantization parameters in order

Adjust parameters systematically rather than changing multiple unrelated settings at once.

Recommended order:

1. keep preprocessing fixed
2. increase calibration quality if needed
3. try `precision_level: 0 -> 1 -> 2`
4. increase `finetune_level: 1 -> 2 -> 3` if broad model quality still lags
5. switch to dynamic or FP16 only if INT8 tuning is still unacceptable

Use `finetune_level: 1` as the general default. Raise it when the model shows moderate error across many layers rather than one obvious hotspot.

Load `references/precision-and-finetune.md` for recommended combinations by model type and goal.

### 5. Read the Graphwise Analysis report before applying local fixes

When `analysis_enable: true`, XSlim generates a Markdown report sorted by quantization error. Use the report to decide whether the issue is global or localized.

Prioritize operators with:

- `SNR >= 0.1`
- `Cosine < 0.99`
- obvious gaps between `F.MinMax` and `Q.MinMax`
- suspiciously sparse or skewed `F.Hist`

Interpretation rules:

- high SNR + low Cosine on one layer usually means a local hotspot
- moderate error across many layers often points to a global calibration or precision issue
- much narrower quantized range than float range usually indicates range clipping

Load `references/graphwise-analysis.md` for metric thresholds and a reading checklist.

### 6. Apply targeted fixes only after identifying the failure pattern

Choose the local fix that matches the report evidence:

- raise local precision with `custom_setting`
- change calibration type or percentile settings for clipped activations
- exclude sensitive operator types or operator names from quantization
- raise `finetune_level` when many layers show moderate error

Prefer the smallest targeted adjustment that resolves the problem. Avoid overusing exclusions or broad FP16 fallback if a subgraph-level fix is enough.

Load `references/targeted-tuning.md` when the task needs concrete config patterns.

## Decision Heuristics

Apply these heuristics during analysis:

- Large immediate drop after quantization: suspect preprocessing mismatch first.
- Flaky or sample-sensitive results: suspect calibration data quantity or representativeness.
- Small but persistent global drop: try higher `precision_level` or `finetune_level`.
- One or two red layers in Graphwise report: use `custom_setting` or local calibration changes.
- Repeated issues on `Softmax` or `LayerNormalization`: consider exclusion from quantization.
- INT8 remains unacceptable after structured tuning: recommend dynamic quantization or FP16.

## Output Expectations

When using this skill in a user-facing task:

1. State the likely cause category first.
2. Recommend the next smallest set of changes.
3. Distinguish global tuning from local tuning.
4. Reference concrete XSlim parameters by name.
5. Mention when deeper detail comes from a reference file.

For implementation tasks, keep repository edits aligned with existing XSlim docs and config naming, including `input_parameters`.

## Additional Resources

Load these files only when needed:

- **`references/scripted-automation.md`** - how to use the bundled bootstrap script and what it generates
- **`references/strategy-selection.md`** - strategy comparison and recommended model-to-strategy mapping
- **`references/preprocessing-and-calibration.md`** - preprocessing checklist, calibration sample guidance, common mistakes
- **`references/precision-and-finetune.md`** - `precision_level` / `finetune_level` tuning order and recommended combinations
- **`references/graphwise-analysis.md`** - Graphwise Analysis metrics, thresholds, and interpretation
- **`references/targeted-tuning.md`** - concrete targeted tuning methods and example configuration patterns

Primary repository docs:

- **`doc/accuracy_tuning.md`**
- **`doc/accuracy_tuning_zh.md`**
- **`doc/configuration.md`**
- **`doc/configuration_zh.md`**

## Quality Standard

Keep the skill body focused on process and decision-making. Put detailed tables, thresholds, and examples in `references/`. When extending this skill later, add new detailed material to `references/` first and only keep the reusable workflow in `SKILL.md`.
