---
name: quantization-dev
description: 修改 XSlim 量化流程的落点与流程：precision 分支、静态/动态/FP16 实现、导出器、配置 schema、CLI 与大模型外部数据
---

# Quantization Development

Landing points for changing quantization behavior. Public contracts and their
compatibility rules are owned by `AGENTS.md` Rule 15; this skill maps where each
behavior is implemented.

## 1. Pipeline map

`xslim.quantize_onnx_model(...)` in `src/xslim/xslim_pipeline.py`:

| condition (`PrecisionLevel`) | path |
|---|---|
| `>= 100` | `format_onnx_model` only (simplify) |
| `4` | `convert_to_fp16_onnx_model` (`quantizer/convert_to_fp16.py`) |
| `3` | `dynamic_quantize_onnx_model` (`quantizer/dynamic_quantizer.py`) |
| `0..2` | static INT8: `OnnxParser` → `GraphLegalized` → `dispatch_graph` → calibration dataset → `XSlimQuantizer.quantize` → `ONNXRUNTIMExporter().export` |

Shared stages:

- `format_onnx_model` (`src/xslim/onnx_graph_helper.py`) clears `value_info`,
  normalizes opset, runs `optimize_onnx_model` and shape inference.
- `truncate_onnx_model` / `merge_onnx_model` split large graphs for calibration
  and re-attach them before export.
- Saved `FunctionProto` definitions are re-appended after quantization.
- Output is saved to `<working_dir>/<output_prefix>.onnx`; models above the
  protobuf limit use external data (`is_large_model`).

Do not add a new branch outside this map; extend `PrecisionLevel` only as an
explicit public-interface change (Rule 15).

## 2. FP16

`src/xslim/quantizer/convert_to_fp16.py`:

- `convert_to_fp16_onnx_model` — entry point; selects the normal or
  `_convert_float16_large_model` path for >2GB models.
- `legalize_fp16_graph` — repairs mixed fp16/fp32 inputs, `Cast`, `Range`,
  `Resize`/`Upsample` patterns after conversion.

`src/xslim/quantizer/fp16_legalize.py`:

- `register_extension_op_schemas` — gives schema-less private operators
  (`SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`) a
  registered schema so shape/type inference works.
- `convert_microsoft_ops_to_ai_onnx` — moves an operator to the default domain
  when ai.onnx defines it (`_MS_OPERATORS_IN_AI_ONNX`).
- `repair_fp16_dtype_mismatches` — final type repair before strict inference.

Rules:

- Keep the operator's name and domain when no ai.onnx equivalent exists; describe
  it with an extension schema instead of rewriting the model.
- Do not special-case a model name or node name.
- After conversion, verify `onnx.shape_inference.infer_shapes(..., strict_mode=True,
  check_type=True)` succeeds when the test can do so.

## 3. Dynamic quantization

`src/xslim/quantizer/dynamic_quantizer.py`:

- `dynamic_quantize_onnx_model` — graph-level entry point.
- `dynamic_weight_only_quantize` — weight-only MatMul quantization; keep channel
  order and per-channel axis consistent with the ONNX Runtime convention.

`ignore_op_types` / `ignore_op_names` are applied here and in FP16 conversion; a
new exclusion rule must behave the same in both paths unless documented otherwise.

## 4. Static INT8

- `src/xslim/quantizer/xslim.py` (`XSlimQuantizer`) orchestrates graph search,
  calibration, dispatch and export.
- `src/xslim/optimizer/` holds the passes (`GraphLegalized`, fusion, refine,
  equalization, observer, calibration, training).
- `src/xslim/analyse/graphwise.py` produces the Graphwise Analysis report used by
  the accuracy workflow.
- `src/xslim/ppq_decorator/ppq/parser/`, `IR/`, `executor/`, `scheduler/` are the
  vendored PPQ adaptation layer; change them only when the behavior cannot live
  in `xslim/`.

Static quantization rejects input models that already contain
`QuantizeLinear`/`DequantizeLinear` (`_check_static_quantization_input_model`).
Keep that guard and its error message contract.

## 5. Config and CLI

- Schema: `src/xslim/xslim_setting.py` (`ModelParameterSetting`,
  `InputParameterSetting`, `CalibrationParameterSetting`,
  `QuantizationParameterSetting`, `CustomQuantizationParameterSetting`).
- Validation: each `check()` raises on invalid values; `from_json` ignores unknown
  keys, so a new key must be added to the right settings class, not read ad hoc.
- CLI: `src/xslim/__main__.py` — config-free `--fp16`, `--dynq`, `--opset`,
  `--ignore_op_types`, `--ignore_op_names`.
- Public exports: `src/xslim/__init__.py`.
- Constants and feature gates: `src/xslim/defs.py`
  (`MIN_ONNX_OPSET_VERSION`, `resolve_operator_domain`, `PrecisionLevel`).

Adding a key requires: default value, `check()` validation, documentation in both
languages, and a test that exercises the key through `quantize_onnx_model`.

## 6. Large models

- Detect with `is_large_model`; save with `save_as_external_data` and a single
  `.onnx.data` file.
- FP16 conversion uses disk-based shape inference to avoid the protobuf 2GB
  serialization limit; do not reintroduce an in-memory `SerializeToString` on
  that path.
- Tests must build small models; large-model behavior is verified by targeted
  unit tests or an explicit external-model run, never by committing weights.

## 7. Tests

- `tests/test_convert_to_fp16.py`, `tests/test_fp16_domain_legalize.py` — FP16.
- `tests/test_dynamic_quantizer.py` — dynamic.
- `tests/test_quantize_pipeline.py` — end-to-end flow and static guard.
- `tests/test_onnx_operator_support.py`, `tests/test_global_functions_mapping.py`
  — opset/domain and FunctionProto export.
- Run targeted modules per `.agents/skills/build-verify/SKILL.md`.

## 8. Documentation (Rule 3, mandatory)

Configuration/CLI → `doc/configuration.md` + `_zh`; examples →
`doc/examples.md` + `_zh`; accuracy strategy → `doc/accuracy_tuning.md` + `_zh`;
public API/install/features → `README.md` + `README_zh.md`. A breaking change also
bumps `VERSION_NUMBER` and adds a migration note.

## Principles

- Evidence first: conversion success, executability and numerical accuracy are
  three different claims; state which one was measured.
- Fix the rule, not the model (Rule 4).
