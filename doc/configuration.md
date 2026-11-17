# Configuration Reference

[中文版](configuration_zh.md) | English

XSlim is configured through a JSON file with three top-level sections. All fields are **optional** unless marked otherwise.

## Top-level Structure

```json
{
    "model_parameters": { ... },
    "calibration_parameters": { ... },
    "quantization_parameters": { ... }
}
```

---

## `model_parameters`

Controls input/output paths and model pre-processing.

| Field | Type | Default | Description |
|---|---|---|---|
| `onnx_model` | `string` | — | Path to the input ONNX model file |
| `output_prefix` | `string` | Model filename stem (output appends `.q.onnx`) | Prefix for the output file name |
| `working_dir` | `string` | Directory containing `onnx_model` | Directory for the output model and intermediate files |
| `skip_onnxsim` | `bool` | `false` | Set `true` to skip ONNX model simplification before quantization |
| `opset` | `int` | Keep the model's default `ai.onnx` opset | Convert the default `ai.onnx` opset to a target version during simplify / quantize / export |

### Example

```json
"model_parameters": {
    "onnx_model": "models/resnet18.onnx",
    "output_prefix": "resnet18_int8",
    "working_dir": "./output"
}
```

> `opset` is the JSON equivalent of the CLI `--opset` flag. Use it when your target runtime requires a specific default ai.onnx opset version.

---

## `calibration_parameters`

Controls how calibration data is loaded and how quantization ranges are computed.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `calibration_step` | `int` | `500` | 10–1000 | Maximum number of calibration steps (dataloader iterations/batches); effective sample count ≈ `calibration_step × batch_size` |
| `calibration_batch_size` | `int` | `1` | ≥ `1` | Number of samples loaded per calibration batch; effective sample count ≈ `calibration_step × calibration_batch_size` |
| `calibration_device` | `string` | `cuda` | `cuda`, `cpu` | Inference device for calibration; auto-detected, falls back to `cpu` |
| `auto_calibration_methods` | `string[]` | `["minmax", "percentile", "mse", "kl"]` | Non-empty, distinct subset of these methods | Candidates for `auto`; order breaks equal-MSE ties |
| `auto_calibration_samples` | `int` | `8192` | Positive integer | Maximum sampled activation elements per quantization domain for scoring |
| `calibration_type` | `string` | `default` | `default`, `kl`, `minmax`, `percentile`, `mse`, `auto` | Observer algorithm for computing activation ranges |
| `input_parameters` | `list` | **required** | — | Per-input settings, one entry per model input (see below) |

### Calibration Type Details

| Value | Description |
|---|---|
| `default` | Chip-recommended algorithm (typically `kl` or `percentile`) |
| `kl` | KL-divergence minimization |
| `minmax` | Uses the observed minimum and maximum values |
| `percentile` | Clips activations at `max_percentile` to suppress outliers |
| `mse` | Minimizes mean squared error between original and quantized activations |
| `auto` | Select a calibration method per activation quantization domain by sampled Q/DQ MSE on user data (static INT8 only) |

> **Tip:** Start with `default`. If accuracy is insufficient, try `percentile` or `minmax`.

### `input_parameters` (per input)

Each list entry corresponds to one model input **in the same order as the ONNX model's input list**.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `input_name` | `string` | Read from model | — | Input tensor name |
| `input_shape` | `list[int]` | Read from model | — | Input shape; symbolic batch dimension defaults to `1` |
| `dtype` | `string` | `float32`, then read from model during graph loading when available | Any ONNX tensor dtype | Input data type. Non-float inputs such as `int64` are supported for `npy` calibration data; `img` and `raw` loaders interpret data as `float32` |
| `file_type` | `string` | `img` | `img`, `npy`, `raw` | Calibration file format (see below) |
| `color_format` | `string` | `bgr` | `rgb`, `bgr` | Color channel order for image inputs |
| `mean_value` | `list[float]` | `null` | — | Per-channel mean subtracted during normalization |
| `std_value` | `list[float]` | `null` | — | Per-channel standard deviation used to divide during normalization |
| `preprocess_file` | `string` | `null` | `PT_IMAGENET`, `IMAGENET`, or a custom path | Preprocessing function (see [Custom Preprocessing](#custom-preprocessing)) |
| `data_list_path` | `string` | **required** | — | Path to a text file listing calibration data files, one path per line |

#### `file_type` Details

| Value | Description |
|---|---|
| `img` | Standard image file (JPEG, PNG, BMP, etc.) read with OpenCV |
| `npy` | NumPy `.npy` file containing a single array |
| `raw` | Raw binary file; must contain `float32` data matching `input_shape` |

### Calibration Data List File

`data_list_path` points to a plain-text file with **one file path per line**. Paths may be absolute or relative to the directory containing the list file.

```text
data/calib/ILSVRC2012_val_00002138.JPEG
data/calib/ILSVRC2012_val_00000994.JPEG
data/calib/ILSVRC2012_val_00014467.JPEG
```

For multi-input models, each `input_parameters` entry has its own list file. Files at the same line number across all lists form one calibration batch — the lists must therefore have equal length and consistent ordering.

### Custom Preprocessing

Set `preprocess_file` to one of:

- `"PT_IMAGENET"` – Built-in PyTorch-style ImageNet preprocessing (resize → center-crop → normalize)
- `"IMAGENET"` – Built-in standard ImageNet preprocessing
- `"path/to/script.py:function_name"` – A function in a user-provided Python file

Custom preprocessing functions must have the following signature:

```python
from typing import Sequence
import torch

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Args:
        path_list: List of file paths for one calibration batch.
        input_parametr: The corresponding entry from calibration_parameters.input_parameters.

    Returns:
        A batched torch.Tensor of shape [batch, C, H, W].
    """
    ...
```

Full example:

```python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    batch_list = []
    mean_value = input_parametr["mean_value"]
    std_value = input_parametr["std_value"]
    input_shape = input_parametr["input_shape"]
    for file_path in path_list:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (input_shape[-1], input_shape[-2]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = (img - mean_value) / std_value
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        batch_list.append(img)
    return torch.cat(batch_list, dim=0)
```

For multi-input models where preprocessing is similar across inputs, the same function can be reused for each entry.

---

## `quantization_parameters`

Controls the quantization strategy and precision. All fields in this section are optional.

| Field | Type | Default | Options | Description |
|---|---|---|---|---|
| `precision_level` | `int` | `0` | `0`–`4` | Global precision level (see below) |
| `finetune_level` | `int` | `1` | `0`–`3` | Aggressiveness of calibration parameter tuning (see below) |
| `analysis_enable` | `bool` | `true` | — | Run post-quantization accuracy analysis |
| `max_percentile` | `float` | `0.9999` | ≥ `0.99` | Percentile clipping threshold (applies when `calibration_type` is `percentile`) |
| `custom_setting` | `list` | `null` | — | Per-subgraph precision overrides (see below) |
| `truncate_var_names` | `list[string]` | `[]` | — | Tensor names used to split the graph into two parts (see below) |
| `ignore_op_types` | `list[string]` | `[]` | — | ONNX operator types to exclude from quantization |
| `ignore_op_names` | `list[string]` | `[]` | — | Specific operator names to exclude from quantization |

### Precision Levels

| Level | Description |
|---|---|
| `0` | Full INT8 quantization (default). Maximum compression; quantization tuning stays within INT8 |
| `1` | Partial INT8 – some sensitive operators are kept at higher precision. Suitable for general Transformer models |
| `2` | Partial INT8 with the most operators kept at higher precision. Best for accuracy-sensitive models |
| `3` | Dynamic quantization – weights are statically quantized; activations are quantized at runtime |
| `4` | FP16 conversion – all floating-point ops are cast to FP16 (no calibration data required) |

### Fine-tune Levels

| Level | Description |
|---|---|
| `0` | No calibration parameter adjustment |
| `1` | May apply lightweight static calibration parameter tuning |
| `2` | Block-wise calibration parameter tuning based on quantization loss |
| `3` | More aggressive block-wise tuning; higher quality but slower |

### `custom_setting`

A list of per-subgraph overrides. Each entry selects a contiguous subgraph by its boundary tensors and applies local quantization settings. All tensors between the boundary input operators and the boundary output operators (inclusive) are covered by the override.

| Field | Type | Description |
|---|---|---|
| `input_names` | `list[string]` | Input tensor names that mark the entry edge of the subgraph (constants may be omitted) |
| `output_names` | `list[string]` | Output tensor names that mark the exit edge of the subgraph |
| `precision_level` | `int` | Precision level to apply to this subgraph |
| `calibration_type` | `string` | Calibration algorithm for this subgraph (same values as the global `calibration_type`) |
| `max_percentile` | `float` | Percentile threshold for this subgraph (same semantics as the global `max_percentile`) |

```json
"custom_setting": [
    {
        "input_names": ["input"],
        "output_names": ["input.12"],
        "precision_level": 2,
        "calibration_type": "default"
    }
]
```

### `truncate_var_names`

Specifies a list of tensor names at which the computation graph is **split into exactly two parts**. The split tensors become output tensors of the first subgraph and input tensors of the second. The tool validates the binary split and raises an error if the result is invalid.

> **YOLO note:** XSlim first tries to fuse supported decode-heavy YOLO post-processing into `spacemit_functions.YoloDecode`. Use `truncate_var_names` as a fallback when the export pattern does not match the fusion pass or when you intentionally want to keep post-processing outside the quantized region.

```json
"truncate_var_names": ["/Concat_5_output_0", "/Transpose_6_output_0"]
```

### `ignore_op_types` and `ignore_op_names`

Operators matching any entry in `ignore_op_types` (by ONNX op type) or `ignore_op_names` (by node name) are excluded from quantization and kept at their original precision.

```json
"ignore_op_types": ["LayerNormalization", "Softmax"],
"ignore_op_names": ["/model/encoder/layer.0/attention/MatMul"]
```

---

## ONNX Function Preservation

XSlim preserves existing ONNX `FunctionProto` definitions from the input model and automatically emits the required `spacemit_functions` import when fused nodes such as `YoloDecode` or `BatchMatMul` are present. No extra configuration is required.

---

## Static Quantization Input Requirements

Static INT8 quantization expects a floating-point ONNX input model. Models that already contain `QuantizeLinear` or `DequantizeLinear` are considered already quantized, so XSlim rejects them with a `ValueError` instead of quantizing the graph a second time.

Use the original floating-point model as the input for static INT8. If you only need model simplification, dynamic quantization, or FP16 conversion, invoke the corresponding CLI mode explicitly.

---

## Operator Coverage Notes

XSlim 2.1.0 expands the internal ONNX executor and quantization socket metadata used by analysis and calibration. The updated coverage includes:

- opset-24 `Pad` with the optional `axes` input;
- comparison and logical operators such as `GreaterOrEqual`, `LessOrEqual`, and `Xor`;
- activation and unary operators such as `Celu`, `Hardmax`, `Mish`, `Softsign`, `ThresholdedRelu`, trigonometric, inverse-trigonometric, hyperbolic, `Round`, `Sign`, `IsInf`, and `IsNaN`;
- additional reduce kernels such as `ReduceL1`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceMin`, `ReduceProd`, and `ReduceSumSquare`, including scalar tensors and axes supplied as inputs.

When a model contains an unsupported ONNX operator, XSlim still reports it during graph loading, simplification, analysis, or quantization so you can decide whether to simplify/export the model differently or exclude the affected subgraph.

---

## Complete Configuration Example

```json
{
    "model_parameters": {
        "onnx_model": "models/my_model.onnx",
        "working_dir": "./output"
    },
    "calibration_parameters": {
        "calibration_batch_size": 1,
        "calibration_step": 200,
        "calibration_device": "cuda",
        "calibration_type": "default",
        "input_parameters": [
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
        "precision_level": 1,
        "finetune_level": 2,
        "analysis_enable": true,
        "ignore_op_types": ["Softmax"],
        "custom_setting": [
            {
                "input_names": ["input"],
                "output_names": ["stem_output"],
                "precision_level": 0
            }
        ]
    }
}
```

## Operator quantization rules (static INT8)

`quantization_parameters.operator_quantization` defaults to `[]`. Each entry has
`pattern` (non-empty string), `precision` (`int8` or `fp32`), and optional
`calibration_type` (`default`, `minmax`, `percentile`, `kl`, `mse`, `auto`, INT8 only).
Patterns match optimized PPQ operation names, case-sensitively, using Python
`fnmatchcase`: `*`, `?`, and `[abc]` are supported. The last matching rule wins.
Rules override `ignore_op_names`, `ignore_op_types`, and precision-level/subgraph
precision refinement. An explicit calibration method also overrides global and
subgraph calibration methods. Operator socket constraints (shape/index inputs,
FP32 bias) remain in effect.

Use static `precision_level` 0, 1 or 2. FP16/dynamic modes reject these rules.
An unmatched pattern or explicit INT8 on an unsupported/shape-only operation
raises `ValueError`. Names refer to the graph **after** preprocessing/fusion;
use `skip_onnxsim` when simplification changes names you need to address.
Existing exact-name exclusion settings keep their behavior when rules are empty.

### Automatic layerwise calibration

Set `calibration_parameters.calibration_type` to `"auto"` for static INT8
(`precision_level` 0, 1 or 2). Each activation quantization domain evaluates the
configured candidates using the supplied calibration data, and keeps the method
with the smallest sampled quantize/dequantize reconstruction MSE. Weights retain
the existing calibration behavior. Default configuration remains `"default"`.

The fixed-seed uniform streaming sample buffer is bounded by
`auto_calibration_samples`, samples across calibration batches without batch-split
bias, and is released after selection. Candidate calibrators still
observe the full calibration tensors and use the existing histogram passes;
additional candidates increase calibration time and memory. Selection is local
to activation tensors, not an end-to-end task-accuracy or latency search, and
does not automatically switch bit width or choose FP16/FP32.

`operator_quantization[].calibration_type` can pin a method or opt selected
operators into `"auto"`. Explicit methods apply to the effective shared
quantization domain after fusion/simplification, including adjacent operators
sharing its scale. Conflicting explicit methods in one domain raise `ValueError`.
Logs record each selected method, sampled MSE and sample count.

### Static precision strategy

The optional top-level `strategy_parameters` uses `mode: "standard"`. Defaults are
`allowed_precisions: ["int8", "fp32"]`, `default_precision: null`, empty
`node_precisions` / `op_type_precisions` maps and an empty `rules` list. These defaults
keep legacy dispatch unchanged. Changing allowed precisions also enables validation;
implicit decisions outside that list fail. Active strategies require static precision
levels 0–2. Controlled local FP16 and `search` are described below; the default remains unchanged.

Rules use `{"selector":"node","names":["head"],"precision":"fp32"}`,
`{"selector":"type","op_types":["MatMul"],"precision":"int8"}`, or
`{"selector":"subgraph","start_nodes":["a"],"end_nodes":["b"],"precision":"fp32"}`.
Node rules take priority over subgraph, type and default rules, in that order.
Subgraphs contain only nodes on directed paths between inclusive boundaries.
Conflicting rules at the same priority fail. Effective legacy
`operator_quantization` rules retain last-match ordering and join at node priority;
they override ignore lists, and conflicting new node rules fail.

Targets refer to unique, non-empty original node names, captured before formatting.
After prequantization, mapping requires matching operator type and output tensors;
renames are recorded, and missing or ambiguous selected targets fail. Skip-simplify
does not guarantee name preservation. A global INT8 default applies only to supported
data operators. Explicit unsupported INT8 targets fail.

### Automatic mixed precision (static only)

Set `strategy_parameters.mode` to `"auto_mixed"` to promote unlocked atomic INT8
groups to FP32 using real exported QDQ models and ONNX Runtime outputs. The default
mode remains `"standard"`. Existing rules and locks apply to the initial legal plan.

| Flat strategy field | Default | Validation / meaning |
|---|---|---|
| `score_steps` | `8` | Positive integer; maximum scoring batches |
| `max_trials` | `8` | Positive integer; total full conversions, including baseline, failed attempts and final check |
| `max_adjust_groups` | `2` | Nonnegative integer; maximum accepted group promotions |
| `max_time_s` | `600.0` | Finite positive number; includes preparation, hashing, calibration and scoring |
| `cosine_similarity_min` | `0.9999` | Finite number in `[-1, 1]`; early-stop target |

Booleans are rejected for these numeric fields. Every input must provide real
`data_list_path` data; implicit zero inputs are rejected. The same frozen,
preprocessed batches score the original float model and each candidate. Zero norms,
nonfinite values, empty outputs and shape mismatches invalidate metrics.

Candidate enumeration reserves one full conversion for an independent final export.
Synchronous preparation/conversion may finish after the time limit; no new candidate
starts after the deadline, and `time_overrun` records this. Cache hits do not consume
conversion budget. Reusing `<output_prefix>_candidates` resumes counts and elapsed
work; a changed model/config/data/runtime context requires a new work directory or
output prefix. Identity includes preprocessing file contents, raw samples, processed
batches, source model and effective precision/calibration/explicit-INT8 semantics.

Outputs include `<output_prefix>_auto_mixed_report.json` and the independent
`<output_prefix>_final_check.json`, with metrics, hashes, layout evidence, conversion
counts, stop and fallback reasons. Final outputs must match the selected candidate
with `rtol=1e-5`, `atol=1e-6`. When selection or final verification fails, the original
legal baseline artifact is retained; a missing/failed final check is never marked
verified. Active strategies currently reject models of 2 GB or larger and scoring
that requires truncated-model inputs. They do not switch to another strategy.

When the Python API receives a model path override, its default output name is
`<input_stem>.q.onnx`. An output resolving to the input file (including a symlink
or hardlink alias) is rejected before conversion.


### Task-metric search and backend evidence

`strategy_parameters.mode: "search"` is opt-in and requires the Python API
`quantize_onnx_model(config, model_or_path, output_path, *, evaluator=...,
measurement_callback=None, measurement_file=None)`. Existing positional arguments,
return type and default `standard` behavior are unchanged. No evaluator means an
explicit error. Callback and offline measurement file are mutually exclusive.

All search policy fields are nested under **`strategy_parameters.search`**:

| Field | Default | Meaning |
|---|---|---|
| `objective` | `"min_error"` | `min_error`, `min_latency`, or `min_memory` |
| `max_metric_drop` | `0.01` | Absolute, direction-aware task metric difference in `[0,1]`, not a relative percentage |
| `stop_on_feasible` | `true` | Stop tuning after a feasible candidate; still perform the final conversion |
| `max_full_trials` | `8` | Total full conversions: baseline, failed attempts and independent final export all count |
| `max_time_s` | `600.0` | Positive finite total elapsed-work budget, including preparation and external callbacks |
| `max_candidates` | `32` | Global limit on cheaply screened candidate plans |
| `prior_full_conversion_count` | `0` | Initial consumed budget; restored counts cannot be reduced |
| `prior_elapsed_s` | `0.0` | Initial consumed elapsed budget; restored elapsed time cannot be reduced |
| `backend` | `null` | Deployment identity described below |
| `constraints` | both limits `null` | `max_latency_ms` (positive number), `max_memory_bytes` (positive integer) |

Unknown search/backend/constraint keys and invalid types are rejected, including
booleans supplied as numbers. Backend objectives or constraints require `backend`:

```json
{
  "hardware": {"id": "board-model", "version": "revision", "options": {}},
  "runtime": {"name": "deployment-runtime", "version": "version", "options": {}},
  "protocol": {"id": "measurement-protocol", "version": "version", "options": {}}
}
```

Each descriptor requires its identity and version; `options` is an optional finite
JSON object. The protocol should describe warmup, repetitions, threading, input
shapes and the definition of peak memory. XSlim cannot independently authenticate
callback/file measurements; verified means matching caller-supplied evidence.
PC conversion duration and model file size are never latency/memory substitutes.
A missing or mismatched measurement remains `pending` and cannot satisfy a backend
objective. A record must contain the objective metric **and every constrained
metric**; a known violated constraint is `failed` even if another metric is missing.

The evaluator protocol is `evaluate(model, *, split) -> {"metric": float,
"sample_count": int}`. Metric must be finite in `[0,1]`, sample count positive and
equal for the reference/candidate on each split. Required attributes are
`metric_name`, `higher_is_better` (bool), `direction` (`maximize`/`minimize`),
`preprocessing_version`, `postprocessing_version`, `tuning_data_id`,
`validation_data_id`, and `fingerprint`. Strings must be nonempty, split IDs must
differ and direction must agree with the boolean. The evaluator owns actual split
contents and must apply identical preprocessing/postprocessing and tuning samples
to the float reference and every candidate. Changing any identity invalidates task
scores, while consumed search budget is retained. Calibration batches and their
raw/processed identities are frozen by CandidateRunner.

A cheap structural exposure proxy orders whole-group promotions to allowed
FP32/FP16; it is reported as a heuristic, never a task score or board measurement.
Only limited full exports are evaluated. Candidate deduplication includes effective
precision/calibration/explicit-INT8 semantics. One conversion is reserved for a fresh
final export; a missing or failed final check can publish the legal baseline but is
never verified. After the final model and plan are frozen, validation runs on the
float reference and selected artifact only. Validation failure never changes the
plan, reranks candidates or feeds back into tuning.

`<prefix>_search_report.json`, `<prefix>_final_check.json`, and the precision report
contain actual layout/coverage denominators, source mapping, decisions/locks/groups,
reference/tuning/validation metrics, verified/pending status, measurement rejections,
budget counts and stop/fallback reasons. For `min_error`, task verification may be
successful while optional backend evidence remains pending. Search resume files
live beside the report and in `<prefix>_search_candidates`; a changed conversion
context fails closed and requires a new work directory/prefix. Context binds FP16
implementation bytes and `onnxconverter-common` version. Only volatile
`xslim_export_time` metadata is excluded from numerical model fingerprints.
Synchronous work cannot be interrupted; no new external callback starts after the
deadline, and a callback that finishes late is reported as `time_overrun`, not success.

### Controlled local FP16 in static plans

Add `"fp16"` to `allowed_precisions` and select nodes/groups using the existing
rules. PPQ stages these nodes as FP32, then a controlled post-QDQ
`onnxconverter-common` node-block conversion produces real FP16 computation and
weights. It does not invoke the separate whole-model FP16 path. Public IO remains
unchanged, INT8 QDQ scales/zero-points and axis/type evidence remain checked, and
all activation sockets retain exact producer/output-index provenance across verified
FLOAT/FLOAT16 Cast boundaries. Floating parameter names must retain their original
initializer identity. Inherited fusion groups and locks remain atomic after promotion.

Strict shape inference, full ONNX checker and real CPU ORT comparison against the
same pre-FP16 QDQ artifact are mandatory (`rtol=atol=1e-3`, optimization disabled).
Search/candidate conversion uses frozen real calibration batches; standard mode
also checks the current calibration input. This is a conversion sanity gate, not a
deployment performance claim or task-quality guarantee. Unsupported ONNX functions,
subgraphs, selected shared initializers, converter-blocked operators, unsafe output
boundaries, models >=2 GB and inference/ORT/provenance failures raise explicit errors.
No fake FP16 fallback is published. W4A8 and FP8 are outside this interface.
