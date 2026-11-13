---
name: onnx-pass-dev
description: 在 XSlim 新增/修改 ONNX 图 pass 的落点与流程：onnxslim pattern 注册、显式 ModelProto pass、自定义域 FunctionProto、测试与文档同步
---

# ONNX Pass Development

Landing points and procedure for adding or changing an ONNX graph pass in this
repo. Semantics come from the ONNX specification and the existing implementations
in `src/xslim/onnxslim_pass/`; this skill only covers where things live and how
they are wired.

## 1. Two pass styles

### Registered fusion pattern (preferred for subgraph fusion)

```
src/xslim/onnxslim_pass/<name>.py
    subclass PatternMatcher
    Pattern("""...""") DSL describes the subgraph
    rewrite() returns {node_name: {op, inputs, outputs, domain, name}} or {}
    register_fusion_pattern(<Matcher>(priority))
src/xslim/onnxslim_pass/__init__.py
    import the module so registration happens at import time
```

Registered patterns run inside `onnxslim.slim(...)`, called from
`optimize_onnx_model()` in `onnxslim_pass/__init__.py`. Read an existing matcher
such as `gelu.py` before writing one.

### Explicit model pass

For rewrites that are easier on the ONNX graph than as a fixed pattern
(for example `window_partition.py`, `yolo_decode.py`):

```
def fuse_<name>(onnx_model) -> onnx.ModelProto
def ensure_<op>_function(onnx_model) -> onnx.ModelProto   # when a FunctionProto is needed
```

Wire it into `optimize_onnx_model()` in a deterministic position. Use
`osg.import_onnx` / `osg.export_onnx` when graph-surgeon manipulation is clearer;
call `graph.cleanup().toposort()` before export.

## 2. Registration checklist

1. File under `src/xslim/onnxslim_pass/`.
2. Import in `src/xslim/onnxslim_pass/__init__.py` (registration side effect) and
   wire explicit passes into `optimize_onnx_model()`.
3. If a custom operator is emitted:
   - use domain `spacemit_functions` and a `FunctionProto` via `ensure_*_function`;
   - add the domain to `opset_import` exactly once;
   - keep `functions` and `opset_import` consistent after `format_onnx_model`
     (it clears `value_info` and may run version conversion).
4. Choose the default ai.onnx domain vs a custom domain with
   `xslim.defs.resolve_operator_domain(op_type, opset)`; do not hard-code domains.
5. Keep `MIN_ONNX_OPSET_VERSION` behavior in mind (`defs.py`, currently 24).

## 3. Correctness constraints

- Preserve graph inputs/outputs and their order; never rewire an output to a
  different tensor.
- Handle constants both as `initializer` and as `Constant` nodes.
- Do not assume static shapes; a pattern may see symbolic or `?` dimensions.
- The pass must be idempotent: running `optimize_onnx_model` twice yields the same
  graph shape and no duplicate custom functions.
- Remove orphaned nodes and keep a valid topological order.
- Matching conditions must derive from operator semantics, attributes or
  provable constant values — never from a model name, node name or exact tensor
  shape (Rule 4).
- On an unexpected graph, prefer logging a warning and returning the model
  unchanged over producing a wrong graph.

## 4. Tests

Add cases to `tests/test_onnxslim_pass.py` (or a dedicated `tests/test_*.py`
module when the surface is large):

- construct the smallest ONNX model that exercises the pattern with
  `onnx.helper`;
- assert fused op type, domain, input/output names and node counts;
- assert numerical equivalence against the unfused model with
  `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])` when
  the operators are CPU-supported;
- include a negative case where the pattern must **not** fire;
- include an idempotence or second-run case when the pass rewrites in place.

Prefer one behavior per case; do not snapshot large graphs.

## 5. Documentation (Rule 3, mandatory)

Update both languages when the behavior is user visible:

- `doc/examples.md` + `doc/examples_zh.md` for usage/support examples;
- `doc/configuration.md` + `doc/configuration_zh.md` for config/CLI impact;
- `README.md` + `README_zh.md` for the feature list.

## Principles

- Evidence first: every "fused/fixed" claim carries a test command and output.
- A pass that fires on one model but cannot state its matching rule is not done.
