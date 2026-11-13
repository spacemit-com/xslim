---
name: build-verify
description: XSlim 的环境搭建、定向单元测试、ruff 风格检查、打包 smoke 与提交前检查流程
---

# Build & Verify

Defines how to verify a change before it is reported as done. Use it from
`onnx-pass-developer`, `quantizer-developer` and `qa-inspector`.

## 1. Environment

- Python `>=3.9` (`pyproject.toml`); CI runs 3.12.
- Editable install for development:

```bash
python -m pip install -e .
```

Dependencies come from `requirements.txt`. CI installs CPU-only torch first and
then the rest without reinstalling torch; match that if torch resolution fails.

Do not add a dependency for a test or a helper when the standard library or an
existing dependency already covers it.

## 2. Targeted unit tests

CI command (`.github/workflows/test.yml`):

```bash
XSLIM_TEST_MODEL_CACHE=.cache/test-models \
python -m unittest discover -s tests -p "test_*.py" -v
```

Select the module(s) that cover the change instead of running everything:

```bash
# one module
python -m unittest discover -s tests -p "test_convert_to_fp16.py" -v

# one case across the suite
python -m unittest discover -s tests -p "test_*.py" -k fp16 -v
```

`tests/` has no `__init__.py`; `discover -s tests` is the supported entry point.
Tests must be `unittest.TestCase` classes: `unittest` imports a module of bare
`test_*` functions but collects no cases from it, and `pytest` is not a declared
dependency. Known gap: `tests/test_fp16_domain_legalize.py` currently defines bare
functions, so the CI command does not execute its assertions; convert it to a
`TestCase` before relying on it as evidence.

| module | covers |
|---|---|
| `tests/test_onnxslim_pass.py` | ONNX passes and fusion patterns |
| `tests/test_convert_to_fp16.py` | FP16 conversion / legalization |
| `tests/test_dynamic_quantizer.py` | dynamic weight-only quantization |
| `tests/test_quantize_pipeline.py` | end-to-end FP16 / dynamic / static-guard flows |
| `tests/test_global_functions_mapping.py` | `FunctionProto` mapping and export |
| `tests/test_onnx_operator_support.py` | opset/domain resolution |
| `tests/test_torch_ops.py` | PPQ torch op support |
| `tests/test_packaging_standards.py` | packaging metadata, entry point |

`tests/QuantZoo_*.py` and `tests/coco_eval_helper.py` are evaluation scripts for
external models and datasets; they are not part of the CI unit suite and must not
be reported as unit-test evidence.

Test scope follows `AGENTS.md`「Agent Harness → Test scope」: targeted tests first,
no duplicate or overlapping cases, shard a justified full run.

## 3. Style

CI pins `ruff==0.15.14` and runs `ruff check .`
(`.github/workflows/code-style.yml`). Configuration lives in `pyproject.toml`
(line length 120, `E`/`W`/`F`). Run it before every commit.

## 4. Packaging smoke (only for packaging/metadata changes)

```bash
python -m pip install --upgrade build
python -m build
python -m unittest discover -s tests -p "test_packaging_standards.py" -v
python -m pip install dist/*.whl --no-deps
python -m pip show xslim
```

The console entry point is `xslim = xslim.__main__:main`; the publish workflow
asserts it from wheel metadata.

## 5. Models and datasets

- Never commit models, calibration data, goldens or generated outputs; `.gitignore`
  excludes `.onnx`, `.txt`, `.csv`, `.model` and similar artifacts.
- `XSLIM_TEST_MODEL_CACHE` is the CI cache directory for models exported by the
  pipeline tests.
- `samples/download_data.sh` fetches the sample models and calibration data used
  by `samples/`.
- Accuracy evidence follows Rule 5: name model, dataset, sample count,
  preprocessing, metric and tolerance, and state when an evaluation was not run.

## 6. Harness check

Run after changing `AGENTS.md`, `CLAUDE.md`, `.agents/` or `scripts/`:

```bash
python scripts/check_agent_harness.py
```

It validates cross-references, run-dir prefixes, artifact names, skill
frontmatter and the ignore rules; it also runs in CI.

## 7. Pre-commit gate (Rule 10)

1. Stage only the intended files.
2. Run `ruff check .` and the targeted test module(s).
3. Run `python scripts/check_agent_harness.py` when the harness changed.
4. Commit only after every check passes, with an English
   `type(scope): subject` message.

## Principles

- Evidence first: every "pass" carries the command and its observed result.
- Skipping a step is stated explicitly, never silent.
- A successful conversion is not a numerical-accuracy result.
