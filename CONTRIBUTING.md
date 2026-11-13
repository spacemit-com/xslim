# Contributing

This guide applies to human contributors and agents. It covers development
branches, pull requests (PRs), code style and submission requirements.
Project-wide rules live in [AGENTS.md](AGENTS.md); this guide adds
contributor-facing conventions. Environment setup, targeted tests, style checks
and packaging commands live in
[.agents/skills/build-verify/SKILL.md](.agents/skills/build-verify/SKILL.md).

## Scope of work

- Work on a development branch and merge through a PR into `main`. Do not
  commit development work directly to `main`.
- One branch and one PR per task. Do not mix unrelated refactors, formatting or
  dependency upgrades.
- Read the relevant interfaces, callers, shared helpers and documentation before
  changing code; state the goal, constraints and verification method.
- Prefer the simplest implementation that meets the need; reuse existing types
  and helpers; avoid duplicate implementations, speculative abstractions and
  model-specific hard-coding.
- Do not rewrite or force-push shared history.

## Branch naming

New development branches use one of two formats:

```text
agent/<type>/<topic>
<user_name>/<type>/<topic>
```

- Agent-created branches use the `agent/` prefix, including work delegated to an
  agent by a developer.
- Personal branches use the contributor's username as a lowercase prefix without
  spaces. Do not hard-code a specific person's name in examples, scripts or
  repository files.
- `type` must be one of the lowercase values in the table below.
- `topic` uses lowercase English words and digits separated by single `-`. Do not
  use spaces, underscores or non-ASCII characters.
- To reference an issue, prefix the topic with its number, for example
  `agent/fix/123-large-model-fp16`.
- `main` and `release/*` are maintained by the maintainers and are not used as
  day-to-day development branches.

| Type | Purpose |
|------|---------|
| `feat` | New functionality, pass, quantization behavior or configuration option |
| `fix` | Defect fix |
| `refactor` | Structural change, rename or module split with unchanged behavior |
| `perf` | Performance or memory optimization |
| `test` | Test cases or test-framework changes |
| `docs` | Documentation, contribution guide or development conventions |
| `build` | Dependencies, packaging or version metadata |
| `ci` | GitHub Actions and automated checks |
| `chore` | Maintenance work that fits none of the above |
| `revert` | Revert an existing change |

Use `refactor` for restructuring and `fix` for defect fixes. Express urgency
through PR labels, not by adding synonymous branch types such as `bugfix` or
`hotfix`.

```text
agent/feat/window-partition-fusion
agent/fix/large-model-fp16
<user_name>/fix/qdq-input-guard
<user_name>/docs/contributing-guide
```

## PR titles

Titles use the Conventional Commits form:

```text
<type>(<scope>): English summary
```

- `type` must be one of the branch type values above, in lowercase.
- `scope` names the primary affected module and must be one of the values in the
  table below, in lowercase. Code changes must include a scope; repository-wide
  `docs`, `build`, `ci` and `chore` changes may omit it.
- Use exactly one scope. For cross-module changes, choose the primary purpose;
  do not combine scopes.
- Write the summary in English, start with a verb, and describe the final
  behavior. Keep the whole title under 72 characters and do not end with a
  period.
- Do not include contributor names, branch names or agent identities in the
  title. Avoid vague summaries such as `update code` or `some fixes`.
- Mark incomplete PRs as drafts. Update the title and description when the scope
  changes; the branch name may stay.
- Do not invent out-of-table scopes. Extend this guide before introducing one;
  do not add a one-off scope in a single PR.

| Scope | Area |
|-------|------|
| `onnxslim_pass` | Graph fusion, graph passes and custom-domain `FunctionProto` under `src/xslim/onnxslim_pass/` |
| `quantizer` | Static INT8, quantization algorithms, observers, calibration and equalization |
| `fp16` | FP16 conversion and legalization |
| `torch` | Torch execution and operator support in `ppq_decorator` |
| `pipeline` | Flow orchestration, model loading, truncation and saving in `xslim_pipeline.py` |
| `config` | JSON configuration schema, defaults and validation |
| `cli` | Command-line arguments and entry points in `__main__.py` |
| `export` | PPQ export, `FunctionProto` and external-data layout |
| `tests` | Test cases and test framework |
| `docs` | Documentation, contribution guide and development conventions |
| `ci` | GitHub Actions and automated checks |
| `agents` | `AGENTS.md`, the `.agents/` harness and development scripts |
| `build` | Dependencies, packaging and version metadata |

The branch `type` describes the kind of change; the PR `scope` describes the
primary affected module. Choose them independently:

| Branch | PR title |
|--------|----------|
| `agent/feat/window-partition-fusion` | `feat(onnxslim_pass): fuse window partition subgraphs` |
| `<user_name>/fix/qdq-input-guard` | `fix(pipeline): reject QDQ input models for static quantization` |
| `agent/fix/large-model-fp16` | `fix(fp16): convert models larger than 2GB on disk` |
| `agent/ci/harness-check` | `ci(agents): validate harness references in CI` |
| `<user_name>/docs/contributing-guide` | `docs: add contribution guidelines` |

## Code style and dependency boundaries

- The minimum Python version is 3.9 (`pyproject.toml`); CI uses 3.12.
- Style is checked with `ruff check .`, configured in
  [pyproject.toml](pyproject.toml): line length 120, rules `E`/`W`/`F`, with
  `ruff==0.15.14` pinned in CI. The repository does not enforce isort; preserve
  the import grouping of the file you edit and avoid unrelated reordering.
- Use English identifiers. Comments should explain reasons, constraints and
  boundary conditions rather than restating the implementation.
- Library code logs through `xslim.logger`; do not use `print` for runtime
  output.
- Put new code in the directory responsible for it:
  - graph fusion/passes -> `src/xslim/onnxslim_pass/`
  - quantization, calibration and observers -> `src/xslim/quantizer/`,
    `src/xslim/optimizer/`
  - configuration and validation -> `src/xslim/xslim_setting.py`,
    `src/xslim/defs.py`
  - CLI and orchestration -> `src/xslim/__main__.py`,
    `src/xslim/xslim_pipeline.py`
  - PPQ adaptation and export -> `src/xslim/ppq_decorator/ppq/`
- `src/xslim/ppq_decorator/` is a maintained adaptation of upstream PPQ. Follow
  the upstream style, explain necessary deviations, and do not perform unrelated
  bulk formatting.
- Do not add dependencies casually. When one is required, state its purpose and
  version bounds, and update `requirements.txt` and `pyproject.toml` together.
  Tests must not pull in heavy dependencies used only by the tests.
- Public-interface compatibility follows [AGENTS.md](AGENTS.md) Rule 15: CLI
  flags, exported names such as `quantize_onnx_model`, configuration keys and
  defaults, and the exported-model contract are all published contracts.
- Do not hard-code environment-specific absolute paths in source, scripts, tests
  or documentation. Use repository-relative paths, environment variables or
  command-line arguments. Models, calibration data and evaluation data are
  user-supplied inputs passed by path.

## Verification requirements

Choose the verification scope from the actual impact of the change:

| Change | Verification |
|--------|--------------|
| Documentation, contribution guide, harness | Check links and examples; run `ruff check .`; run `python scripts/check_agent_harness.py` when `.agents/` or the scripts change |
| Graph pass / fusion | Run the relevant module in `tests/test_onnxslim_pass.py` or a targeted test module; verify graph outputs, numerical comparison, a negative case and idempotence |
| Quantization behavior (static/dynamic/FP16) | Run the relevant unit tests; add an end-to-end conversion smoke test when needed; state whether numerical or accuracy validation was performed |
| Configuration, CLI, public API | Targeted tests plus a compatibility statement; breaking changes need a bilingual migration note and a `VERSION_NUMBER` bump |
| Export format, external data, packaging | Targeted tests plus a packaging smoke test (`python -m build`, `test_packaging_standards.py`) |
| Accuracy tuning | Declare the model, dataset, sample count, preprocessing, metric and tolerance; distinguish synthetic data from real data |
| CI / workflows | Reproduce the corresponding workflow command locally, or state why it cannot be reproduced |

- Name the test modules and the `-p` / `-k` selection instead of running the full
  suite by default (see [build-verify](.agents/skills/build-verify/SKILL.md)).
- Tests must verify the purpose and key boundaries of the change. Do not repeat
  existing cases or merely restate the implementation. When a full run is
  justified, shard it by test module.
- A successful conversion, an executable graph and passing numerical or accuracy
  checks are three different claims; report them separately.
- State explicitly what was not run, skipped or failed. Do not present a partial
  result as a complete conclusion.
- Follow [AGENTS.md](AGENTS.md) Rule 5 for model and data evidence. The accuracy
  workflow is described in the
  [accuracy-tuning prompt](.agents/prompts/accuracy-tuning.md) and the
  [xslim-accuracy-tuning skill](.agents/skills/xslim-accuracy-tuning/SKILL.md).

## Documentation and repository content

- Update the documentation that describes an interface, configuration option,
  CLI behavior or other user-visible behavior in the same PR. User-facing docs
  are maintained in English and Chinese pairs:
  - configuration/CLI -> `doc/configuration.md` + `doc/configuration_zh.md`
  - usage examples -> `doc/examples.md` + `doc/examples_zh.md`
  - accuracy strategy -> `doc/accuracy_tuning.md` + `doc/accuracy_tuning_zh.md`
  - installation, public API, feature list -> `README.md` + `README_zh.md`
  - sample projects -> `samples/README.md` + `samples/README_zh.md`
- Do not commit models, weights, calibration data, goldens, build artifacts or
  credentials. `.gitignore` excludes `.onnx`, `.model`, `.csv`, `.txt` and
  similar data files; update the ignore rules when adding a new data format.
- Agent plans, logs and intermediate artifacts belong in the ignored
  `.agents/workspace/` directory and must not be committed. Local conclusions
  that need to be retained belong in `.agents/history/`, which is also ignored.
- Shared agent roles and workflows are documented in the "Agent Harness"
  section of [AGENTS.md](AGENTS.md).

## Commit and PR checks

The repository does not use a pre-commit framework. Run these checks directly
before committing:

```bash
ruff check .
python -m unittest discover -s tests -p "test_<affected-module>.py" -v
python scripts/check_agent_harness.py   # when AGENTS.md, .agents/ or scripts/ change
```

Before each commit:

1. Review the diff and stage only the files intended for this commit.
2. Run the checks above; add an end-to-end smoke test when quantization behavior
   changed.
3. Inspect any modified files, restage if necessary, and rerun until every check
   passes.
4. Write the commit message in English, describing the actual change, using the
   same `<type>(<scope>): <summary>` form as the PR title.

A PR description contains at least:

- **Problem and change**: the problem being solved and the resulting behavior,
  with before/after examples when useful.
- **Impact**: affected modules, compatibility impact and migration requirements.
- **Verification evidence**: test commands or selections and their results; for
  quantization or accuracy work, the model, data, sample count, metric and
  tolerance.
- **Unverified scope**: what was not run, was skipped, failed, or remains
  limited.

Put the key conclusions in the PR body; do not point reviewers only at local
paths or logs inside ignored directories. Before merging, pass the applicable CI
checks:

- [Code Style](.github/workflows/code-style.yml): `ruff check .` and
  `python scripts/check_agent_harness.py`.
- [Unit Tests](.github/workflows/test.yml):
  `python -m unittest discover -s tests -p "test_*.py"`.
- [Python package](.github/workflows/python-package.yml): build the sdist/wheel
  and verify the entry point.

At least one maintainer must review the PR.
