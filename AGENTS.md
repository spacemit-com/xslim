# Project Rules

This file is the single source of truth for the project's rules. `CLAUDE.md`
imports it; if the two ever disagree, this file wins.

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## Rule 1 - Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

## Rule 2 - Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 - Keep Documentation Current
XSlim documents user-facing behavior twice — an English file and a `_zh`
counterpart — and both must stay current in the same change.

- CLI flags, JSON config keys, defaults or validation: `doc/configuration.md` and
  `doc/configuration_zh.md`; usage examples: `doc/examples.md` and
  `doc/examples_zh.md`.
- Accuracy behavior, precision-level guidance, Graphwise Analysis interpretation:
  `doc/accuracy_tuning.md` and `doc/accuracy_tuning_zh.md`.
- Installation, packaging, public API, feature list: `README.md` and
  `README_zh.md`.
- Sample configurations and scripts: `samples/README.md` and
  `samples/README_zh.md`.
- Documentation index or structure: `doc/README.md` and `doc/README_zh.md`.

A change that alters behavior but leaves any of these stale is incomplete. Never
update only one language.

## Rule 4 - Systematic, Modular Optimization
Base every optimization on system-level rules. Keep strategy logic modular,
maintainable, composable and reusable. Do not add specialization solely for
correctness or performance. Avoid magic numbers and long-distance parameter
coupling. Decouple modules through explicit interfaces while avoiding unnecessary
abstraction. A pass, quantizer or observer that special-cases one model, one
node name or one exact tensor shape is a defect, not a fix.

## Rule 5 - Model and Data Evidence Discipline
Accuracy and conversion claims are only as good as the data behind them.

- Never fabricate a metric, a baseline or a model result. If an evaluation was not
  run, say so.
- Name the evidence: model file and version/hash, precision variant, calibration
  or evaluation dataset, sample count, preprocessing, metric, tolerance, and the
  command that produced it.
- Distinguish synthetic tensors from real deployment data. A random-input run
  proves a graph executes; it does not prove accuracy.
- Declare integer/exact-match checks and floating-point tolerances before looking
  at the result. Never widen a tolerance to hide a real regression.
- Test models and datasets are external inputs: never commit weights, calibration
  images, ONNX files or generated outputs. Pass them through explicit paths or
  environment variables and keep them outside the repository.
- Do not overwrite a downloaded or published model with a generated one. Preserve
  original inputs, references and goldens.

## Rule 6 - Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

## Rule 7 - Surface Conflicts, Don't Average Them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

## Rule 8 - Read Before You Write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

## Rule 9 - Tests Verify Intent, Not Just Behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when the quantization logic changes is wrong.
Prefer a tiny constructed ONNX model that reproduces the failure over snapshots of
a large graph. One behavior per case; do not add a case that restates an existing
one.

## Rule 10 - Checkpoint Only After Verification
Summarize what was done, what is verified, and what remains after every
significant step. Do not continue from a state you cannot describe. If you lose
track, stop and restate the current state.

Before every commit:
1. Stage only the intended files.
2. Run `ruff check .` and the targeted tests for the change.
3. Run `python scripts/check_agent_harness.py` when `AGENTS.md`, `CLAUDE.md`,
   `.agents/` or `scripts/` changed.
4. Commit only after every check passes.

After completing each meaningful iteration, automatically create a Git commit
with a concise English message in the existing history style, for example
`feat(onnxslim_pass): ...` or `fix(fp16): ...`. Do not wait for separate user
approval. Bounds:
- Commit on a branch, never directly on `main`.
- A workflow with a human checkpoint may end at a HANDOFF instead of a commit:
  `quantizer-dev` and `accuracy-tuning` do when the change is user-visible or the
  measured improvement is the user's call. Follow that workflow's own ending.
- Never commit a state you cannot describe (Rule 12). Leave mid-debug work
  uncommitted and say so.
- Do not rewrite or force-push shared history.

## Rule 11 - Match the Codebase's Conventions, Even If You Disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.
Existing conventions include the `logger` usage, the `_zh` documentation pairing,
`ruff` with line length 120 and the `E`/`W`/`F` rule set, and the
`type(scope): subject` commit style.

## Rule 12 - Fail Loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## Rule 13 - Portable Paths
Do not hard-code environment-specific absolute paths in source code, tests,
scripts, build configuration or checked-in documentation. Never embed a developer
home directory, checkout location, output directory, username or temporary
directory. Use repository-relative paths, environment variables, command-line
inputs or runtime discovery instead. Model, calibration and evaluation datasets
are user-supplied inputs: accept them as arguments or environment variables.

## Rule 14 - Agent Workspace
Store every agent-created plan, design, scratch note, temporary report, generated
patch and intermediate artifact under `.agents/workspace/`. Do not place these
files in the repository root, `doc/`, source directories or test directories.

Name each run's directory `.agents/workspace/<prefix>-<slug>-<yyyymmdd>`, with the
prefix taken from the workflow: `pass`, `quant`, `acc`, `review`, `maint`. Runs
that execute concurrently must not share one working tree or one git index: give
each run its own `git worktree`, or serialize the runs, because a shared index
races on `git add` / `commit`.

`.agents/workspace/` and `.agents/history/` are ignored scratch space: never stage
or commit anything inside them. Retention is enforced by
`scripts/clean_agent_workspace.sh`, whose policy lives in `.agents/shared/run.md`.
During cleanup, retain important local conclusions and recovery evidence in
`.agents/history/`; this is also ignored and must never be staged or committed.
Use the `workspace-cleanup` skill for archive-before-delete work. The rest of
`.agents/` is the committed harness (`agents/`, `prompts/`, `skills/`, `shared/`)
and follows Rules 3 and 10 like any other checked-in file. Permanent
user-facing project documentation is not an agent artifact and belongs in `doc/`.
A finding that gates a merge must be summarized in the PR description, because
reviewers cannot read the ignored run records. The `.claude/` directory stays
ignored as a local-only directory: never put project rules there.

## Rule 15 - Public Interface Stability
XSlim is a published package. Treat the following as contracts:

- the `xslim` console entry point and every CLI flag in `src/xslim/__main__.py`;
- `xslim.quantize_onnx_model` and the other names exported from
  `src/xslim/__init__.py`;
- JSON configuration keys, their defaults and validation in
  `src/xslim/xslim_setting.py`;
- ONNX opset behavior (`MIN_ONNX_OPSET_VERSION` and `resolve_operator_domain`);
- the serialized output contract: operator domains, emitted `FunctionProto`
  definitions, external-data layout and metadata.

Additive change is preferred. A breaking change needs an explicit decision, a
migration note in both language documents, and a version bump. Never silently
rename a config key, drop a CLI flag or change a default.

## Language

All user-facing discussion must be in Chinese. All content in this file, code
identifiers, and commit messages must be in English. User-facing repository
documentation is bilingual per Rule 3.

## Agent Harness

This repo ships an agent harness under `.agents/` (committed). It is an evolving
system; extend it after observing real task failures, not speculatively.

### Test scope
Scope every test run to the change, not to the repository.

- When defining what a feature must prove, choose targeted tests — name the test
  modules and the `-p` / `-k` selection in the task — instead of running the full
  suite by default.
- Add a test only when it can fail for a reason this change is responsible for.
  No case that restates an existing one, no case added to raise a count.
- Keep cases non-overlapping: one behavior per case, and if two cases would fail
  from the same root cause, keep the one closest to the change and drop the
  other.
- A full-suite run is justified only by a cross-cutting change (public API,
  config schema, exporter/serialization, dependency pins, packaging). When one is
  needed, shard it by test module and run the shards concurrently; each shard
  needs its own `.agents/workspace/` run directory and its own git worktree or a
  serialized index (Rule 14). Do not serialize a suite that can be sharded.

### Agents (`.agents/agents/`)
- `codebase-scout` — read-only recon: locate code, trace the quantization
  pipeline, find pass registration points, cross-read docs and tests.
- `onnx-pass-developer` — implement ONNX graph passes and `onnxslim` fusion
  patterns, with regression tests and doc sync.
- `quantizer-developer` — implement quantization behavior: static INT8, dynamic
  quantization, FP16 conversion/legalization, exporters, config and CLI.
- `accuracy-analyst` — read-only accuracy diagnosis: baseline evaluation,
  Graphwise Analysis, tuning experiments and reports; escalates code defects.
- `qa-inspector` — read-only verification: interface compatibility, doc sync,
  test intent, targeted runs, style and packaging.

Mechanics — which tool starts a child, why role files must be inlined, and what a
child cannot be trusted to do — live in `.agents/shared/delegation.md`; the run
directory, artifact names and how a run ends live in `.agents/shared/run.md`.
Both are shared by every workflow below. Read them before adapting a workflow to
a new runtime.

The harness is runtime-agnostic. In Claude Code, use the `Agent` tool with an
`Explore`-type subagent for read-only roles; in Codex, use its agent-spawning
tools; a runtime that exposes `delegate_to_agent` / `get_delegation_status` can
use that pair directly.

### Skills (`.agents/skills/`)
No runtime in this repo auto-discovers `.agents/skills/`; read
`<name>/SKILL.md` directly.

- `build-verify` — environment setup, targeted `unittest` runs, `ruff`, packaging
  smoke tests and the pre-commit gate (Rule 10).
- `onnx-pass-dev` — how to add or change an ONNX pass in this repo: pattern
  registration vs explicit model passes, custom-domain functions, tests.
- `quantization-dev` — how to change the static/dynamic/FP16 pipeline, exporter,
  config schema and large-model handling.
- `workspace-cleanup` — archive important local history, preserve dirty-worktree
  recovery, and clean scoped run directories.

The accuracy-tuning skill lives in the GitHub-facing catalog and is used as-is:
`.github/skills/xslim-accuracy-tuning/SKILL.md` (references under
`.github/skills/xslim-accuracy-tuning/references/`). Do not duplicate it under
`.agents/`.

### Orchestration prompts (`.agents/prompts/`)
These are prompt files, not registered commands. Hand the prompt text to an agent
and it runs the workflow: `/onnx-pass-dev <change>` — add or fix a graph
pass/fusion; `/quantizer-dev <change>` — change quantization, FP16/dynamic
behavior, config or CLI; `/accuracy-tuning <model or issue>` — evidence-based
accuracy diagnosis and tuning; `/code-review <scope>` — evidence-based review of a
change.

### Routing

- Build, test, style or packaging question: `.agents/skills/build-verify/SKILL.md`.
- Adding/changing an ONNX fusion or graph pass: `.agents/prompts/onnx-pass-dev.md`
  and `.agents/skills/onnx-pass-dev/SKILL.md`.
- Quantization pipeline, FP16 conversion, dynamic quantization, config keys, CLI:
  `.agents/prompts/quantizer-dev.md` and
  `.agents/skills/quantization-dev/SKILL.md`.
- Accuracy drop, precision-level choice, Graphwise Analysis, calibration data:
  `.agents/prompts/accuracy-tuning.md`, then
  `.github/skills/xslim-accuracy-tuning/SKILL.md`.
- Review of a change: `.agents/prompts/code-review.md` (delegates `qa-inspector`,
  evidence-based PASS/FAIL).
- Workspace/history cleanup: `.agents/skills/workspace-cleanup/SKILL.md`.
