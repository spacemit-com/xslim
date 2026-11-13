# Run lifecycle

Shared by every workflow prompt. A *run* is one execution of one workflow.

## One directory per run

`.agents/workspace/<prefix>-<slug>-<yyyymmdd>/`, where `<prefix>` is fixed by
the workflow and `<slug>` is a short English identifier for the change (Rule 14;
each workflow prompt names its prefix). The directory is ignored scratch: never
staged, never committed, and the only place a workflow writes.

Concurrent runs must not share one working tree or one git index — a shared
index races on `git add`. Give each run its own `git worktree`, or serialize the
runs.

## Fixed artifact names

Fixed names are what make a run resumable in a later session: a run that died
mid-way leaves a directory a fresh agent can read and continue from.

| artifact | author | contents |
|---|---|---|
| `scout.md` | `codebase-scout` | located code, call chains, registration points |
| `design.md` | writer role | change points, interface impact, risks, doc-sync list |
| `impl.md` | writer role | what was implemented, and where |
| `eval.md` | `accuracy-analyst` | baseline, dataset/model identity, metrics, commands |
| `review.md` | `qa-inspector` | issues as `file:line` + evidence, and PASS/FAIL |
| `handoff.md` | orchestrator | final summary, written last |

*Writer role* is the role that implements the change in that workflow — each
workflow prompt names it. It authors `design.md` when the workflow has a design
step, and always authors `impl.md`.

## Ending a run

Every run ends by writing `handoff.md`: what changed, what was verified (with
the command and its observed result), and what is unresolved. Then:

- Verified PASS, and the workflow allows it → commit per Rule 10.
- The workflow declares a human checkpoint → stop at the HANDOFF and let the
  user decide; the workflow prompt says when this applies.
- FAIL, or anything left unverified → keep it uncommitted and say so in
  `handoff.md` (Rule 12).

## Retention

Run directories are scratch, not history: a finding that gates a merge belongs in
the PR description, not in a directory one sweep can delete.

`scripts/clean_agent_workspace.sh` implements the retention policy: a run
directory is deleted once it is older than `--days` (default 14) and is not among
the `--keep` (default 3) most recent runs of its prefix; a directory that is
still a git worktree is removed with `git worktree remove --force` first. It is a
dry run unless `--apply` is given, and it only ever touches
`<prefix>-<slug>-<yyyymmdd>` directories whose prefix `AGENTS.md` Rule 14
declares — anything else is reported and skipped.

## Archive before cleanup

Use `.agents/skills/workspace-cleanup/SKILL.md` when the user requests historical
run consolidation or cleanup. Preserve important conclusions, verification limits,
pending designs and recoverable unique work under `.agents/history/` before
deleting their workspace sources. Both workspace and history are Git-ignored:
never stage or commit their contents. Historical records do not replace current
project documentation or the PR description.

The retention script only selects/deletes directories; it does not archive them
or check active jobs. Inspect its dry-run selection and archive/check that set
before `--apply`. An explicit user request can override age/count retention; keep
active work, shared data and unrelated worktrees outside the deletion set.

## Keeping this file honest

`scripts/check_agent_harness.py` validates the committed harness and fails when it
drifts from what is written here: a workflow prompt that does not wire in both
shared files or name its run directory, a run-dir prefix Rule 14 does not
declare, an artifact name the table above does not define, a dangling
`.agents/...` reference, a role or skill file nothing references, a skill without
frontmatter, or a `.gitignore` that stops ignoring `.agents/workspace/` or
`.agents/history/`, or tracks their contents despite the ignore rules. It runs in
CI (`.github/workflows/code-style.yml`) and before every commit that touches the
harness (Rule 10).

## Failure handling common to every workflow

- A verification FAIL round-trips back to the writer role, at most twice, with
  the scope unchanged. Still failing after two rounds → list the open root
  causes in `handoff.md` and stop. Do not claim completion.
- No usable reference implementation or no similar existing entry → state the
  gap and fall back to the simplest construct. Do not force reuse of something
  that does not fit, and do not paper over it with a special case.
- Config-schema, ONNX-domain or output-format questions → re-read the
  authoritative source (`xslim_setting.py`, `defs.py`, the exporter, the ONNX
  spec) and cross-check. Do not guess, and do not enter implementation while the
  question is open.
- A design that contradicts existing documentation → pick the more recent / more
  exercised side and say which one you dropped and why (Rule 7).
