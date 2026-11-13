---
name: workspace-cleanup
description: Archive important local work history and clean historical agent workspaces and generated dumps. Use when the user requests workspace cleanup or historical-run consolidation.
---

# Workspace Cleanup

Complete the user's requested cleanup while preserving useful conclusions and
recoverable unfinished work. Keep `.agents/history/` local and Git-ignored;
the skill and project rules are versioned, historical records are not.

## Establish the deletion scope

Read [run lifecycle](../../shared/run.md) and the repository's `AGENTS.md`.
Use `.agents/workspace/maint-<slug>-<yyyymmdd>/` for this cleanup's temporary
inventory and scripts. Resolve repository roots at runtime; never bake a username,
checkout path or credential into reusable artifacts.

Inspect Git status, registered worktrees, direct workspace entries and sizes
before selecting directories. Prune copied source/build trees when searching for
reports; recursively listing every checkout obscures actual history. Check
symlinks and mount boundaries without traversing targets outside the scope.

Use the user's scope and exclusions. Ordinary retention follows the existing
`scripts/clean_agent_workspace.sh` dry-run policy. An explicit request to clear all
historical work may override its age/count limits; it does not authorize unrelated
worktrees, production builds, original models, other users' files or active jobs.
Unrecognized run names require inspection, not automatic deletion. Exclude the
current cleanup run until its final receipts are saved.

The retention script does not archive information or detect active work. Do not
run its `--apply` mode until the selected set is archived and checked. Its force
removal must not discard unrecorded changes.

## Archive before deleting

Create a unique dated directory under `.agents/history/`; preserve earlier
archives. Check both ignore rules and `git ls-files`: adding an ignore pattern
does not untrack existing files. Never use `git add -f` on history. If history is
tracked, remove only its index entries while retaining local files. Follow the
user's instruction about existing commits; do not rewrite shared history.

Write a concise summary and run index covering:

- What changed, source commits, decisions and superseded approaches.
- Observed test results, commands or case filters, and verification limits.
- Accuracy/evaluation baselines: model and dataset identity, sample counts,
  metrics, tolerances and failed gates; distinguish synthetic from real data.
- Unresolved failures, pending designs and evidence needed to resume them.

Retain primary reports. Preserve useful experiment sources, raw result tables and
selected logs when they cannot be reconstructed from Git. Avoid preserving
reproducible binaries, copied model weights, build trees and bulk generated
models by default. A top-level directory named `src` inside a scratch experiment
may contain its only implementation; do not drop it merely because copied
repository source trees are excluded elsewhere.

For each dirty worktree, record its base commit/status and capture tracked binary
patches against HEAD, staged changes separately, untracked files and symlink
metadata. Preserve any required unreferenced base commit before unregistering its
worktree. Recheck status and patches before deletion to detect concurrent changes.
If unique work cannot be recovered, retain that directory and report it while
cleaning independent candidates.

Use compressed local archives for bulky evidence. Record member/archive hashes and
reread members to verify them before deleting their sources. Keep credentials out
of summaries; protect raw local recovery data when needed. State clearly that
historical reports are records, not current instructions or fresh test results.

## Execute and verify

Delete only the inventoried, archived, inactive candidates. Unregister selected
worktrees with Git; do not prune or remove unrelated registered worktrees.
Preserve symlink targets. Restrict dump cleanup to the requested generated dump
roots and confirm no tracked or unique input files are included.

Check archive readability, remaining entries, retained shared data and worktree
registration. Save cleanup receipts in history, then remove the cleanup run if an
empty workspace was requested. Report deleted sizes, archive size, intentional
exclusions and incomplete actions.

Before any versioned rule/skill change is committed, run
`python scripts/check_agent_harness.py`, the targeted tests and `ruff check .`
(per `AGENTS.md` Rule 10). Confirm neither workspace nor history files are tracked
or staged. Cleanup alone needs no test suite run; do not describe directory checks
as model or accuracy verification.
