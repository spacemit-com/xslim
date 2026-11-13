#!/usr/bin/env python3
"""Validate the committed agent harness described by AGENTS.md.

Fails when the harness drifts from the documented layout: a workflow prompt that
does not wire in the shared files or name its run directory, an undeclared
run-dir prefix, an artifact unknown to `.agents/shared/run.md`, a dangling
reference, an unreferenced role or skill, a skill without frontmatter, or an
ignore rule that stops covering agent scratch space.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
CLAUDE = ROOT / "CLAUDE.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
AGENTS_DIR = ROOT / ".agents"
PROMPTS_DIR = AGENTS_DIR / "prompts"
ROLES_DIR = AGENTS_DIR / "agents"
SKILLS_DIR = AGENTS_DIR / "skills"
RUN_DOC = AGENTS_DIR / "shared" / "run.md"
DELEGATION_DOC = AGENTS_DIR / "shared" / "delegation.md"

PREFIXES = ("pass", "quant", "acc", "review", "maint")
ARTIFACTS = ("scout.md", "design.md", "impl.md", "eval.md", "review.md", "handoff.md")

errors: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def frontmatter(path: Path) -> dict[str, str]:
    text = read(path)
    if not text.startswith("---\n"):
        fail(f"{rel(path)}: missing YAML frontmatter")
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        fail(f"{rel(path)}: unterminated YAML frontmatter")
        return {}
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def check_required_files() -> None:
    for path in (AGENTS, CLAUDE, RUN_DOC, DELEGATION_DOC):
        if not path.is_file():
            fail(f"missing required harness file: {rel(path)}")
    if CLAUDE.is_file() and "@AGENTS.md" not in read(CLAUDE):
        fail("CLAUDE.md must import AGENTS.md with '@AGENTS.md'")


def check_prompts(agents_text: str, run_text: str) -> list[str]:
    prompt_texts: list[str] = []
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        text = read(path)
        prompt_texts.append(text)
        name = rel(path)
        for required in (".agents/shared/delegation.md", ".agents/shared/run.md"):
            if required not in text:
                fail(f"{name}: does not reference {required}")
        match = re.search(r"\.agents/workspace/([a-z]+)-<slug>-<yyyymmdd>/", text)
        if not match:
            fail(f"{name}: does not name a run directory of the form .agents/workspace/<prefix>-<slug>-<yyyymmdd>/")
        elif match.group(1) not in PREFIXES:
            fail(f"{name}: run-dir prefix '{match.group(1)}' is not declared in AGENTS.md Rule 14")
        for token in re.findall(r"`([^`]+)`", text):
            if not token.endswith(".md") or "/" in token:
                continue
            if token not in ARTIFACTS and token not in {"AGENTS.md", "CLAUDE.md", "SKILL.md", "README.md"}:
                fail(f"{name}: mentions artifact '{token}' not defined in .agents/shared/run.md")
    if not prompt_texts:
        fail("no orchestration prompts found under .agents/prompts/")
    for prefix in PREFIXES:
        if f"`{prefix}`" not in agents_text:
            fail(f"AGENTS.md Rule 14 does not declare run-dir prefix '{prefix}'")
    for artifact in ARTIFACTS:
        if f"`{artifact}`" not in run_text:
            fail(f".agents/shared/run.md artifact table does not define '{artifact}'")
    return prompt_texts


def check_roles(agents_text: str, prompt_texts: list[str]) -> None:
    roles = sorted(ROLES_DIR.glob("*.md"))
    if not roles:
        fail("no role files found under .agents/agents/")
    for path in roles:
        name = path.stem
        fields = frontmatter(path)
        for field in ("name", "description"):
            if not fields.get(field):
                fail(f"{rel(path)}: frontmatter is missing '{field}'")
        if fields.get("name") and fields["name"] != name:
            fail(f"{rel(path)}: frontmatter name '{fields['name']}' does not match file name")
        if f"`{name}`" not in agents_text:
            fail(f"{rel(path)}: role is not listed in AGENTS.md")
        if not any(name in text for text in prompt_texts):
            fail(f"{rel(path)}: role is not used by any orchestration prompt")


def check_skills(agents_text: str) -> None:
    skills = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    if not skills:
        fail("no skills found under .agents/skills/")
    for path in skills:
        fields = frontmatter(path)
        for field in ("name", "description"):
            if not fields.get(field):
                fail(f"{rel(path)}: frontmatter is missing '{field}'")
        skill_name = path.parent.name
        if fields.get("name") and fields["name"] != skill_name:
            fail(f"{rel(path)}: frontmatter name '{fields['name']}' does not match directory name")
        if f"`{skill_name}`" not in agents_text:
            fail(f"{rel(path)}: skill is not listed in AGENTS.md")


def check_references() -> None:
    candidates = [AGENTS, CLAUDE, *sorted(AGENTS_DIR.rglob("*.md"))]
    if CONTRIBUTING.is_file():
        candidates.append(CONTRIBUTING)
    pattern = re.compile(r"\.agents/[A-Za-z0-9_./-]+")
    for path in candidates:
        text = read(path)
        for match in pattern.findall(text):
            reference = match.rstrip(".,;:")
            if "workspace/" in reference or "history/" in reference or "<" in reference:
                continue
            if not (ROOT / reference).exists():
                fail(f"{rel(path)}: dangling harness reference '{reference}'")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                fail(f"{rel(path)}: dangling relative link '{target}'")


def check_gitignore() -> None:
    gitignore = ROOT / ".gitignore"
    if not gitignore.is_file():
        fail("missing .gitignore")
        return
    entries = {line.strip().lstrip("/").rstrip("/") for line in read(gitignore).splitlines()}
    for required in (".agents/workspace", ".agents/history"):
        if required not in entries:
            fail(f".gitignore does not ignore '{required}/'")
    try:
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--", ".agents/workspace", ".agents/history"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"could not run git ls-files: {exc}")
        return
    if tracked:
        fail("agent workspace/history files are tracked by git:\n" + tracked)


def main() -> int:
    if not AGENTS.is_file() or not RUN_DOC.is_file():
        check_required_files()
    else:
        agents_text = read(AGENTS)
        run_text = read(RUN_DOC)
        check_required_files()
        prompt_texts = check_prompts(agents_text, run_text)
        check_roles(agents_text, prompt_texts)
        check_skills(agents_text)
        check_references()
        check_gitignore()

    if errors:
        print("agent harness check failed:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print("agent harness check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
