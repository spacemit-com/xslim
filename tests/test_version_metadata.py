from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_version_matches_version_number():
    version = (REPO_ROOT / "VERSION_NUMBER").read_text(encoding="utf-8").strip()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_zh = (REPO_ROOT / "README_zh.md").read_text(encoding="utf-8")

    assert f"version-{version}-blue.svg" in readme
    assert f"| {version} | Current development version |" in readme
    assert f"版本-{version}-blue.svg" in readme_zh
    assert f"| {version} | 当前开发版本 |" in readme_zh
