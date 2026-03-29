import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class VersionMetadataTest(unittest.TestCase):
    def test_readme_version_matches_version_number(self):
        version = (REPO_ROOT / "VERSION_NUMBER").read_text(encoding="utf-8").strip()
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (REPO_ROOT / "README_zh.md").read_text(encoding="utf-8")

        self.assertIn(f"version-{version}-blue.svg", readme)
        self.assertIn(f"| {version} | Current development version |", readme)
        self.assertIn(f"版本-{version}-blue.svg", readme_zh)
        self.assertIn(f"| {version} | 当前开发版本 |", readme_zh)

    def test_minimum_python_version_metadata_is_39(self):
        setup_py = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (REPO_ROOT / "README_zh.md").read_text(encoding="utf-8")

        self.assertIn('python_requires=">=3.9"', setup_py)
        self.assertIn('Programming Language :: Python :: 3.9', setup_py)
        self.assertIn("python-%3E%3D3.9-blue.svg", readme)
        self.assertIn("python-%3E%3D3.9-blue.svg", readme_zh)
