import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
SETUP_PATH = REPO_ROOT / "setup.py"
MANIFEST_PATH = REPO_ROOT / "MANIFEST.in"
README_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README_zh.md",
]
MAIN_MODULE_PATH = REPO_ROOT / "xslim" / "__main__.py"


def _load_main_module():
    spec = importlib.util.spec_from_file_location(
        "xslim_cli_main", MAIN_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPackagingStandards(unittest.TestCase):
    def test_pyproject_declares_standard_build_metadata(self):
        pyproject_text = PYPROJECT_PATH.read_text(encoding="utf-8")

        self.assertIn('[build-system]', pyproject_text)
        self.assertIn(
            'build-backend = "setuptools.build_meta"', pyproject_text
        )
        self.assertIn('dynamic = ["version", "dependencies"]', pyproject_text)
        self.assertIn('license-files = ["LICENSE"]', pyproject_text)
        self.assertIn('[project.scripts]', pyproject_text)
        self.assertIn('xslim = "xslim.__main__:main"', pyproject_text)

    def test_setup_py_is_legacy_compatibility_shim(self):
        setup_text = SETUP_PATH.read_text(encoding="utf-8")

        self.assertIn('from setuptools import setup', setup_text)
        self.assertIn('setup()', setup_text)
        self.assertNotIn('install_requires', setup_text)
        self.assertNotIn('find_packages', setup_text)
        self.assertNotIn('version=', setup_text)

    def test_manifest_includes_repository_docs_and_samples(self):
        manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")

        self.assertIn('include README.md', manifest_text)
        self.assertIn('include README_zh.md', manifest_text)
        self.assertIn('graft doc', manifest_text)
        self.assertIn('graft samples', manifest_text)

    def test_readmes_document_standard_install_and_cli(self):
        for readme_path in README_PATHS:
            with self.subTest(readme=readme_path.name):
                readme_text = readme_path.read_text(encoding="utf-8")
                self.assertIn('pip install .', readme_text)
                self.assertIn('pip install -e .', readme_text)
                self.assertIn('xslim --config config.json', readme_text)
                self.assertIn(
                    'python -m xslim --config config.json', readme_text
                )

    def test_cli_help_path_uses_standard_entry_name(self):
        main_module = _load_main_module()
        parser = main_module.build_parser()

        self.assertEqual(parser.prog, 'xslim')

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main_module.main([])

        self.assertEqual(exit_code, 1)
        help_text = stdout.getvalue()
        self.assertIn('usage: xslim', help_text)
        self.assertIn('--config', help_text)
        self.assertIn('--input_path', help_text)
        self.assertIn('--output_path', help_text)


if __name__ == '__main__':
    unittest.main()
