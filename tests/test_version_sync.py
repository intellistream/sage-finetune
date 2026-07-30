"""Regression tests for package version metadata."""

import re
from pathlib import Path


def read_version(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(), re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_runtime_version_matches_project_metadata():
    """The runtime version file must match the build metadata."""
    root = Path(__file__).parents[1]
    project_version = read_version(root / "pyproject.toml", r'^version = "([^"]+)"$')
    runtime_version = read_version(
        root / "src/sage_libs/sage_finetune/_version.py",
        r'^__version__ = "([^"]+)"$',
    )
    assert runtime_version == project_version
