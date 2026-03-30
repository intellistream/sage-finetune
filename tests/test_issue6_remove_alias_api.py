from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_removed_sage_finetune_package_dir() -> None:
    removed_pkg = _repo_root() / "src" / "sage_finetune"
    assert not removed_pkg.exists()


def test_no_removed_finetune_directory() -> None:
    removed_dir = _repo_root() / "finetune"
    assert not removed_dir.exists()


def test_no_removed_sage_finetune_imports_in_source_or_readme() -> None:
    source_root = _repo_root() / "src" / "sage_libs" / "sage_finetune"
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    assert "from sage_finetune import" not in readme
    assert "import sage_finetune" not in readme

    for py_file in source_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "from sage_finetune import" not in text
        assert "import sage_finetune" not in text
