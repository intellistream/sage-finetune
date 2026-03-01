from __future__ import annotations

from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "sage_libs" / "sage_finetune"


def test_manager_does_not_expose_trainer_model_listing_responsibility() -> None:
    manager_source = (_package_root() / "manager.py").read_text(encoding="utf-8")
    assert "def list_available_models(" not in manager_source
    assert "def get_current_model(" not in manager_source


def test_register_module_has_no_importerror_branch() -> None:
    register_source = (_package_root() / "_register.py").read_text(encoding="utf-8")
    assert "except ImportError" not in register_source
    assert "_SAGE_REGISTERED" not in register_source
