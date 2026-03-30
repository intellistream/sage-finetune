"""Basic smoke tests for sage-finetune."""


def test_import_package() -> None:
    import sage_finetune

    assert sage_finetune.__version__


def test_import_cli_app() -> None:
    from sage_finetune.cli import app

    assert app is not None
