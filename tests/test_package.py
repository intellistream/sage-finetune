"""Tests for package structure and imports."""


class TestPackageImports:
    """Tests for package imports."""

    def test_import_package(self):
        """Test that main package imports correctly."""
        import sage_libs.sage_finetune as sage_finetune
        from sage_libs.sage_finetune._version import __version__ as pkg_version

        assert hasattr(sage_finetune, "__version__")
        assert sage_finetune.__version__ == pkg_version

    def test_import_trainers(self):
        """Test importing trainers."""
        from sage_libs.sage_finetune import LoRATrainer, MockTrainer

        assert MockTrainer is not None
        assert LoRATrainer is not None

    def test_import_loaders(self):
        """Test importing data loaders."""
        from sage_libs.sage_finetune import JSONDatasetLoader

        assert JSONDatasetLoader is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import sage_libs.sage_finetune as sage_finetune

        expected = {
            "__version__",
            "__author__",
            "__email__",
            "LoRATrainer",
            "MockTrainer",
            "JSONDatasetLoader",
            "FinetuneManager",
            "finetune_manager",
            "FinetuneStatus",
            "FinetuneTask",
        }

        assert set(sage_finetune.__all__) == expected

    def test_subpackage_imports(self):
        """Test importing from subpackages."""
        from sage_libs.sage_finetune.data import JSONDatasetLoader
        from sage_libs.sage_finetune.trainers import LoRATrainer, MockTrainer

        assert MockTrainer is not None
        assert LoRATrainer is not None
        assert JSONDatasetLoader is not None


class TestRegistration:
    """Tests for SAGE registration."""

    def test_registration_status(self):
        """Test registration status function."""
        from sage_libs.sage_finetune._register import is_registered

        assert is_registered() is True
