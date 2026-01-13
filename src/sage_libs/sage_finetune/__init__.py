"""SAGE Finetune - Fine-tuning implementations for LLMs.

This package provides fine-tuning tools for SAGE pipelines:
- LoRA Trainer: Low-rank adaptation fine-tuning
- Full Trainer: Full parameter fine-tuning (mock for testing)
- Dataset loaders: JSON/JSONL data loading
"""

# Auto-register with SAGE if available
from . import _register as _  # noqa: F401
from ._version import __author__, __email__, __version__

# Data loaders
from .data import JSONDatasetLoader

# Trainers
from .trainers import LoRATrainer, MockTrainer

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "LoRATrainer",
    "MockTrainer",
    "JSONDatasetLoader",
]
