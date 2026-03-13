"""SAGE Fine-tuning Framework."""

from ._version import __author__, __email__, __version__

# Data loaders
from .data import JSONDatasetLoader

# Trainers
from .trainers import LoRATrainer, MockTrainer

# Task management
from .manager import FinetuneManager, finetune_manager
from .status import FinetuneStatus
from .task import FinetuneTask

__all__ = [
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
]
