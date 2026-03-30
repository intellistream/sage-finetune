"""Auto-registration of sage-finetune components with SAGE framework."""

from __future__ import annotations

from sage.libs.finetune.interface.factory import (
    register_loader,
    register_trainer,
    registered_loaders,
    registered_trainers,
)

from .data import JSONDatasetLoader
from .trainers import LoRATrainer, MockTrainer

if "lora" not in registered_trainers():
    register_trainer("lora", LoRATrainer)
if "mock" not in registered_trainers():
    register_trainer("mock", MockTrainer)
if "json" not in registered_loaders():
    register_loader("json", JSONDatasetLoader)


def is_registered() -> bool:
    """Check registration status."""
    return True
