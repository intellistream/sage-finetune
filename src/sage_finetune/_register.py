"""Auto-registration of sage-finetune components with SAGE framework."""

from __future__ import annotations

# Try to register with SAGE framework
try:
    from sage.libs.finetune.interface.factory import register_loader, register_trainer

    from .data import JSONDatasetLoader
    from .trainers import LoRATrainer, MockTrainer

    register_trainer("lora", LoRATrainer)
    register_trainer("mock", MockTrainer)
    register_loader("json", JSONDatasetLoader)

    _SAGE_REGISTERED = True

except ImportError:
    _SAGE_REGISTERED = False


def is_registered() -> bool:
    """Check if registered with SAGE."""
    return _SAGE_REGISTERED
