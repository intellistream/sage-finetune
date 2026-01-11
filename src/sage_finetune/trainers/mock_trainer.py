"""Mock trainer for testing without GPU/heavy dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Try importing SAGE base class
try:
    from sage.libs.finetune.interface.base import FineTuner, TrainingConfig

    _HAS_SAGE = True
except ImportError:
    FineTuner = object
    _HAS_SAGE = False


@dataclass
class TrainingConfig:
    """Training configuration."""

    model_name_or_path: str = "mock-model"
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-5
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }


@dataclass
class TrainingResult:
    """Result from training."""

    train_loss: float
    eval_loss: float | None = None
    training_time: float = 0.0
    num_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MockTrainer(FineTuner):
    """Mock trainer for testing fine-tuning pipeline.

    This trainer simulates fine-tuning without actual model training.
    Useful for testing data pipelines and integration.

    Example:
        >>> trainer = MockTrainer()
        >>> config = TrainingConfig(num_train_epochs=1)
        >>> result = trainer.train(train_data, config=config)
        >>> print(f"Loss: {result['train_loss']:.4f}")
    """

    def __init__(self, simulate_time: float = 0.1) -> None:
        """Initialize mock trainer.

        Args:
            simulate_time: Time to simulate per epoch (seconds).
        """
        self._simulate_time = simulate_time
        self._model_loaded = False
        self._trained = False

    @property
    def name(self) -> str:
        """Return trainer name."""
        return "mock"

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        config: TrainingConfig | None = None,
    ) -> dict[str, Any]:
        """Simulate training.

        Args:
            train_dataset: Training data (list or iterable).
            eval_dataset: Evaluation data (optional).
            config: Training configuration.

        Returns:
            Training metrics dictionary.
        """
        config = config or TrainingConfig()
        start_time = time.time()

        # Count samples
        if hasattr(train_dataset, "__len__"):
            num_samples = len(train_dataset)
        else:
            num_samples = sum(1 for _ in train_dataset)

        # Simulate training
        for _epoch in range(config.num_train_epochs):
            time.sleep(self._simulate_time)

        training_time = time.time() - start_time

        # Generate mock metrics
        train_loss = 2.5 / (config.num_train_epochs + 1)
        result = {
            "train_loss": train_loss,
            "training_time": training_time,
            "num_samples": num_samples,
            "num_epochs": config.num_train_epochs,
        }

        if eval_dataset is not None:
            result["eval_loss"] = train_loss * 1.1

        self._trained = True
        return result

    def evaluate(self, eval_dataset: Any) -> dict[str, float]:
        """Simulate evaluation.

        Args:
            eval_dataset: Evaluation data.

        Returns:
            Evaluation metrics.
        """
        if hasattr(eval_dataset, "__len__"):
            num_samples = len(eval_dataset)
        else:
            num_samples = sum(1 for _ in eval_dataset)

        return {
            "eval_loss": 0.8,
            "perplexity": 2.2,
            "num_samples": num_samples,
        }

    def save_model(self, output_dir: str) -> None:
        """Simulate saving model.

        Args:
            output_dir: Output directory.
        """
        # In mock, just record the path
        self._output_dir = output_dir

    def load_model(self, model_path: str) -> None:
        """Simulate loading model.

        Args:
            model_path: Path to model.
        """
        self._model_loaded = True
        self._model_path = model_path

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate mock response.

        Args:
            prompt: Input prompt.
            **kwargs: Generation parameters.

        Returns:
            Mock generated text.
        """
        return f"[MockTrainer response to: {prompt[:50]}...]"
