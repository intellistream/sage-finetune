"""LoRA trainer for parameter-efficient fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Try importing SAGE base class
try:
    from sage.libs.finetune.interface.base import FineTuner

    _HAS_SAGE = True
except ImportError:
    FineTuner = object
    _HAS_SAGE = False


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 8  # Rank of update matrices
    lora_alpha: int = 16  # LoRA scaling factor
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class TrainingConfig:
    """Training configuration."""

    model_name_or_path: str = "gpt2"
    output_dir: str = "./lora_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()


class LoRATrainer(FineTuner):
    """LoRA (Low-Rank Adaptation) trainer for parameter-efficient fine-tuning.

    LoRA freezes the pre-trained model weights and injects trainable rank
    decomposition matrices into each layer of the Transformer architecture.

    Requires: pip install isage-finetune[peft]

    Example:
        >>> trainer = LoRATrainer(
        ...     model_name="gpt2",
        ...     lora_config=LoRAConfig(r=8, lora_alpha=16),
        ... )
        >>> result = trainer.train(train_dataset)
        >>> trainer.save_model("./my_lora_model")
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        lora_config: LoRAConfig | None = None,
        training_config: TrainingConfig | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize LoRA trainer.

        Args:
            model_name: HuggingFace model name or path.
            lora_config: LoRA configuration.
            training_config: Training configuration.
            device: Device to use ("auto", "cuda", "cpu").
        """
        self._model_name = model_name
        self._lora_config = lora_config or LoRAConfig()
        self._training_config = training_config or TrainingConfig(model_name_or_path=model_name)
        self._device = device

        # Will be initialized on first train/load
        self._model = None
        self._tokenizer = None
        self._peft_model = None
        self._trainer = None

    @property
    def name(self) -> str:
        """Return trainer name."""
        return "lora"

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LoRA training requires PEFT and Transformers. "
                "Install with: pip install isage-finetune[peft]"
            ) from e

    def _setup_model(self) -> None:
        """Set up model and tokenizer."""
        if self._model is not None:
            return

        self._check_dependencies()

        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map=self._device if self._device != "auto" else "auto",
        )

        # Apply LoRA
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }

        peft_config = LoraConfig(
            r=self._lora_config.r,
            lora_alpha=self._lora_config.lora_alpha,
            target_modules=self._lora_config.target_modules,
            lora_dropout=self._lora_config.lora_dropout,
            bias=self._lora_config.bias,
            task_type=task_type_map.get(self._lora_config.task_type, TaskType.CAUSAL_LM),
        )

        self._peft_model = get_peft_model(self._model, peft_config)

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        config: TrainingConfig | None = None,
    ) -> dict[str, Any]:
        """Train with LoRA.

        Args:
            train_dataset: Training dataset (HuggingFace Dataset or list of dicts).
            eval_dataset: Evaluation dataset (optional).
            config: Training configuration (overrides init config).

        Returns:
            Training metrics dictionary.
        """
        self._setup_model()

        from transformers import (
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        config = config or self._training_config

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            seed=config.seed,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        # Create trainer
        self._trainer = Trainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        train_result = self._trainer.train()

        metrics = {
            "train_loss": train_result.training_loss,
            "training_time": train_result.metrics.get("train_runtime", 0),
        }

        if eval_dataset is not None:
            eval_metrics = self._trainer.evaluate()
            metrics["eval_loss"] = eval_metrics.get("eval_loss", 0)

        return metrics

    def evaluate(self, eval_dataset: Any) -> dict[str, float]:
        """Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset.

        Returns:
            Evaluation metrics.
        """
        if self._trainer is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self._trainer.evaluate(eval_dataset)

    def save_model(self, output_dir: str) -> None:
        """Save the LoRA adapters.

        Args:
            output_dir: Directory to save adapters.
        """
        if self._peft_model is None:
            raise RuntimeError("Model not initialized. Call train() first.")

        self._peft_model.save_pretrained(output_dir)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(output_dir)

    def load_model(self, model_path: str) -> None:
        """Load LoRA adapters.

        Args:
            model_path: Path to saved adapters.
        """
        self._check_dependencies()

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load base model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map=self._device if self._device != "auto" else "auto",
        )

        # Load LoRA adapters
        self._peft_model = PeftModel.from_pretrained(self._model, model_path)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs: Any) -> str:
        """Generate text using the fine-tuned model.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        if self._peft_model is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._peft_model.device) for k, v in inputs.items()}

        outputs = self._peft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
