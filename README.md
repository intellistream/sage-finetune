# sage-finetune

Fine-tuning component implementations for SAGE L3.

## Installation

```bash
pip install isage-finetune
```

For LoRA training:
```bash
pip install isage-finetune[peft]
```

## Features

- **LoRA Trainer**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Mock Trainer**: Testing trainer for pipeline validation
- **JSON/JSONL Loader**: Flexible data loading for instruction and chat formats

## Boundary

- `FinetuneManager` only manages task scheduling and status lifecycle.
- Trainers only implement training/evaluation behavior.
- Package registration is explicit and fail-fast.

## Quick Start

```python
from sage_libs.sage_finetune import MockTrainer, JSONDatasetLoader

# Load training data
loader = JSONDatasetLoader()
train_data = loader.load("train.jsonl")

# Train (mock for testing)
trainer = MockTrainer()
result = trainer.train(train_data)
print(f"Loss: {result['train_loss']:.4f}")
```

### LoRA Fine-tuning

```python
from sage_libs.sage_finetune import LoRATrainer
from sage_libs.sage_finetune.trainers.lora_trainer import LoRAConfig

trainer = LoRATrainer(
    model_name="gpt2",
    lora_config=LoRAConfig(r=8, lora_alpha=16),
)

result = trainer.train(train_dataset)
trainer.save_model("./my_lora_model")
```

## Data Formats

### Instruction Format
```json
{"instruction": "Summarize this text", "input": "Long text...", "output": "Summary..."}
```

### Chat Format
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
```

## Integration with SAGE

Components register into the SAGE finetune factory on package import:

```python
from sage.libs.finetune import create_trainer

trainer = create_trainer("lora", model_name="gpt2")
```

## License

Apache 2.0
