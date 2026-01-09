# SAGE Finetune

**Lightweight LLM fine-tuning toolkit** for the SAGE ecosystem.

[![PyPI version](https://badge.fury.io/py/isage-finetune.svg)](https://badge.fury.io/py/isage-finetune)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- 📉 **Quantization**: 8-bit and 4-bit quantized training support
- ⚡ **Mixed Precision**: Automatic mixed precision for faster training
- 💾 **Gradient Checkpointing**: Reduce memory usage during training
- 📊 **Monitoring**: TensorBoard and Wandb integration
- 🎯 **Multi-format Data**: Support for various dataset formats
- 🤖 **Agent Fine-tuning**: Specialized tools for agent trajectory training

## Installation

### Basic Installation

```bash
pip install isage-finetune
```

### With Optional Dependencies

```bash
# With TensorBoard support
pip install isage-finetune[tensorboard]

# With Wandb support
pip install isage-finetune[wandb]

# With all optional features
pip install isage-finetune[all]
```

### From Source

```bash
git clone https://github.com/intellistream/sage-finetune.git
cd sage-finetune
pip install -e .
```

## Quick Start

### Basic Fine-tuning

```python
from sage_finetune import LoRATrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    output_dir="./output",
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-4,
)

# Create trainer
trainer = LoRATrainer(config)

# Load and prepare data
dataset = load_training_data("path/to/data.jsonl")

# Train
trainer.train(dataset)
```

### CLI Usage

```bash
# Quick start with preset
sage-finetune train \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --data ./data.jsonl \
    --output ./output \
    --preset small

# Custom configuration
sage-finetune train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data ./data.jsonl \
    --output ./output \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 8 \
    --epochs 5

# Agent trajectory fine-tuning
sage-finetune agent train \
    --trajectories ./agent_data.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output ./agent_model
```

## Agent Fine-tuning

Specialized tools for training agents on task trajectories:

```python
from sage_finetune.agent import (
    AgentTrainer,
    TrajectoryDataFormatter,
    MultiTaskAgentTrainer
)

# Format agent trajectories
formatter = TrajectoryDataFormatter()
formatted_data = formatter.format(raw_trajectories)

# Train with multi-task learning
trainer = MultiTaskAgentTrainer(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    output_dir="./agent_model"
)
trainer.train(formatted_data)
```

## Configuration Presets

Pre-configured settings for different hardware:

```python
from sage_finetune import PresetConfigs

# Small model preset (< 2B parameters)
config = PresetConfigs.SMALL

# Medium model preset (2B-7B parameters)
config = PresetConfigs.MEDIUM

# Large model preset (> 7B parameters)
config = PresetConfigs.LARGE
```

## Data Formats

Supported data formats:

### Instruction Format
```json
{
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing uses quantum mechanics..."
}
```

### Chat Format
```json
{
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]
}
```

### Agent Trajectory Format
```json
{
    "task": "Search for papers about RAG",
    "trajectory": [
        {"thought": "I need to search", "action": "search", "observation": "..."},
        {"thought": "Now summarize", "action": "summarize", "observation": "..."}
    ],
    "final_answer": "Found 10 papers about RAG..."
}
```

## Advanced Features

### Quantization

```python
config = TrainingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # or load_in_4bit=True
)
```

### Gradient Checkpointing

```python
config = TrainingConfig(
    model_name="...",
    gradient_checkpointing=True,  # Reduce memory usage
)
```

### Custom LoRA Configuration

```python
from sage_finetune import LoRAConfig

lora_config = LoRAConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

config = TrainingConfig(
    model_name="...",
    lora_config=lora_config
)
```

## Integration with SAGE

This package can be used standalone or as part of the SAGE ecosystem:

```python
# Standalone usage (this package)
from sage_finetune import LoRATrainer

# SAGE integration (if sage-libs is installed)
from sage.libs.finetune import LoRATrainer
```

## Documentation

- [Full Documentation](https://intellistream.github.io/SAGE/finetune/)
- [API Reference](https://intellistream.github.io/SAGE/finetune/api/)
- [Examples](https://github.com/intellistream/sage-finetune/tree/main/examples)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers >= 4.36.0
- peft >= 0.7.0
- datasets >= 2.14.0

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use SAGE Finetune in your research, please cite:

```bibtex
@software{sage_finetune,
  title = {SAGE Finetune: Lightweight LLM Fine-tuning Toolkit},
  author = {IntelliStream Team},
  year = {2025},
  url = {https://github.com/intellistream/sage-finetune}
}
```

## Acknowledgments

Part of the [SAGE](https://github.com/intellistream/SAGE) project - Streaming AI/LLM data processing pipelines with declarative dataflow.

## Related Projects

- [SAGE](https://github.com/intellistream/SAGE) - Main SAGE framework
- [sage-anns](https://github.com/intellistream/sage-anns) - Approximate nearest neighbor search
- [sage-amms](https://github.com/intellistream/sage-amms) - Approximate matrix multiplication
