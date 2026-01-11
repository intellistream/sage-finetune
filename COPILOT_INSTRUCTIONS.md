# SAGE Finetune (isage-finetune) - Copilot Instructions

## Package Identity

| 属性 | 值 |
|-----|-----|
| **PyPI 包名** | `isage-finetune` |
| **导入名称** | `sage_finetune` |
| **SAGE 架构层级** | **L3 (Algorithm Library)** |
| **版本格式** | 四段式 `0.0.0.x` |
| **仓库** | `intellistream/sage-finetune` |

## 层级定位

这是一个 **L3 纯算法库**，提供微调训练器和数据加载器的实现。

### ✅ 允许的依赖

- Python 标准库
- `sage-common` (L1) - 通过 SAGE 框架使用时
- `sage-libs` 接口层 (L3) - 注册到 SAGE 工厂
- 第三方库：`transformers`, `peft`, `torch`, `datasets`

### ❌ 禁止的依赖

- 任何 L4+ 层的包 (`sage-middleware`, `sage-kernel`)
- 网络服务、数据库连接（除了 HuggingFace Hub）
- 分布式训练框架（使用 `accelerate` 抽象）

## 功能模块

### 1. 训练器 (Trainers)

```python
from sage_finetune import (
    SFTTrainer,      # Supervised Fine-Tuning
    LoRATrainer,     # LoRA 微调
    QLoRATrainer,    # QLoRA 量化微调
    DPOTrainer,      # Direct Preference Optimization
)

# LoRA 微调
trainer = LoRATrainer(
    model="meta-llama/Llama-2-7b-hf",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)
trainer.train(dataset, output_dir="./output")
```

### 2. 数据加载器 (Data Loaders)

```python
from sage_finetune import (
    InstructionDataLoader,
    ChatDataLoader,
    PreferenceDataLoader,
)

# 指令数据加载
loader = InstructionDataLoader(
    format="alpaca",  # alpaca, sharegpt, openai
)
dataset = loader.load("path/to/data.json")
```

### 3. 配置和工具

```python
from sage_finetune import (
    TrainingConfig,
    LoRAConfig,
    QuantizationConfig,
)

# 训练配置
config = TrainingConfig(
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
)
```

## 目录结构

```
sage-finetune/
├── src/sage_finetune/
│   ├── __init__.py
│   ├── _version.py      # 版本：__version__ = "0.0.0.x"
│   ├── _register.py     # 自动注册到 SAGE 工厂
│   ├── trainers/        # 训练器
│   │   ├── sft.py
│   │   ├── lora.py
│   │   ├── qlora.py
│   │   └── dpo.py
│   ├── data/            # 数据加载
│   │   ├── instruction.py
│   │   ├── chat.py
│   │   └── preference.py
│   └── config/          # 配置
│       ├── training.py
│       └── lora.py
├── tests/
├── pyproject.toml
└── README.md
```

## 与 SAGE 主仓库的关系

### SAGE 侧 (`sage.libs.finetune`)

SAGE 主仓库中的 `sage.libs.finetune` 包含：

1. **接口层** (`sage.libs.finetune.interface`)：
   - 抽象基类：`Trainer`, `DataLoader`, `Config`
   - 工厂函数：`create_trainer()`, `create_data_loader()` 等

### 本包 (`sage_finetune`) 提供

**具体实现**，通过 `_register.py` 自动注册到 SAGE 工厂。

## 常见问题修复指南

### 问题 1：GPU 内存不足

```python
# 使用 gradient checkpointing
trainer = LoRATrainer(
    model="...",
    gradient_checkpointing=True,
)

# 或使用 QLoRA
trainer = QLoRATrainer(
    model="...",
    load_in_4bit=True,
)
```

### 问题 2：数据格式不兼容

```python
# 支持多种格式
loader = InstructionDataLoader(format="alpaca")  # instruction, input, output
loader = InstructionDataLoader(format="sharegpt")  # conversations
loader = InstructionDataLoader(format="openai")  # messages
```

### 问题 3：模型保存格式

```python
# 保存为 safetensors（推荐）
trainer.save_model(output_dir, safe_serialization=True)

# 合并 LoRA 权重
trainer.merge_and_save(output_dir)
```

## 测试

```bash
# 需要 GPU 的测试可能较慢
pytest tests/ -v

# 仅 CPU 测试
pytest tests/ -v -k "not gpu"
```

## 发布

```bash
# 版本递增：修改 src/sage_finetune/_version.py
python -m build
twine upload dist/*
```
