"""
SAGE Finetune - 轻量级大模型微调工具

这是一个独立的微调模块，可以作为 SAGE 生态的一部分使用，
也可以被其他项目引用。

主要特性:
- LoRA 微调
- 量化训练 (8-bit/4-bit)
- 混合精度训练
- 梯度检查点
- TensorBoard/Wandb 集成
- 支持多种数据格式

使用示例:
    from sage.libs.finetune import LoRATrainer, TrainingConfig

    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        output_dir="./output",
        num_epochs=3,
    )

    trainer = LoRATrainer(config)
    trainer.train(dataset)
"""

# 仅导入轻量模块，避免在包导入阶段触发对历史 SAGE 命名空间的硬依赖。
from .config import LoRAConfig, PresetConfigs, TrainingConfig
from .data import load_training_data, prepare_dataset


# LoRATrainer 延迟导入，使用 __getattr__
def __getattr__(name):
    """延迟导入运行时组件，避免模块加载阶段失败。"""
    if name == "LoRATrainer":
        from .trainer import LoRATrainer

        return LoRATrainer
    if name == "app":
        from .cli import app

        return app
    if name in {
        "FinetuneConfig",
        "FinetuneEngine",
    }:
        from .engine import FinetuneConfig, FinetuneEngine

        return {"FinetuneConfig": FinetuneConfig, "FinetuneEngine": FinetuneEngine}[name]
    if name in {
        "FinetuneManager",
        "FinetuneStatus",
        "FinetuneTask",
        "check_gpu_resources",
        "finetune_manager",
    }:
        from .manager import (
            FinetuneManager,
            FinetuneStatus,
            FinetuneTask,
            check_gpu_resources,
            finetune_manager,
        )

        return {
            "FinetuneManager": FinetuneManager,
            "FinetuneStatus": FinetuneStatus,
            "FinetuneTask": FinetuneTask,
            "check_gpu_resources": check_gpu_resources,
            "finetune_manager": finetune_manager,
        }[name]
    if name == "agent":
        from . import agent

        return agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 核心训练类（延迟导入）
    "LoRATrainer",  # type: ignore[attr-defined]
    # 配置类（轻量级，直接导入）
    "TrainingConfig",
    "LoRAConfig",
    "PresetConfigs",
    # 数据处理（轻量级，直接导入）
    "prepare_dataset",
    "load_training_data",
    # CLI 应用
    "app",
    # Agent 微调子模块（延迟导入）
    "agent",  # type: ignore[attr-defined]
    # Studio backend components (moved from L6 to L3)
    "FinetuneManager",
    "FinetuneStatus",
    "FinetuneTask",
    "finetune_manager",  # Global singleton
    "check_gpu_resources",
]

__version__ = "0.1.0"
