"""Fine-tune task data model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .status import FinetuneStatus


@dataclass
class FinetuneTask:
    """Fine-tune task information.
    
    Attributes:
        task_id: Unique task identifier
        model_name: Base model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
        dataset_path: Path to training dataset file
        output_dir: Directory to save fine-tuned model
        status: Current task status
        config: Training configuration (epochs, batch_size, etc.)
        created_at: Task creation timestamp
        started_at: Training start timestamp
        completed_at: Training completion timestamp
        logs: Training log messages
        error_message: Error message if failed
        progress: Training progress (0-100)
        metrics: Training metrics (loss, accuracy, etc.)
    """
    
    task_id: str
    model_name: str
    dataset_path: str
    output_dir: str
    status: FinetuneStatus = FinetuneStatus.PENDING
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    started_at: str | None = None
    completed_at: str | None = None
    logs: list[str] = field(default_factory=list)
    error_message: str | None = None
    progress: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "status": self.status.value,
            "config": self.config,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "logs": self.logs,
            "error_message": self.error_message,
            "progress": self.progress,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinetuneTask:
        """Create task from dictionary."""
        # Convert status string to enum
        if "status" in data and isinstance(data["status"], str):
            data["status"] = FinetuneStatus(data["status"])
        return cls(**data)
