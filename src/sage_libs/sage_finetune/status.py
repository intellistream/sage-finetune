"""Fine-tune task status enumeration."""

from __future__ import annotations

from enum import Enum


class FinetuneStatus(str, Enum):
    """Fine-tune task status.
    
    Lifecycle:
        PENDING → PREPARING → TRAINING → COMPLETED/FAILED/CANCELLED
                                     ↓
                                  QUEUED (if another task is running)
    """
    
    PENDING = "pending"        # Task created, not started yet
    PREPARING = "preparing"    # Loading model, preparing data
    TRAINING = "training"      # Actively training
    COMPLETED = "completed"    # Training finished successfully
    FAILED = "failed"          # Training failed with error
    CANCELLED = "cancelled"    # Task cancelled by user
    QUEUED = "queued"         # Waiting for GPU to be free
