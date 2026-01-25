"""Fine-tune task manager (singleton)."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .status import FinetuneStatus
from .task import FinetuneTask


class FinetuneManager:
    """Singleton manager for fine-tune tasks.
    
    Responsibilities:
        - Task lifecycle management (create, start, cancel, delete)
        - Task persistence (save/load from disk)
        - Training process management
        - Status tracking and logging
        - Queue management (one task at a time)
    
    Usage:
        ```python
        from sage_libs.sage_finetune import finetune_manager
        
        # Create task
        task = finetune_manager.create_task(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dataset_path="data.jsonl",
            config={"num_epochs": 3}
        )
        
        # Start training
        success = finetune_manager.start_training(task.task_id)
        
        # Check status
        task = finetune_manager.get_task(task.task_id)
        print(task.status, task.progress)
        ```
    """
    
    _instance: FinetuneManager | None = None
    _lock = multiprocessing.Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize manager (only once)."""
        if self._initialized:
            return
        
        self.tasks: dict[str, FinetuneTask] = {}
        self.active_task_id: str | None = None
        self.processes: dict[str, subprocess.Popen] = {}
        
        # Storage directory
        self.data_dir = Path.home() / ".sage" / "studio_finetune"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_file = self.data_dir / "tasks.json"
        
        # Load existing tasks
        self._load_tasks()
        self._initialized = True
    
    def create_task(
        self,
        model_name: str,
        dataset_path: str,
        config: dict[str, Any] | None = None,
    ) -> FinetuneTask:
        """Create a new fine-tune task.
        
        Args:
            model_name: Base model name
            dataset_path: Path to training dataset
            config: Training configuration
        
        Returns:
            Created task
        """
        task_id = f"finetune_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        output_dir = str(self.data_dir / "outputs" / task_id)
        
        task = FinetuneTask(
            task_id=task_id,
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config or {},
        )
        
        self.tasks[task_id] = task
        self._save_tasks()
        
        return task
    
    def get_task(self, task_id: str) -> FinetuneTask | None:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> list[FinetuneTask]:
        """List all tasks."""
        return list(self.tasks.values())
    
    def start_training(self, task_id: str) -> bool:
        """Start training for a task.
        
        Args:
            task_id: Task ID to start
        
        Returns:
            True if started successfully, False if queued
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # If another task is running, queue this one
        if self.active_task_id and self.active_task_id != task_id:
            self.update_task_status(task_id, FinetuneStatus.QUEUED)
            self.add_task_log(
                task_id,
                f"任务已加入队列，等待 GPU 资源释放（当前运行: {self.active_task_id}）"
            )
            return True
        
        # Mark as preparing
        self.update_task_status(task_id, FinetuneStatus.PREPARING)
        task.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.active_task_id = task_id
        
        # Create output directory
        Path(task.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Start training in background process
        try:
            # Mark as training
            self.update_task_status(task_id, FinetuneStatus.TRAINING)
            self.add_task_log(task_id, f"开始训练: {task.model_name}")
            
            # TODO: Spawn actual training process
            # For now, this is a placeholder
            self.add_task_log(task_id, "训练进程已启动")
            
            self._save_tasks()
            return True
            
        except Exception as e:
            self.update_task_status(task_id, FinetuneStatus.FAILED)
            task.error_message = str(e)
            self.add_task_log(task_id, f"启动失败: {e}")
            self.active_task_id = None
            self._save_tasks()
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: Task ID to cancel
        
        Returns:
            True if cancelled successfully
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Can only cancel running/preparing/queued tasks
        if task.status not in (
            FinetuneStatus.TRAINING,
            FinetuneStatus.PREPARING,
            FinetuneStatus.QUEUED,
        ):
            return False
        
        # Kill process if exists
        if task_id in self.processes:
            process = self.processes[task_id]
            if process.poll() is None:  # Still running
                process.terminate()
                process.wait(timeout=5)
            del self.processes[task_id]
        
        # Update status
        self.update_task_status(task_id, FinetuneStatus.CANCELLED)
        self.add_task_log(task_id, "任务已取消")
        
        # Clear active task
        if self.active_task_id == task_id:
            self.active_task_id = None
            self._start_next_queued_task()
        
        self._save_tasks()
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task (only if not running).
        
        Args:
            task_id: Task ID to delete
        
        Returns:
            True if deleted successfully
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Cannot delete running/preparing/queued tasks
        if task.status in (
            FinetuneStatus.TRAINING,
            FinetuneStatus.PREPARING,
            FinetuneStatus.QUEUED,
        ):
            return False
        
        # Remove from memory
        del self.tasks[task_id]
        self._save_tasks()
        
        return True
    
    def update_task_status(self, task_id: str, status: FinetuneStatus):
        """Update task status."""
        task = self.tasks.get(task_id)
        if task:
            task.status = status
            
            # Update completion time if finished
            if status in (FinetuneStatus.COMPLETED, FinetuneStatus.FAILED, FinetuneStatus.CANCELLED):
                task.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Clear active task and start next queued
                if self.active_task_id == task_id:
                    self.active_task_id = None
                    self._start_next_queued_task()
            
            self._save_tasks()
    
    def add_task_log(self, task_id: str, message: str):
        """Add log message to task."""
        task = self.tasks.get(task_id)
        if task:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            task.logs.append(f"[{timestamp}] {message}")
            self._save_tasks()
    
    def update_task_progress(self, task_id: str, progress: float, metrics: dict[str, Any] | None = None):
        """Update task training progress.
        
        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            metrics: Optional training metrics
        """
        task = self.tasks.get(task_id)
        if task:
            task.progress = progress
            if metrics:
                task.metrics.update(metrics)
            self._save_tasks()
    
    def list_available_models(self) -> list[dict[str, Any]]:
        """List available models (base models + completed fine-tuned models).
        
        Returns:
            List of model info dictionaries
        """
        models = []
        
        # Add completed fine-tuned models
        for task in self.tasks.values():
            if task.status == FinetuneStatus.COMPLETED:
                output_path = Path(task.output_dir)
                merged_path = output_path / "merged_model"
                lora_path = output_path / "lora"
                
                if merged_path.exists():
                    models.append({
                        "name": task.task_id,
                        "path": str(merged_path),
                        "type": "merged",
                        "base_model": task.model_name,
                        "completed_at": task.completed_at,
                    })
                elif lora_path.exists():
                    models.append({
                        "name": task.task_id,
                        "path": str(lora_path),
                        "type": "lora",
                        "base_model": task.model_name,
                        "completed_at": task.completed_at,
                    })
        
        return models
    
    def get_current_model(self) -> str | None:
        """Get currently active model path.
        
        Returns:
            Path to current model, or None
        """
        # Read from environment variable or config
        return os.getenv("SAGE_STUDIO_LLM_MODEL")
    
    def _start_next_queued_task(self):
        """Start next queued task if any."""
        # Find first queued task
        for task in self.tasks.values():
            if task.status == FinetuneStatus.QUEUED:
                self.add_task_log(task.task_id, "GPU 资源已释放，开始训练")
                self.start_training(task.task_id)
                break
    
    def _save_tasks(self):
        """Save tasks to disk."""
        data = {
            task_id: task.to_dict()
            for task_id, task in self.tasks.items()
        }
        
        with open(self.tasks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_tasks(self):
        """Load tasks from disk."""
        if not self.tasks_file.exists():
            return
        
        try:
            with open(self.tasks_file, encoding="utf-8") as f:
                data = json.load(f)
            
            for task_id, task_data in data.items():
                self.tasks[task_id] = FinetuneTask.from_dict(task_data)
        
        except Exception as e:
            print(f"Warning: Failed to load tasks: {e}")


# Singleton instance
finetune_manager = FinetuneManager()
