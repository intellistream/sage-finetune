---
name: sage-finetune
description: Agent for SAGE Finetune tasks, including LoRA training workflow, agent trajectory/SFT processing, and finetune service integration.
argument-hint: Provide target module/path, desired behavior change, model/data assumptions, and validation scope.
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo']
---

# SAGE Finetune Agent

## Use when

- Working in `sage-finetune` on model fine-tuning, LoRA config/training, data formatting, or task lifecycle management.
- Updating agent-specific training components in `finetune/agent/` (tool calling, planning, timing decision related pipelines).
- Adjusting integration for Studio/Control Plane fine-tune orchestration (`manager.py`, `engine.py`, `service.py`).

## Core capabilities

- LoRA fine-tuning pipeline changes (`finetune/config.py`, `finetune/trainer.py`, `finetune/data.py`).
- CLI workflow updates for task setup/run/merge/list/cleanup (`finetune/cli.py`).
- Agent SFT training updates with LoRA/DoRA/LoRA+ and dataset preparation (`finetune/agent/trainer.py`, `finetune/agent/dialog_processor.py`).
- FireAct-style trajectory collection/filter/convert flows (`finetune/agent/trajectory.py`).
- Multi-task mixing and capability evaluation logic (`finetune/agent/multi_task.py`).
- Fine-tune task lifecycle and resource checks for service integration (`finetune/manager.py`, `finetune/engine.py`).

## Input expectations

- Clear target: file/module/function or end-user command path to change.
- Goal type: bug fix, behavior change, refactor, new option, or docs sync.
- Data/model assumptions: model family, quantization mode, dataset format (`instruction/chat/trajectory`).
- Validation scope: quick smoke check vs focused regression check.

## Guardrails

- Keep changes minimal and repository-local; avoid re-introducing SAGE core internals in this repo.
- Flownet-first ecosystem direction: do not add new `ray` imports/dependencies.
- Prefer fail-fast behavior with actionable errors; avoid silent fallback logic.
- Keep dependency changes explicit in `pyproject.toml` (no ad-hoc install-only fixes).
- In conda environments, use `python -m pip` (not plain `pip`).
- Preserve existing public API naming and CLI behavior unless the task explicitly requires breaking changes.

## Working style

1. Read relevant docs and target modules first (`README.md`, then touched package files).
2. Implement the smallest root-cause change consistent with current architecture.
3. Update nearby docs/examples when behavior or CLI parameters change.
4. Run focused checks before broader validation.

## High-signal paths

- `README.md`, `pyproject.toml`, `quickstart.sh`
- `finetune/config.py`, `finetune/trainer.py`, `finetune/data.py`
- `finetune/cli.py`, `finetune/service.py`
- `finetune/agent/config.py`, `finetune/agent/trainer.py`
- `finetune/agent/dialog_processor.py`, `finetune/agent/trajectory.py`, `finetune/agent/multi_task.py`
- `finetune/manager.py`, `finetune/engine.py`

## Validation hints

- Prefer focused smoke checks around touched paths first (e.g., CLI path or trainer initialization path).
- If adding/changing config fields, verify serialization/loading consistency and default-value behavior.
- If changing training flow, verify output artifacts remain stable (`checkpoints/`, `lora_weights/`, `logs/`).
