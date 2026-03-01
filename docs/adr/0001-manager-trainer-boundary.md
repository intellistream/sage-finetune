# ADR 0001: Separate manager scheduling and trainer responsibilities

## Status

Accepted

## Context

Issue `intellistream/sage-finetune#5` requires boundary cleanup between scheduling, status management, and trainer responsibilities.

Findings before this change:

- `FinetuneManager` included model-discovery methods tied to trainer artifacts.
- `_register.py` used optional `ImportError`-based registration behavior.

These patterns blurred module responsibilities and weakened fail-fast constraints.

## Decision

1. Keep `FinetuneManager` focused on task lifecycle and scheduling only.
2. Remove trainer/model-specific manager APIs:
   - `list_available_models`
   - `get_current_model`
3. Remove optional registration logic; use explicit interface dependency and idempotent registration.
4. Enforce fail-fast task persistence loading errors (`_load_tasks`).

## Consequences

- Manager boundary is limited to scheduling and status lifecycle.
- Trainer responsibilities stay in trainer modules.
- Registration/import problems surface immediately instead of being masked.
