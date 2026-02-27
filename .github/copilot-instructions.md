# sage-finetune Copilot Instructions

## Scope
- Package: `isage-finetune`, import path `sage_libs.sage_finetune`.
- Layer: **L3** — fine-tuning algorithm library; no L4+ dependencies.
- Purpose: Unified fine-tuning interface for LLM/model training workflows in SAGE pipelines.

## Polyrepo Context (Important)
SAGE was restructured from a monorepo into a polyrepo. `sage-finetune` is a **standalone L3 algorithm repo** providing fine-tuning task management, trainers, and status tracking. It integrates with `sage-libs` fine-tuning interfaces.

## Critical rules
- Keep runtime/service-neutral; no L4+ dependencies.
- Do not create new local virtual environments (`venv`/`.venv`); use the existing configured Python environment.
- In conda environments, use `python -m pip` (never plain `pip`).
- `_version.py` is the **sole version source**.
- No fallback logic; fail fast.

## Architecture focus
- `src/sage_libs/sage_finetune/` — main implementation directory.
  - `manager.py` — fine-tuning job manager.
  - `task.py` — task definition and lifecycle.
  - `status.py` — status tracking.
  - `trainers/` — concrete trainer implementations.
  - `data/` — data loading/preparation utilities.
  - `_register.py` — factory registration with `sage-libs`.
  - `_version.py` — version source of truth.

## Dependencies
- **Depends on**: `isage-common` (L1), `isage-libs` (L3 finetune interfaces).
- **Depended on by**: `sage-middleware`, application repos requiring fine-tuning workflows.

## Workflow
1. Make minimal changes under `src/sage_libs/sage_finetune/`.
2. Keep public imports stable in `__init__.py`.
3. Run `pytest tests/ -v` and update docs for behavior changes.

## Development setup
```bash
./quickstart.sh       # installs hooks + pip install -e .[dev]
```

## Git Hooks (Mandatory)
- Never use `git commit --no-verify` or `git push --no-verify`.
- If hooks fail, fix the issue first.
- Run `./quickstart.sh` after cloning to install hooks.
