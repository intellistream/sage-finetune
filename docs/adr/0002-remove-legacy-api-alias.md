# ADR 0002: Remove old API alias entrypoint

## Status

Accepted

## Context

Issue `intellistream/sage-finetune#6` requires removing old API alias entrypoints and using a single canonical import path.

Observed residue before this change:

- repository tracked generated `src/isage_finetune.egg-info/*` metadata that still contained `sage_finetune` old import examples;
- repository still contained a top-level `finetune/` code tree, outside current packaged module boundary;
- no source package directory exists at `src/sage_finetune`, but old alias text could be reintroduced through generated artifacts.

## Decision

1. Delete tracked generated metadata under `src/isage_finetune.egg-info/*`.
2. Delete the top-level `finetune/` directory directly.
3. Ignore build artifacts (`build/`, `*.egg-info/`) in `.gitignore` to prevent reintroduction.
4. Add regression test to enforce:
   - no `src/sage_finetune` package directory,
   - no top-level `finetune/` directory,
   - no `sage_finetune` import usage in source and README.

## Consequences

- Public import path remains single and explicit: `sage_libs.sage_finetune`.
- Only the canonical import path is supported.
- CI can detect alias reintroduction early.

## Verification

- `ruff check tests/test_issue6_remove_alias_api.py`
- `pytest -q tests/test_issue6_remove_alias_api.py tests/test_package.py`
