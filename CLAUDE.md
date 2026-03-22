# Project Guidelines

## Linting

When running lint checks, use **pyright** and **ruff** only. Do not use **mypy** — the ML ecosystem (torch, transformers, datasets) has incomplete type stubs that produce false positives.
