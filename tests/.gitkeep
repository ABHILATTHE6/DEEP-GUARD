# Development Guide

## Requirements

- Python 3.11+
- Git
- Node.js 20+ for the frontend when it is added
- Optional NVIDIA GPU for model training/inference

## Local setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev,ml]"
```

## Quality checks

```bash
ruff check .
pytest
mypy backend ml
```

## Development principles

- Keep data preparation deterministic and documented.
- Separate training, evaluation, and inference code.
- Version models and record their training configuration.
- Never commit credentials or raw restricted datasets.
- Report uncertainty and limitations with model results.
