# Instructions for coding agents

This file is read by coding agents (see https://agents.md). Humans: see `CONTRIBUTING.md`.

## What this project is

A chest-versus-abdomen radiograph classifier, rewritten from a Jupyter notebook
into a small Python package to teach software engineering practices. It is a
teaching example, not a medical device. Never claim otherwise in code or docs.

## Architecture (keep it)

`src/hello_world_medical_imaging/`

- `config.py`: experiment settings from a TOML file (paths, hyperparameters, seed).
- `data.py`: data sources. Each returns `(train, validation)` datasets of images in
  `[0, 255]` with binary labels. `load_image_datasets` reads the real folders;
  `synthetic_datasets` generates noise for tests.
- `model.py`: builds and compiles the Keras model. Rescaling and augmentation live inside it.
- `training.py`: `train(model, train_dataset, validation_dataset, epochs)`. It receives the
  model and the data as arguments and knows nothing about files or downloads.
- `predict.py`: classify one image with a trained model.
- `cli.py`: the only module that imports all the others and wires them together.

The data source is injected into training (`docs/adr/0001-inject-the-data-source-into-training.md`).
Keep that boundary: do not make `training.py` or `model.py` read files or configuration.

## Two guard-rails

- KISS: prefer a plain function to a class, a class to a framework.
- YAGNI: do not add abstractions, options or layers for needs that do not exist in this repository yet.

## Commands

```bash
uv sync                      # install (Python 3.12, exact versions from uv.lock)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                # unit tests, integration test and doctests
uv run mkdocs build --strict # documentation
```

All four must pass before a commit. Write the test before the code when fixing a bug or
adding behaviour (`docs/testing.md`).

## Conventions

- Conventional Commits in English (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`, `build:`, `chore:`).
  The version number and changelog are derived from them.
- Docstrings follow PEP 257 and the Google style; put a doctest in them when it documents the function well.
- Use `logging`, never `print`.
- No data, model weights or notebooks outputs in Git (`.gitignore`).
- Do not add dependencies without pinning them in `pyproject.toml` and running `uv lock`.
