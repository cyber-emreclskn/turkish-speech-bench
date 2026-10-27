# Contributing

There are three ways to help, from easiest to most involved.

## 1. Submit results from your hardware

```bash
pip install "turkish-speech-bench[whisper,vosk]"
tsb run -m faster-whisper:small -m vosk:vosk-model-small-tr-0.3 \
        --label "Intel N100 kiosk" -t 4
```

This writes JSON files under `results/<hardware-id>/`. Fork the repo, add those
files and open a pull request. Please use a descriptive `--label` and close other
heavy programs while measuring.

## 2. Add a model

Any `backend:name` from the supported backends works without code changes, e.g.
`hf-ctc:<any CTC model on the Hub>`. If a model is useful for Turkish on CPU, open
an issue with the *Add a model* template so it gets run on the shared hardware.

## 3. Add a backend

Backends live in `src/tsbench/models/`. Implement `STTModel`:

```python
class MyBackend(STTModel):
    backend = "my-backend"
    extra = "my-extra"  # optional dependency group in pyproject.toml

    def prepare(self) -> None: ...  # download weights (not timed)
    def load(self) -> None: ...  # load into memory (timed)
    def transcribe(self, audio) -> str: ...  # 16 kHz mono float32 numpy array
```

then register it in `_BUILTIN` in `models/base.py` and add tests. Keep heavy imports
inside `prepare`/`load` so `tsb` stays fast to start.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest --cov
uv run ruff check . && uv run mypy
```

Tests use a synthetic dataset and a fake model, so they run offline in seconds.
Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/);
release-please turns them into versions and the changelog, and merged release PRs
publish to PyPI.
