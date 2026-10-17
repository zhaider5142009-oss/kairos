# Contributing to KAIROS

Thanks for your interest in making KAIROS better. This document explains how
to contribute. KAIROS is a research-oriented, **paper-first** market engine —
contributions should preserve that doctrine.

## Ground rules

1. **Never fabricate data.** A missing data source must be reported, not
   guessed or silently zero-filled.
2. **Fail closed.** Execution must be blocked unless every risk gate passes.
   Do not weaken `RiskGate` to make a demo work.
3. **Prove it on paper first.** No change should move the default execution
   path toward live orders without walk-forward + paper validation.
4. **Keep secrets out of the repo.** `config.yaml` is git-ignored and holds
   real API keys. Commit only placeholder values in `config.example.yaml`.

## Getting set up

```bash
git clone https://github.com/zhaider5142009-oss/kairos.git
cd kairos
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"              # core deps + pytest
cp config.example.yaml config.yaml    # optional; the engine runs without it
```

Install optional extras as needed:

| Extra | What it adds |
|---|---|
| `.[neural]` | `torch` — LSTM/GRU models |
| `.[quantum]` | `qiskit` |
| `.[mt5]` | `MetaTrader5` bridge (Windows) |
| `.[all]` | everything |

## Making a change

- Work on a branch off `main`: `git checkout -b feat/your-idea`.
- Keep diffs focused. One concern per PR.
- Add or update a test in `tests/` when you change behaviour.
- Run the checks before opening a PR:

```bash
pytest                 # smoke + unit tests
python -m kairos health
python -m kairos predict GOLD WTI BTC --horizon 5
```

## Pull requests

- Fill in the template: what changed, why, and how you verified it.
- CI (`.github/workflows/ci.yml`) must be green.
- A maintainer reviews every PR; expect at least one round of feedback.

## Reporting issues

Use GitHub Issues. Include the exact command, the instrument, the horizon, and
the full traceback. Redact any API keys before pasting logs.

For **security** issues, see [SECURITY.md](SECURITY.md) — do not open a public
issue.

## Style

- Python 3.10+, type hints on public functions, `from __future__ import annotations`.
- Modules stay import-light: heavy/optional dependencies are guarded imports so
  `python -m kairos` never fails on a missing `torch`.
- Follow the existing naming and comment density; add comments only when the
  *why* is non-obvious.
