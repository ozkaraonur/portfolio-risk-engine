# Portfolio Risk Engine

Multi-broker portfolio risk and stress-testing engine. Milestone 1: project skeleton, domain models, price providers (synthetic GBM + Stooq) and a basic CLI.

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pre summary examples/portfolio.json --provider synthetic --seed 42
pytest && ruff check . && mypy --strict
```
