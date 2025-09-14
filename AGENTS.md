# Repository Guidelines

## Project Structure & Module Organization
- Source: `maxsharpe/` (data fetching, optimizer, core workflow, utilities)
  - `data.py`, `optimizer.py`, `core.py`, `utils.py`, `__init__.py`
- CLI entry: `portfolio.py` (also exposed as `maxsharpe` via `pyproject.toml`)
- UI: `streamlit_app.py`
- Tests: `tests/` (pytest discovery configured in `pyproject.toml` and `pytest.ini`)
- Config: `pyproject.toml` (tools, scripts), `requirements.txt`

## Build, Test, and Development Commands
- Install (dev): `pip install -e .[dev]`
- Run tests: `pytest -q`
- Coverage: `pytest --cov=maxsharpe`
- Format: `black . && isort .`
- Lint: `flake8`
- Type check: `mypy maxsharpe`
- CLI example: `python portfolio.py --market CN --years 5 --rf 0.02 --output ./data`

## Coding Style & Naming Conventions
- Formatter: Black (line length 88); Imports: isort (profile=black)
- Lint: flake8; Typing: mypy settings in `pyproject.toml`
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Docstrings: concise, include Args/Returns/Raises where useful; keep language consistent with surrounding files

## Testing Guidelines
- Framework: pytest (`tests/`, files `test_*.py` or `*_test.py`, classes `Test*`, functions `test_*`)
- Run locally: `pytest -q` (strict markers enabled)
- Prefer deterministic tests (fixed seeds), cover edge cases (NaNs, short series, invalid params)
- Aim to cover new/changed logic with unit tests; use `pytest --cov=maxsharpe` when iterating

## Commit & Pull Request Guidelines
- Commits: imperative mood, small scope; example: `fix(utils): handle NaNs via ffill`
- Before PR: ensure `black`, `isort`, `flake8`, `mypy`, and tests pass
- PRs should include: clear description, motivation, brief implementation notes, affected paths, and test evidence; attach screenshots for UI changes (Streamlit)
- Link related Issues and include any breaking-change notes

## Notes & Tips
- Data dependencies use external providers; flaky network may cause intermittent failures—mock in tests
- Keep utilities pure and side‑effect free where possible to simplify testing
