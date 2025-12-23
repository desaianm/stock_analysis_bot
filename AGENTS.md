# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the Discord entrypoint and routes slash commands into CrewAI flows housed under `stockbot/flows/`. Agent factories live in `stockbot/agents/financial.py`, while portfolio analysis crews sit in `stockbot/portfolio/analysis.py`. Shared data adapters, charting helpers, and markdown utilities were consolidated into `stockbot/tools/data.py`, and command-specific tasks reside in `stockbot/tasks/workflows.py`. Generated artefacts land in `reports/`, `outputs/`, and `plots/`. Keep new modules alongside the nearest flow and include a brief module docstring describing the agent’s responsibility.

## Build, Test, and Development Commands
Create an isolated environment with `python -m venv .venv` and `source .venv/bin/activate`, then install dependencies via `pip install -r requirements.txt`. Run the bot locally with `python main.py`; pass `--log-level DEBUG` while diagnosing Discord or CrewAI events. Execute `python test.py` for the current integration smoke test of market data tools. For quick experiments, open `python -i stockbot/agents/financial.py` to reuse agent factories, or `python -m ipdb main.py` to inspect runtime state.

## Coding Style & Naming Conventions
Target Python 3.10+, follow PEP 8, and use 4-space indentation. Functions and modules stay `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`. Group imports as standard library, third party, then local, alphabetized within each block. Annotate public functions with type hints, prefer `dataclasses` when passing structured results between agents, and keep f-string formatting consistent (`f"{value:.2f}"`). Run `python -m py_compile <file>` before pushing if linting is unavailable.

## Testing Guidelines
Expand coverage as you modify agent logic. Prefer pytest and place tests under `tests/` once added; name files after their targets (e.g., `tests/test_tools.py`). Mock external APIs or gate them behind `pytest.mark.external` to avoid consuming QuickFS, OpenAI, or Tavily quotas. Keep golden markdown reports in `reports/snapshots/` and reference them in assertions rather than hitting live services. Update `test.py` only for high-value end-to-end smoke checks.

## Commit & Pull Request Guidelines
Use short, present-tense commit messages similar to `added memory to all agents`. Rebase before opening a PR and collapse fixups locally. Each PR should include a problem statement, bullet summary of changes, evidence of tests run (`python test.py` or `pytest`), and notes on new environment variables or assets. Link related issues or Discord tickets and attach before/after plots when visuals change.

## Configuration & Secrets
Store secrets (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DISCORD_BOT_TOKEN`, QuickFS credentials) in a local `.env`; `python-dotenv` loads them in `main.py`, `stockbot/tasks/workflows.py`, and `test.py`. Mirror any new key requirements in `README.md` and rotate the credential immediately if they surface in logs, reports, or git history.
