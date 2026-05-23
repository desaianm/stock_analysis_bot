# Repository Guidelines

## Project Structure & Module Organization

`main.py` is the Discord entrypoint and dispatches slash commands to Agno flows under `stockbot/flows/`:

- `undervalued.py` — funnel-first value flow; deep-dive agent emits strict JSON output
- `recommendations.py` — `/top20` portfolio construction (3 Agno agents)
- `single_stock.py` — `/analyze` multi-agent single-stock report (5 Agno agents)
- `company_lookup.py` — name → ticker resolver (1 Agno agent)
- `performance_tracker.py` — portfolio tracking + learning insights

Quant funnel (used by `/undervalued`):

- `stockbot/screening/universe.py` — S&P 500 + S&P 600 + TSX Composite loader from Wikipedia (24h cache).
- `stockbot/screening/numeric_screen.py` — parallel yfinance scan + hard gates from `ScreeningGates`.
- `stockbot/screening/ranking.py` — sector-relative percentile ranking + composite value score.
- `stockbot/screening/valuation.py` — reverse-DCF: solves for implied growth at multiple discount rates.
- `stockbot/screening/funnel.py` — orchestrator chaining stages 1-5.

Shared infrastructure:

- `stockbot/tools/data.py` — yfinance + Tavily + EXA wrappers. All tools extend a tiny local `BaseTool` shim; no third-party tool base class required.
- `stockbot/tools/insider.py` — Financial Datasets API wrapper for Form 4 insider trades with 0-10 signal score.
- `stockbot/tools/performance_tools.py` — portfolio price update + insight generation.
- `stockbot/web/` — Flask watchlist UI ("Tradesheet" light theme, lazy-loads live quotes for legacy rows).
- `stockbot/database/` — SQLite schema and managers.
- `stockbot/audit.py` — JSON state file writers for in-flight runs.
- `stockbot/scheduler/daily_tracker.py` — daily 5 PM ET portfolio update with pre-flight price verification.
- `prompts/<flow_name>/` — flow-specific prompt templates loaded via `load_prompt(name)`.

Generated artefacts land in `outputs/`, `plots/`, `reports/`, `logs/`, `state/`, and the local SQLite databases. All are gitignored.

## Build, Test, and Development Commands

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py                                       # run the Discord bot
python -m stockbot.flows.undervalued                 # run the funnel directly
PYTHONPATH=. python scripts/run_funnel_under20.py    # under-$20 preset
PYTHONPATH=. python scripts/run_web.py               # launch watchlist UI :5050
PYTHONPATH=. python scripts/smoke_models.py          # cheap model-id check
PYTHONPATH=. python scripts/smoke_funnel.py          # 150-ticker funnel smoke (~$0.05)
PYTHONPATH=. python scripts/probe_apis.py            # Financial Datasets + Polygon endpoint probe
python test.py                                       # data-tool smoke (no LLM calls)
RUN_FLOW_TESTS=1 python test.py                      # also exercise all three flows
```

`python -m py_compile <file>` before pushing if a linter isn't available.

## Coding Style & Naming Conventions

Python 3.10+, PEP 8, 4-space indentation. Functions and modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`. Group imports as standard library → third-party → local, alphabetized within each block. Annotate public functions; prefer `dataclasses` or `pydantic.BaseModel` for structured payloads. Format f-strings consistently (`f"{value:.2f}"`).

When extending a flow, mirror the structure of `undervalued.py`: a flow class owns its agents, holds tool wrappers as methods returning JSON strings, and loads prompts from `prompts/<flow_name>/`.

## Models

Three-tier OpenAI split is the convention:

- `gpt-5.4-mini` — reasoning (main agents)
- `gpt-5.4-nano` — summarization (Reddit batches, lightweight summarisers)
- `gpt-4.1-nano` — JSON extraction (ticker resolution, structured output)

Set these as `reasoning_model_id` / `summary_model_id` / `extraction_model_id` in the flow's `__init__`.

## Testing Guidelines

Stand-alone tool diagnostics live under `stockbot/tools/tests/`. Each `test_<name>_tool.py` is a CLI runner that takes a ticker and prints the tool's live result. Use them when changing `data.py`. Heavier integration tests go in `test.py` at the repo root.

Mock external APIs in unit tests; full flow runs consume real OpenAI quota. Gate live-API tests behind environment variables (e.g. `RUN_FLOW_TESTS=1`).

## Commit & Pull Request Guidelines

Short, present-tense commit subjects (`fix undervalued model id`, `port top20 to agno`). Rebase before opening a PR. Each PR should include: problem statement, bullet summary of changes, evidence of tests run, and notes on new environment variables. Attach before/after screenshots when the Discord output changes.

## Configuration & Secrets

Required for `/undervalued`: `OPENAI_API_KEY`, `FINANCIAL_DATASETS_API_KEY` (insider trades). Required for the Discord bot: `DISCORD_TOKEN`. Optional fallbacks for web search: `TAVILY_API_KEY`, `EXA_API_KEY`. Loaded via `python-dotenv` from `.env`. Mirror new keys in `README.md`; rotate any credential that lands in logs, reports, or git history.
