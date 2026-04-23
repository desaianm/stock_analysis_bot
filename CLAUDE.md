# CLAUDE.md

Guidance for Claude Code and other coding agents working in this repository.

## Project Snapshot

This is a Discord stock-analysis bot with local agent workflows for:

- top-20 stock recommendation reports
- undervalued and turnaround screening
- single-company research
- portfolio performance tracking and learning insights

Primary runtime entrypoint: `main.py`.

## Important Paths

```text
stockbot/flows/                 Analysis workflows
stockbot/tools/                 Data, search, charting, and performance tools
stockbot/database/              SQLite schema and database managers
stockbot/scheduler/             Daily portfolio update scheduler
prompts/                        Prompt templates used by agents
outputs/, logs/, plots/         Generated local artifacts, ignored by git
```

## Data And Search

- Financial statements and quote data use `yfinance` first.
- Tavily is optional when `TAVILY_API_KEY` is configured.
- Search falls back through `ddgs` when Tavily is unavailable.
- QuickFS is legacy fallback-only and should not be treated as the primary data path.

## Development Rules

- Do not commit `.env`, local SQLite DBs, logs, outputs, plots, or `.DS_Store`.
- Keep generated analysis reports out of source control.
- Prefer focused smoke tests before full agent runs because full flows consume API quota.
- When editing undervalued screening, preserve code-level candidate validation before database persistence.

## Useful Commands

```bash
source venv/bin/activate
python main.py
python stockbot/flows/undervalued.py
python test.py
python -m py_compile stockbot/tools/data.py stockbot/flows/undervalued.py
```

See `README.md` for setup and usage details.
