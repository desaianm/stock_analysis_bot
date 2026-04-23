# Stock Analysis Bot

Discord bot and local research workflows for stock screening, single-company analysis, undervalued-candidate discovery, and portfolio performance tracking.

The project combines agent workflows with market-data tools, web search, chart generation, and a local SQLite database. Reports and run logs are generated locally and are intentionally ignored by git.

## Features

- Discord slash commands for stock research:
  - `/top20` builds a portfolio-style recommendation report.
  - `/undervalued` screens TSX/Canadian and US candidates for value and turnaround setups.
  - `/analyze <company_name>` creates a single-company investment report.
  - `/portfolio` summarizes tracked holdings and learning insights.
- Market-data tooling built around `yfinance`, with optional search providers.
- Code-level validation for undervalued-flow candidates before saving them to the database.
- Local persistence for analysis runs, stock finds, holdings, catalysts, and performance snapshots.
- Daily portfolio tracking scheduler started by the Discord bot.

## Project Layout

```text
main.py                         Discord bot entrypoint and slash commands
stockbot/agents/                CrewAI agent factories
stockbot/flows/                 Agno/CrewAI analysis flows
stockbot/tools/                 Market data, search, charting, and performance tools
stockbot/tasks/                 Command-specific task helpers
stockbot/database/              SQLite schema and managers
stockbot/scheduler/             Daily portfolio tracking scheduler
prompts/                        Agent and workflow prompt templates
scripts/                        Local diagnostic and runner scripts
```

Generated local files:

```text
logs/                           Agno/debug logs
outputs/                        Markdown reports
plots/                          Generated chart PNGs
stock_analysis.db               Local SQLite database
agno.db                         Local Agno state
```

These are ignored by git.

## Setup

Use Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
OPENAI_API_KEY=...
DISCORD_TOKEN=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
TAVILY_API_KEY=...              # optional; search falls back when unavailable
EXA_API_KEY=...                 # optional
SERPER_API_KEY=...              # optional
REDDIT_CLIENT_ID=...            # optional for Reddit integrations
REDDIT_CLIENT_SECRET=...        # optional for Reddit integrations
```

Do not commit `.env` or generated databases.

## Run

Start the Discord bot:

```bash
python main.py
```

Run the undervalued flow directly:

```bash
python stockbot/flows/undervalued.py
```

Run focused diagnostics:

```bash
python test.py
python -m py_compile stockbot/tools/data.py stockbot/flows/undervalued.py
```

## Data Sources

The primary financial statement and quote path uses `yfinance`. Web search uses Tavily when `TAVILY_API_KEY` is configured, with fallback search support through `ddgs`. Exa and Serper remain optional where existing tools use them.

The undervalued flow rejects candidates before persistence when required values are missing or fail configured hard screens, such as market cap, price range, volume, P/E, current ratio, or debt/equity.

## Development Notes

- Keep generated artifacts out of commits: `logs/`, `outputs/`, `plots/`, `.DS_Store`, and local database files are ignored.
- Add new flow-specific prompts under `prompts/<flow_name>/`.
- Add reusable tool code under `stockbot/tools/`.
- Add database schema changes to `stockbot/database/schema.sql` and update the matching manager.
- Prefer focused smoke checks over full agent runs when changing a single tool; full runs can spend API quota.

## Troubleshooting

- If a slash command times out, check the latest file in `logs/`.
- If market data is empty for Canadian symbols, verify the Yahoo suffix (`.TO`, `.V`, `.CN`, `.NE`).
- If no candidates are saved after `/undervalued`, inspect the “Code-Level Hard Screen Review” section in the generated final report.
- If search fails, verify optional API keys or install `ddgs`.
