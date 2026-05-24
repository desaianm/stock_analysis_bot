# CLAUDE.md

Guidance for Claude Code and other coding agents working in this repository.

## Project Snapshot

Discord stock-analysis bot built on [Agno](https://github.com/agno-agi/agno) with OpenAI models and `yfinance` data. Workflows:

- top-20 portfolio recommendation (`/top20`)
- undervalued + turnaround screening (`/undervalued`)
- single-company research (`/analyze <company>`)
- portfolio performance + learning insights (`/portfolio`)

Primary runtime entrypoint: `main.py`.

## Architecture

Each slash command runs an Agno flow in `stockbot/flows/`. A flow owns its agents, tool wrappers, prompt-loading, persistence, and state. The reference pattern is `stockbot/flows/undervalued.py` — mirror it when adding new flows.

## Important paths

```text
stockbot/flows/                 Analysis workflows (one per slash command)
stockbot/screening/             Quant funnel package (universe → numeric → ranking → DCF → orchestrator)
stockbot/tools/data.py          yfinance + Tavily + EXA wrappers, local BaseTool shim
stockbot/tools/insider.py       Financial Datasets Form 4 wrapper + signal score
stockbot/tools/performance_tools.py  Portfolio price update + insight generation
stockbot/web/                   Flask watchlist UI ("Tradesheet" light theme)
stockbot/database/              SQLite schema + managers
stockbot/audit.py               JSON state file writers (state/<run_type>_state.json)
stockbot/scheduler/             Daily 5 PM ET portfolio update with price verification
prompts/<flow_name>/            Per-flow prompt templates (load_prompt loads these)
outputs/, logs/, plots/, reports/, state/   Generated, all gitignored
```

## Quant funnel architecture (`/undervalued`)

`/undervalued` now runs the deterministic funnel BEFORE invoking the LLM:

```
Stage 1  Universe        ~1,325 tickers (S&P 500 + S&P 600 + TSX Composite)
Stage 2  Numeric screen  10-worker parallel yfinance, hard gates from ValueScreeningPreferences
Stage 3  Ranking         sector-relative percentiles, composite value score
Stage 4  Reverse-DCF     implied growth at 8/10/12% discount; margin-of-safety vs historical FCF growth
Stage 5  Insider trades  Financial Datasets Form 4, 180-day net buying, signal score
Stage 6  Agent deep-dive gpt-5.4-mini writes JSON thesis on top-10 only (strict schema)
```

The agent reasoning happens ONLY on the deterministic shortlist — not on universe discovery. Reddit is a Stage-5 catalyst overlay, never the driver. The agent's output must validate against the JSON schema in `prompts/undervalued/funnel_deep_dive_instructions.txt`.

State files: `state/numeric_screen_cache.json` (24h cache), `state/undervalued_state.json` (in-flight), `state/universe_cache.json` (24h cache), `state/watchlist.json` (user CRUD).

**Cache gotcha**: if the funnel returns very few candidates, check whether `state/numeric_screen_cache.json` is from an earlier small-universe smoke run. Delete to force a fresh full scan.

## Models

Three-tier OpenAI split — see each flow's `__init__`:

- `gpt-5.4-mini` reasoning agents
- `gpt-5.4-nano` summarization
- `gpt-4.1-nano` JSON extraction

## Data and search

- `yfinance` is the primary path for prices, quotes, financials, news, options.
- Tavily REST when `TAVILY_API_KEY` is set; EXA when `EXA_API_KEY` is set; DDGS as fallback.
- No CrewAI, no LangChain, no QuickFS — those were removed during the Agno migration.

## Development rules

- Do not commit `.env`, local SQLite DBs, logs, outputs, plots, reports, state, or `.DS_Store`.
- Keep generated analysis reports out of source control.
- Prefer smoke tests in `scripts/` before full flow runs — full flows consume API quota.
- When editing undervalued screening, preserve code-level candidate validation (`_screen_candidate_metrics`) before database persistence.
- New tools subclass the local `BaseTool` in `stockbot/tools/data.py` — do not add a third-party tool framework back in.

## Useful commands

```bash
source venv/bin/activate
python main.py                                                          # run Discord bot
python -m stockbot.flows.undervalued                                    # run undervalued funnel directly
PYTHONPATH=. python scripts/run_funnel_under20.py                       # under-$20 preset (fresh full scan)
PYTHONPATH=. python scripts/run_web.py                                  # launch watchlist UI on :5050
PYTHONPATH=. python scripts/smoke_models.py                             # verify model IDs resolve
PYTHONPATH=. python scripts/smoke_agent_with_tools.py                   # agent + tool E2E
PYTHONPATH=. python scripts/smoke_company_lookup.py                     # name → ticker resolver
PYTHONPATH=. python scripts/smoke_undervalued_init.py                   # flow construction + ritual
PYTHONPATH=. python scripts/smoke_funnel.py                             # funnel-first flow on 150-ticker subset
PYTHONPATH=. python scripts/probe_apis.py                               # check Polygon / Financial Datasets endpoints
python test.py                                                          # data tool smoke (no LLM)
RUN_FLOW_TESTS=1 python test.py                                         # all three flows (slow, costly)
python -m py_compile stockbot/tools/data.py stockbot/flows/undervalued.py
```

## Watchlist UI

Flask app at `stockbot/web/app.py`. Two routes that matter for development:
- `GET /` — dashboard with watchlist cards + library table
- `GET /stock/<ticker>` — per-stock detail with 1Y price chart (Chart.js, range toggle 1M/6M/1Y/5Y)

The dashboard lazy-loads live yfinance quotes (4 concurrent fetches) for any library row with missing data — this patches legacy `discovery_source='screening'` rows whose original regex extraction couldn't reliably capture prices/sectors. Saffron-colored cells indicate freshly-fetched values not stored in the DB.

Watchlist CRUD lives at `state/watchlist.json` (no DB writes — respects the project DB constraint). Theme: warm off-white (#fbfaf7) "paper" with near-black ink, Fraunces italic for display headings, IBM Plex Mono for data + paragraph body.

See `README.md` for full setup and usage.
