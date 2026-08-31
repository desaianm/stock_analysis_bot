# Stock Analysis Bot

Discord bot and local research workflows for stock screening, single-company analysis, undervalued-candidate discovery, and portfolio performance tracking. Built on the [Agno](https://github.com/agno-agi/agno) multi-agent framework with OpenAI models, `yfinance` for market data, and SQLite for local persistence.

## Slash commands

- `/top20` — three-phase portfolio construction: screening → tournament ranking → final allocation.
- `/undervalued` — **Quant funnel** over S&P 500 + S&P 600 + TSX Composite (~1,325 tickers). Stages: universe → numeric gate → sector-relative ranking → reverse-DCF margin of safety → insider trades (Financial Datasets API) → agent deep-dive with strict JSON output. Reddit is a supporting catalyst signal only — not the discovery driver.
- `/analyze <company_name>` — resolve company name to ticker, then run the multi-agent single-stock report.
- `/portfolio` — view tracked-holding performance and learning insights.
- `/help` — list commands.

### Watchlist web UI ("Tradesheet")

A Flask dashboard for tracking saved/watchlist stocks with live price charts. Light editorial theme (off-white paper, Fraunces italic + IBM Plex Mono, saffron accent).

```bash
PYTHONPATH=. python scripts/run_web.py   # serves on http://127.0.0.1:5050
```

Features: every stock surfaced into `stock_finds`, full thesis on click, live 1y price chart per stock with 1M/6M/1Y/5Y range toggle, 5-star interest rating, free-text notes, performance since added. Watchlist persists to `state/watchlist.json` (no DB migration required). The library table **lazy-loads live yfinance quotes** (4 concurrent fetches) for any row missing price/market cap/sector — saffron-colored values indicate freshly-fetched data not stored in the DB. This patches the legacy `screening` rows whose original regex-extraction couldn't reliably capture structured fields.

## Architecture

Each slash command kicks off an Agno flow (`stockbot/flows/<name>.py`). A flow owns:

- A set of Agno `Agent` instances (one per analytical role: screening, turnaround, tournament, etc.).
- A shared toolbelt of `yfinance` wrappers + web search (Tavily / EXA / DDGS fallback).
- A SQLite-backed persistence layer that records every run, candidate, holding, and learning insight.
- A JSON state file (`state/<run_type>_state.json`) updated at each phase so crashes leave a breadcrumb.

## Models

The bot uses a three-tier OpenAI model split for cost control. Overrides live in each flow's `__init__`:

| Role | Model | Used for |
|---|---|---|
| Reasoning | `gpt-5.4-mini` | screening, turnaround, fundamental, tournament agents |
| Summary | `gpt-5.4-nano` | Reddit batch summaries, BayStreet scout |
| Extraction | `gpt-4.1-nano` | JSON ticker extraction, company lookup |

## Project layout

```text
main.py                            Discord entrypoint + slash commands
stockbot/flows/
    undervalued.py                 /undervalued — funnel-first flow with deep-dive agent
    recommendations.py             /top20      — screening → tournament → allocation
    single_stock.py                /analyze    — five-agent single-stock report
    company_lookup.py              Name → ticker resolver (gpt-4.1-nano + Tavily)
    performance_tracker.py         /portfolio  — tracked-holding performance + insights
stockbot/screening/                Quant funnel package (used by /undervalued)
    universe.py                    S&P 500 + S&P 600 + TSX Composite loader (Wikipedia, 24h cache)
    numeric_screen.py              Parallel yfinance scan (ThreadPoolExecutor) + hard gates
    ranking.py                     Sector-relative percentile ranking + composite value score
    valuation.py                   Reverse-DCF implied growth across 8/10/12% discount rates
    funnel.py                      Orchestrator chaining stages 1-5 → top-N shortlist
stockbot/tools/
    data.py                        yfinance + Tavily + EXA wrappers (local BaseTool shim)
    insider.py                     Financial Datasets API — Form 4 insider trades + signal score
    performance_tools.py           portfolio price update + learning insights
stockbot/web/
    app.py                         Flask dashboard ("Tradesheet" light theme)
    watchlist.py                   JSON-backed watchlist CRUD (state/watchlist.json)
stockbot/database/                 SQLite schema + managers
stockbot/scheduler/daily_tracker.py Daily 5 PM ET portfolio update (price verification)
stockbot/audit.py                  JSON state file writers (state/<run_type>_state.json)
prompts/
    undervalued/                   value-screening prompts + funnel deep-dive schema
    top20/                         portfolio-construction flow prompts
    single_stock/                  single-company flow prompts
    performance_tracker/           performance-flow prompts
    agents/company_lookup_agent.txt company resolver prompt
```

Generated files (gitignored):

```text
logs/                              Agno debug logs (one file per run)
outputs/<flow>/                    Per-phase markdown reports
plots/                             Generated chart PNGs
reports/                           Discord-uploaded reports (deleted after send)
state/                             In-flight JSON state files
stock_analysis.db                  Local SQLite database
agno.db                            Local Agno state
```

## Setup

Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` (do not commit):

```bash
OPENAI_API_KEY=...
DISCORD_TOKEN=...
TAVILY_API_KEY=...                 # optional; web search falls back to DDGS
EXA_API_KEY=...                    # optional; alternative web search
REDDIT_CLIENT_ID=...               # optional; enables authenticated Reddit catalyst requests
REDDIT_CLIENT_SECRET=...           # optional; paired with REDDIT_CLIENT_ID
REDDIT_USER_AGENT=...              # optional; defaults to stock-analysis-bot/1.0
```

## Run

Discord bot:

```bash
python main.py
```

Direct flow runs (for development; consume API quota):

```bash
python -m stockbot.flows.undervalued       # full /undervalued funnel run (default prefs)
python -m stockbot.flows.recommendations   # full /top20 run
python stockbot/flows/single_stock.py      # prompts for ticker
PYTHONPATH=. python scripts/run_funnel_under20.py   # /undervalued with max_price=$20 preset
PYTHONPATH=. python scripts/run_web.py              # launch the watchlist UI
```

Smoke tests (live API calls, but tiny):

```bash
PYTHONPATH=. python scripts/smoke_models.py            # verify model IDs resolve
PYTHONPATH=. python scripts/smoke_agent_with_tools.py  # agent calls a yfinance tool
PYTHONPATH=. python scripts/smoke_company_lookup.py    # name → ticker resolver
PYTHONPATH=. python scripts/smoke_undervalued_init.py  # flow construction + startup ritual
PYTHONPATH=. python scripts/smoke_funnel.py            # funnel-first flow on 150-ticker subset
PYTHONPATH=. python scripts/probe_apis.py              # check Polygon / Financial Datasets endpoints
python test.py                                         # data-tool smoke (no model calls)
RUN_FLOW_TESTS=1 python test.py                        # also runs all three flows (slow + costly)
```

### The quant funnel (Stages 1-6)

`/undervalued` runs deterministic stages 1-5 before invoking the LLM:

```
1. Universe        ~1,325 tickers   (S&P 500 + S&P 600 + TSX Composite)
2. Numeric screen  ~30s             (parallel yfinance, hard gates from preferences)
3. Ranking         instant          (sector-relative percentiles, composite value score)
4. Reverse-DCF     instant          (implied growth at 8/10/12% discount rates)
5. Insider trades  ~20s             (Financial Datasets Form 4, 180-day net buying)
6. Agent deep-dive ~15s             (gpt-5.4-mini writes JSON thesis on top-10 only)
```

Composite ranking = sector value score + reverse-DCF margin-of-safety + insider signal. Reddit is consulted only as a Stage-5 catalyst overlay on the shortlist, never as the discovery driver.

**Required keys**: `OPENAI_API_KEY`, `FINANCIAL_DATASETS_API_KEY`. Optional fallbacks for web search: `TAVILY_API_KEY`, `EXA_API_KEY`. Polygon's free tier paywalls the bulk endpoints, so we use yfinance for screening and Financial Datasets for insider data.

## Data sources

- `yfinance` — primary for prices, quotes, financials, news, options, analyst recs.
- Tavily REST API — web search when `TAVILY_API_KEY` is set.
- EXA — web search when `EXA_API_KEY` is set (used by `WebSearchTool`).
- DDGS (`ddgs` package) — web search fallback when neither is configured.

The undervalued flow rejects candidates before persistence when required values are missing or fail configured hard screens: market cap, price range, volume, P/E, current ratio, debt/equity. See `_screen_candidate_metrics` in `stockbot/flows/undervalued.py`.

## Development notes

- Keep generated artifacts out of commits: `logs/`, `outputs/`, `plots/`, `reports/`, `state/`, `.DS_Store`, and local databases are gitignored.
- New flow-specific prompts go under `prompts/<flow_name>/`.
- New reusable tools go in `stockbot/tools/data.py` (extend the local `BaseTool` shim — no third-party base class needed).
- Database schema changes go in `stockbot/database/schema.sql` plus the matching manager.
- Use the smoke scripts before any full flow run — they cost cents and surface wiring bugs immediately.

## Troubleshooting

- Slash command times out → check the latest file in `logs/`. Every run writes `logs/{HHMMSS_YYYYMMDD}.log`.
- Empty Canadian market data → confirm the Yahoo suffix (`.TO`, `.V`, `.CN`, `.NE`); `_resolve_yf_symbol` in `undervalued.py` scores candidates against live history.
- `/undervalued` saves zero candidates → inspect the funnel `stats` block in the final report; check whether `stage_2_gate_survivors` is the bottleneck (gates too tight) or `stage_5_shortlist_size` is empty (no candidates pass DCF + insider weighting).
- Funnel returns only 1-2 picks unexpectedly → the 24-hour cache in `state/numeric_screen_cache.json` may be stale from an earlier small-universe smoke run. Delete it to force a fresh full-universe scan.
- Many yfinance fetches failing (high `stage_2_fetched_failed` count) → yfinance's unofficial rate-limiter throttling the 10 concurrent workers. Lower `workers=` on `QuantFunnel(...)` or add backoff.
- Web search fails → either configure `TAVILY_API_KEY` / `EXA_API_KEY` or confirm `ddgs` is installed.
- `state/undervalued_state.json` exists after a crash → carries the last phase the previous run reached. Safe to delete after diagnosing.
- Library row in the watchlist UI shows `—` → legacy `screening` rows had unreliable regex extraction; the UI lazy-loads live quotes (saffron-colored cells). No backfill needed.
