"""Flask app for the stock watchlist + tracked-stocks dashboard.

Routes:
    GET  /                     dashboard listing watchlist + saved stocks
    GET  /stock/<ticker>       per-stock detail (price chart, thesis, performance)
    POST /watchlist/<ticker>   add to watchlist  (form: interest, notes)
    POST /watchlist/<ticker>/remove
    POST /watchlist/<ticker>/interest  (form: interest)
    GET  /api/prices/<ticker>?range=1y   JSON price history for chart
    GET  /api/snapshot/<ticker>          JSON quote + key metrics

Reads from stock_analysis.db (read-only) and ``state/watchlist.json`` (CRUD).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz
import yfinance as yf
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from stockbot.web.watchlist import (
    add_to_watchlist,
    list_watchlist,
    remove_from_watchlist,
    update_interest,
)

ny_timezone = pytz.timezone("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "stock_analysis.db"

app = Flask(__name__)


# ---------------------------------------------------------------------------
# DB helpers (read-only)
# ---------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _saved_stocks(limit: int = 200) -> List[Dict[str, Any]]:
    """Return distinct tickers from stock_finds, latest find per ticker."""
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT sf.ticker, sf.company_name, sf.sector, sf.industry, sf.exchange,
                   sf.confidence_score, sf.current_price, sf.market_cap, sf.pe_ratio,
                   sf.investment_thesis, sf.discovered_at, sf.discovery_source,
                   sf.analysis_run_id
            FROM stock_finds sf
            INNER JOIN (
                SELECT ticker, MAX(discovered_at) AS latest
                FROM stock_finds
                GROUP BY ticker
            ) latest ON sf.ticker = latest.ticker AND sf.discovered_at = latest.latest
            ORDER BY sf.discovered_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return _rows_to_dicts(cur.fetchall())


def _holdings_by_ticker(ticker: str) -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ticker, entry_price, entry_date, current_price,
                   total_return_pct, holding_days, max_gain_pct, max_drawdown_pct
            FROM portfolio_holdings
            WHERE ticker = ?
            ORDER BY entry_date DESC
            """,
            (ticker.upper(),),
        )
        return _rows_to_dicts(cur.fetchall())


def _finds_by_ticker(ticker: str) -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ticker, company_name, sector, confidence_score,
                   current_price, investment_thesis, discovered_at,
                   discovery_source, analysis_run_id
            FROM stock_finds
            WHERE ticker = ?
            ORDER BY discovered_at DESC
            """,
            (ticker.upper(),),
        )
        return _rows_to_dicts(cur.fetchall())


def _run_summary(run_id: int) -> Optional[Dict[str, Any]]:
    if not DB_PATH.exists() or run_id is None:
        return None
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, run_type, started_at, completed_at, status, total_candidates, final_selections
            FROM analysis_runs WHERE id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _quote(ticker: str) -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        prev_close = info.get("previousClose")
        change_pct = ((price - prev_close) / prev_close * 100) if (price and prev_close) else None
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE") or info.get("forwardPE"),
            "fifty_two_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc)}


def _price_history(ticker: str, period: str = "1y") -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist is None or hist.empty:
            return {"ticker": ticker, "dates": [], "closes": []}
        return {
            "ticker": ticker.upper(),
            "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
            "closes": [round(float(v), 2) for v in hist["Close"].tolist()],
        }
    except Exception as exc:
        return {"ticker": ticker.upper(), "error": str(exc), "dates": [], "closes": []}


def _performance_since(ticker: str, since_iso: str) -> Optional[Dict[str, Any]]:
    """Return {entry_price, current_price, return_pct, days} since `since_iso`."""
    try:
        start_date = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.date().isoformat(), interval="1d")
        if hist is None or hist.empty:
            return None
        entry_price = float(hist["Close"].iloc[0])
        current_price = float(hist["Close"].iloc[-1])
        days = (datetime.now() - start_date.replace(tzinfo=None)).days
        return {
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "return_pct": round((current_price - entry_price) / entry_price * 100, 2),
            "days": days,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tradesheet — Personal Equity Almanac</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT,WONK@0,9..144,300..900,0..100,0..1;1,9..144,300..900,0..100,0..1&family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #fbfaf7;
    --bg-2: #f4f2ec;
    --surface: #ffffff;
    --rule: #d8d3c5;
    --rule-strong: #9c9684;
    --ink: #14110d;
    --ink-mid: #3a342a;
    --ink-mute: #6b6555;
    --ink-faint: #9c9684;
    --saffron: #b85d12;
    --saffron-soft: rgba(184, 93, 18, 0.09);
    --sage: #3f5a1f;
    --sage-soft: rgba(63, 90, 31, 0.09);
    --oxblood: #962e21;
    --oxblood-soft: rgba(150, 46, 33, 0.08);
  }
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--ink); }
  body {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 13px;
    line-height: 1.55;
    font-variant-numeric: tabular-nums slashed-zero;
    background-color: var(--bg);
    background-image:
      radial-gradient(ellipse at top, rgba(184, 93, 18, 0.04), transparent 70%);
    min-height: 100vh;
  }
  .serif { font-family: 'Fraunces', Georgia, serif; font-variation-settings: 'opsz' 144, 'SOFT' 30; }
  .serif-italic { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-variation-settings: 'opsz' 144, 'SOFT' 50, 'WONK' 1; }
  .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums slashed-zero; }
  .label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-mute); font-weight: 500; }
  .micro { font-size: 11px; color: var(--ink-mute); }

  /* Masthead */
  .masthead {
    border-bottom: 2px solid var(--ink);
    background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%);
    position: relative;
  }
  .masthead::after { content: ''; position: absolute; left: 0; right: 0; bottom: -6px; height: 1px; background: var(--ink); opacity: 0.3; }
  .masthead-inner { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; gap: 32px; }
  .brand { font-family: 'Fraunces', serif; font-style: italic; font-weight: 600; font-size: 18px; letter-spacing: -0.01em; color: var(--ink); font-variation-settings: 'opsz' 144, 'SOFT' 100, 'WONK' 1; }
  .brand-dot { color: var(--saffron); }
  .live-dot { display: inline-block; width: 7px; height: 7px; background: var(--saffron); border-radius: 50%; box-shadow: 0 0 0 0 var(--saffron); animation: pulse 2s infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(184, 93, 18, 0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(184, 93, 18, 0); }
    100% { box-shadow: 0 0 0 0 rgba(184, 93, 18, 0); }
  }

  /* Hero */
  .hero-h1 { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 300; font-size: clamp(3.5rem, 8vw, 6rem); line-height: 0.92; letter-spacing: -0.045em; color: var(--ink); font-variation-settings: 'opsz' 144, 'SOFT' 100, 'WONK' 1; }
  .hero-rule { height: 1px; background: var(--rule); margin: 24px 0; position: relative; }
  .hero-rule::before { content: ''; position: absolute; left: 0; top: 0; width: 120px; height: 1px; background: var(--saffron); }

  /* Watchlist row (not boxed — left-rule only) */
  .wrow {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 24px;
    padding: 24px 0 24px 28px;
    border-bottom: 1px solid var(--rule);
    border-left: 1px solid var(--rule);
    position: relative;
    transition: border-left-color 0.3s ease, background 0.3s ease;
  }
  .wrow::before {
    content: '';
    position: absolute;
    left: -1px; top: 0; bottom: 0;
    width: 3px;
    background: var(--saffron);
    transform: scaleY(0);
    transform-origin: top;
    transition: transform 0.4s cubic-bezier(.2,.7,.2,1);
  }
  .wrow:hover { background: linear-gradient(90deg, var(--saffron-soft), transparent 30%); }
  .wrow:hover::before { transform: scaleY(1); }
  .wrow:first-child { border-top: 1px solid var(--rule); }

  .ticker-glyph { font-family: 'Fraunces', serif; font-style: italic; font-weight: 400; font-size: 56px; line-height: 0.9; letter-spacing: -0.04em; color: var(--ink); font-variation-settings: 'opsz' 144, 'SOFT' 80, 'WONK' 1; }
  .ticker-glyph a { color: inherit; text-decoration: none; }
  .ticker-glyph a:hover { color: var(--saffron); }
  .company-name { font-family: 'Fraunces', serif; font-style: italic; font-weight: 300; font-size: 15px; color: var(--ink-mid); margin-top: 2px; }

  .quote-price { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 400; color: var(--ink); letter-spacing: -0.02em; }
  .quote-change { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500; letter-spacing: 0.02em; padding: 2px 8px; border-radius: 0; border: 1px solid; display: inline-block; }
  .pos { color: var(--sage); border-color: var(--sage); background: var(--sage-soft); }
  .neg { color: var(--oxblood); border-color: var(--oxblood); background: var(--oxblood-soft); }
  .neu { color: var(--ink-mute); border-color: var(--rule); }

  /* Star rating as dots */
  .dot-rating { display: inline-flex; gap: 6px; }
  .dot-rating button { background: none; border: 0; padding: 0; cursor: pointer; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; border: 1px solid var(--rule-strong); transition: all 0.2s; }
  .dot-on { background: var(--saffron); border-color: var(--saffron); box-shadow: 0 0 8px var(--saffron-soft); }
  .dot-rating button:hover .dot { transform: scale(1.3); }

  /* Library table — newspaper hairlines */
  .libtable { width: 100%; border-collapse: collapse; }
  .libtable thead th {
    text-align: left;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-mute);
    font-weight: 500;
    border-bottom: 1px solid var(--rule-strong);
    border-top: 1px solid var(--rule-strong);
  }
  .libtable tbody td {
    padding: 14px 16px;
    border-bottom: 1px solid var(--rule);
    vertical-align: middle;
  }
  .libtable tbody tr { transition: background 0.25s; position: relative; }
  .libtable tbody tr:hover { background: var(--saffron-soft); }
  .libtable td.tk a { font-family: 'Fraunces', serif; font-style: italic; font-size: 22px; color: var(--ink); text-decoration: none; font-variation-settings: 'opsz' 144, 'SOFT' 60, 'WONK' 1; }
  .libtable td.tk a:hover { color: var(--saffron); }
  .libtable td.co { color: var(--ink-mid); font-size: 13px; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .libtable td.sec { color: var(--ink-mute); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }

  .conf-pill { font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 2px 8px; border-radius: 0; border: 1px solid; }
  .conf-high { color: var(--saffron); border-color: var(--saffron); background: var(--saffron-soft); }
  .conf-mid  { color: var(--ink-mid); border-color: var(--rule-strong); }
  .conf-low  { color: var(--oxblood); border-color: var(--oxblood); background: var(--oxblood-soft); }
  .source-tag { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-mute); }

  .add-watch-btn { color: var(--ink-mute); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; background: none; border: 1px solid var(--rule); padding: 4px 10px; transition: all 0.2s; cursor: pointer; }
  .add-watch-btn:hover { color: var(--saffron); border-color: var(--saffron); }
  .on-list { color: var(--saffron); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; }

  .remove-btn { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); background: none; border: 0; cursor: pointer; transition: color 0.2s; }
  .remove-btn:hover { color: var(--oxblood); }

  .notes { font-family: 'IBM Plex Mono', 'Courier New', monospace; font-style: normal; font-size: 14px; line-height: 1.7; color: var(--ink-mid); margin-top: 10px; }

  .empty-state { padding: 64px 32px; text-align: center; color: var(--ink-mute); border: 1px dashed var(--rule); }
  .empty-state .serif-italic { font-size: 28px; color: var(--ink-mid); margin-bottom: 8px; display: block; }

  /* Reveal animation on load */
  @keyframes reveal { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .reveal { animation: reveal 0.7s cubic-bezier(.2,.7,.2,1) both; }
  .reveal-1 { animation-delay: 0.05s; }
  .reveal-2 { animation-delay: 0.15s; }
  .reveal-3 { animation-delay: 0.25s; }
  .reveal-4 { animation-delay: 0.35s; }

  ::selection { background: var(--saffron); color: var(--bg); }
</style>
</head>
<body>

<div class="masthead">
  <div class="max-w-[1280px] mx-auto px-8">
    <div class="masthead-inner">
      <div class="flex items-center gap-8">
        <div class="brand">Tradesheet<span class="brand-dot">.</span></div>
        <div class="label hidden md:block">Personal Equity Almanac</div>
      </div>
      <div class="flex items-center gap-6 micro">
        <span><span class="live-dot"></span>LIVE</span>
        <span id="market-clock">—</span>
        <span class="hidden md:inline" id="today-date">—</span>
      </div>
    </div>
  </div>
</div>

<main class="max-w-[1280px] mx-auto px-8 pb-24">

  <!-- Hero -->
  <section class="pt-16 pb-8 reveal reveal-1">
    <div class="label mb-4">§ One · The Watchlist</div>
    <h1 class="hero-h1">Watchlist<span style="color: var(--saffron);">.</span></h1>
    <div class="hero-rule"></div>
    <div class="flex justify-between items-end flex-wrap gap-4 micro">
      <div>
        <span class="mono" style="color: var(--ink);">{{ watchlist|length }}</span> on file
        &nbsp;·&nbsp; <span class="mono" style="color: var(--ink);">{{ saved|length }}</span> in library
        &nbsp;·&nbsp; quotes via yfinance
      </div>
      <div class="label" style="color: var(--ink-faint);">db&nbsp;·&nbsp;{{ db_path.split('/')[-1] }}</div>
    </div>
  </section>

  <!-- Watchlist rows -->
  <section class="mb-24 reveal reveal-2">
    {% if watchlist %}
    <div>
      {% for w in watchlist %}
      <article class="wrow">
        <div>
          <div class="ticker-glyph"><a href="{{ url_for('stock_detail', ticker=w.ticker) }}">{{ w.ticker }}</a></div>
          <div class="company-name">{{ w.quote.name or 'Quote unavailable' }}{% if w.quote.sector %} <span class="micro" style="margin-left: 6px;">· {{ w.quote.sector }}</span>{% endif %}</div>

          <div class="flex items-center gap-6 mt-4 flex-wrap">
            <form method="post" action="{{ url_for('set_interest', ticker=w.ticker) }}" class="dot-rating">
              {% for n in [1,2,3,4,5] %}
                <button type="submit" name="interest" value="{{ n }}" title="set interest {{ n }}">
                  <span class="dot {% if n <= w.interest_level %}dot-on{% endif %}"></span>
                </button>
              {% endfor %}
            </form>

            {% if w.perf %}
            <span class="micro">
              <span class="label" style="margin-right: 6px;">since {{ w.added_at[:10] }} ({{ w.perf.days }}d)</span>
              {% set pcls = 'pos' if w.perf.return_pct >= 0 else 'neg' %}
              <span class="quote-change {{ pcls }}">
                {% if w.perf.return_pct >= 0 %}▲{% else %}▼{% endif %} {{ '%+.2f'|format(w.perf.return_pct) }}%
              </span>
            </span>
            {% endif %}

            <form method="post" action="{{ url_for('remove_watch', ticker=w.ticker) }}" style="margin-left: auto;">
              <button class="remove-btn" type="submit">— remove</button>
            </form>
          </div>

          {% if w.notes %}<div class="notes">"{{ w.notes }}"</div>{% endif %}
        </div>

        <div class="text-right whitespace-nowrap">
          {% if w.quote.price is not none %}
            <div class="quote-price">${{ '%.2f'|format(w.quote.price) }}</div>
            {% if w.quote.change_pct is not none %}
              {% set pcls = 'pos' if w.quote.change_pct >= 0 else 'neg' %}
              <div class="mt-2"><span class="quote-change {{ pcls }}">
                {% if w.quote.change_pct >= 0 %}▲{% else %}▼{% endif %} {{ '%+.2f'|format(w.quote.change_pct) }}%
              </span></div>
            {% endif %}
          {% else %}
            <div class="micro">no quote</div>
          {% endif %}
        </div>
      </article>
      {% endfor %}
    </div>
    {% else %}
    <div class="empty-state">
      <span class="serif-italic">An empty ledger.</span>
      <div class="micro">Add stocks from the library below to begin tracking.</div>
    </div>
    {% endif %}
  </section>

  <!-- Library -->
  <section class="reveal reveal-3">
    <div class="label mb-4">§ Two · The Library</div>
    <h2 class="hero-h1" style="font-size: clamp(2.5rem, 5vw, 4rem);">Library<span style="color: var(--saffron);">.</span></h2>
    <div class="hero-rule"></div>
    <div class="flex justify-between items-baseline mb-6 flex-wrap gap-2">
      <div class="micro"><span class="mono" style="color: var(--ink);">{{ saved|length }}</span> entries · latest find per ticker · sourced from <span class="mono">stock_finds</span></div>
    </div>

    {% if saved %}
    <div style="overflow-x: auto; border: 1px solid var(--rule);">
      <table class="libtable">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Company</th>
            <th>Sector</th>
            <th style="text-align: right;">Conf</th>
            <th style="text-align: right;">Last Px</th>
            <th style="text-align: right;">Mkt Cap</th>
            <th>Source</th>
            <th>Found</th>
            <th style="text-align: right;">Since Find</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {% for s in saved %}
          <tr data-ticker="{{ s.ticker }}" data-discovered="{{ s.discovered_at or '' }}" data-entry-price="{{ s.current_price or '' }}">
            <td class="tk"><a href="{{ url_for('stock_detail', ticker=s.ticker) }}">{{ s.ticker }}</a></td>
            <td class="co" data-field="company">{{ s.company_name or '—' }}</td>
            <td class="sec" data-field="sector">{{ s.sector or '—' }}</td>
            <td style="text-align: right;">
              {% if s.confidence_score %}
                {% set cls = 'conf-high' if s.confidence_score >= 7 else ('conf-mid' if s.confidence_score >= 5 else 'conf-low') %}
                <span class="conf-pill {{ cls }}">{{ '%.1f'|format(s.confidence_score) }}</span>
              {% else %}<span class="micro">—</span>{% endif %}
            </td>
            <td style="text-align: right;" class="mono" data-field="price">{% if s.current_price %}${{ '%.2f'|format(s.current_price) }}{% else %}—{% endif %}</td>
            <td style="text-align: right;" class="mono" data-field="mcap">{% if s.market_cap %}{{ '{:,.0f}'.format(s.market_cap/1e6) }}M{% else %}—{% endif %}</td>
            <td><span class="source-tag">{{ s.discovery_source or '—' }}</span></td>
            <td class="micro mono">{{ s.discovered_at[:10] if s.discovered_at else '—' }}</td>
            <td style="text-align: right;" data-field="perf"><span class="micro">…</span></td>
            <td style="text-align: right;">
              {% if s.ticker in watchlist_tickers %}
                <span class="on-list">★ on list</span>
              {% else %}
                <form method="post" action="{{ url_for('add_watch', ticker=s.ticker) }}" class="add-watch-form" style="display: inline;">
                  <button class="add-watch-btn" type="submit">+ watch</button>
                </form>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <div class="empty-state">
      <span class="serif-italic">The library awaits its first entry.</span>
      <div class="micro">Run <span class="mono">/undervalued</span> in Discord to populate.</div>
    </div>
    {% endif %}
  </section>

  <footer class="mt-24 pt-8 border-t border-[#c5b48d] flex justify-between micro reveal reveal-4">
    <div>Tradesheet · a personal equity almanac</div>
    <div class="label">Composed in CSS &amp; Python</div>
  </footer>
</main>

<script>
  // Clock + market status (US Eastern, rough check by UTC hour)
  function updateClock() {
    const now = new Date();
    const opts = { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false };
    const time = now.toLocaleTimeString('en-US', opts);
    document.getElementById('market-clock').textContent = time + ' ET';
    const dOpts = { timeZone: 'America/New_York', weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' };
    document.getElementById('today-date').textContent = now.toLocaleDateString('en-US', dOpts);
  }
  updateClock();
  setInterval(updateClock, 30000);

  // Intercept "+ watch" form submits → POST via fetch + replace cell in place,
  // so the click feedback is immediate and the page doesn't lose scroll position.
  document.querySelectorAll('.add-watch-form').forEach(form => {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = form.querySelector('button');
      const cell = form.parentElement;
      btn.textContent = '...';
      btn.disabled = true;
      try {
        const res = await fetch(form.action, { method: 'POST', body: new FormData(form) });
        if (res.ok || res.redirected) {
          cell.innerHTML = '<span class="on-list">★ on list</span>';
          // Flash a brief saffron flash on the row
          const row = form.closest('tr');
          if (row) {
            row.style.transition = 'background 0.6s';
            row.style.background = 'var(--saffron-soft)';
            setTimeout(() => { row.style.background = ''; }, 1200);
          }
        } else {
          btn.textContent = '+ watch';
          btn.disabled = false;
          alert('Could not add to watchlist');
        }
      } catch (err) {
        btn.textContent = '+ watch';
        btn.disabled = false;
      }
    });
  });

  // Lazy-fill missing prices: any library row with a `—` in price/mcap/sector
  // gets a live quote fetched on page load. Throttled to 4 concurrent.
  async function fillMissing() {
    const rows = Array.from(document.querySelectorAll('tr[data-ticker]'));
    const needs = rows.filter(r => {
      const priceCell = r.querySelector('[data-field="price"]');
      const mcapCell = r.querySelector('[data-field="mcap"]');
      const secCell = r.querySelector('[data-field="sector"]');
      return (priceCell && priceCell.textContent.trim() === '—') ||
             (mcapCell && mcapCell.textContent.trim() === '—') ||
             (secCell && secCell.textContent.trim() === '—');
    });
    if (needs.length === 0) return;

    // Mark them dim while loading
    needs.forEach(r => r.querySelectorAll('[data-field]').forEach(c => {
      if (c.textContent.trim() === '—') c.style.opacity = '0.4';
    }));

    const limit = 4;
    let cursor = 0;
    async function worker() {
      while (cursor < needs.length) {
        const row = needs[cursor++];
        const ticker = row.dataset.ticker;
        try {
          const r = await fetch(`/api/snapshot/${ticker}`);
          const q = await r.json();
          const priceCell = row.querySelector('[data-field="price"]');
          const mcapCell = row.querySelector('[data-field="mcap"]');
          const secCell = row.querySelector('[data-field="sector"]');
          const coCell = row.querySelector('[data-field="company"]');
          if (priceCell && priceCell.textContent.trim() === '—' && q.price != null) {
            priceCell.textContent = '$' + q.price.toFixed(2);
            priceCell.style.color = 'var(--saffron)';
            priceCell.title = 'lazy-fetched live quote';
          }
          if (mcapCell && mcapCell.textContent.trim() === '—' && q.market_cap != null) {
            mcapCell.textContent = (q.market_cap / 1e6).toLocaleString('en-US', { maximumFractionDigits: 0 }) + 'M';
            mcapCell.style.color = 'var(--saffron)';
          }
          if (secCell && secCell.textContent.trim() === '—' && q.sector) {
            secCell.textContent = q.sector.toUpperCase();
          }
          if (coCell && (coCell.textContent.trim() === '—' || coCell.textContent.trim() === 'Unknown') && q.name) {
            coCell.textContent = q.name;
          }
          row.querySelectorAll('[data-field]').forEach(c => c.style.opacity = '1');
        } catch (e) {
          row.querySelectorAll('[data-field]').forEach(c => c.style.opacity = '1');
        }
      }
    }
    await Promise.all(Array.from({ length: limit }, () => worker()));
  }
  fillMissing();

  // Fill "Since Find" return column for every library row.
  // Uses the row's data-discovered timestamp; calls /api/perf/<ticker>?since=<iso>.
  // This is the "how right was our analysis" signal — return since the bot first
  // discovered the stock.
  async function fillPerf() {
    const rows = Array.from(document.querySelectorAll('tr[data-ticker][data-discovered]'));
    const limit = 4;
    let cursor = 0;
    async function worker() {
      while (cursor < rows.length) {
        const row = rows[cursor++];
        const ticker = row.dataset.ticker;
        const since = row.dataset.discovered;
        const cell = row.querySelector('[data-field="perf"]');
        if (!since || !cell) continue;
        try {
          const res = await fetch(`/api/perf/${ticker}?since=${encodeURIComponent(since)}`);
          const data = await res.json();
          if (data.error || data.return_pct == null) {
            cell.innerHTML = '<span class="micro">—</span>';
            continue;
          }
          const ret = data.return_pct;
          const days = data.days;
          const arrow = ret >= 0 ? '▲' : '▼';
          const cls = ret >= 0 ? 'pos' : 'neg';
          const sign = ret >= 0 ? '+' : '';
          cell.innerHTML = `<span class="quote-change ${cls}" title="${days} days since first discovery">${arrow} ${sign}${ret.toFixed(2)}%</span>`;
        } catch (err) {
          cell.innerHTML = '<span class="micro">—</span>';
        }
      }
    }
    await Promise.all(Array.from({ length: limit }, () => worker()));
  }
  fillPerf();
</script>
</body>
</html>
"""

STOCK_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ ticker }} · Tradesheet</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT,WONK@0,9..144,300..900,0..100,0..1;1,9..144,300..900,0..100,0..1&family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #fbfaf7;
    --bg-2: #f4f2ec;
    --surface: #ffffff;
    --rule: #d8d3c5;
    --rule-strong: #9c9684;
    --ink: #14110d;
    --ink-mid: #3a342a;
    --ink-mute: #6b6555;
    --ink-faint: #9c9684;
    --saffron: #b85d12;
    --saffron-soft: rgba(184, 93, 18, 0.09);
    --sage: #3f5a1f;
    --sage-soft: rgba(63, 90, 31, 0.09);
    --oxblood: #962e21;
    --oxblood-soft: rgba(150, 46, 33, 0.08);
  }
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--ink); }
  body {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 13px;
    line-height: 1.55;
    font-variant-numeric: tabular-nums slashed-zero;
    background-color: var(--bg);
    background-image:
      radial-gradient(ellipse at top, rgba(184, 93, 18, 0.04), transparent 70%);
    min-height: 100vh;
  }
  .serif { font-family: 'Fraunces', Georgia, serif; }
  .serif-italic { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-variation-settings: 'opsz' 144, 'SOFT' 50, 'WONK' 1; }
  .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums slashed-zero; }
  .label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-mute); font-weight: 500; }
  .micro { font-size: 11px; color: var(--ink-mute); }

  /* Masthead */
  .masthead { border-bottom: 2px solid var(--ink); background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%); position: relative; }
  .masthead::after { content: ''; position: absolute; left: 0; right: 0; bottom: -6px; height: 1px; background: var(--ink); opacity: 0.3; }
  .masthead-inner { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; gap: 32px; }
  .brand { font-family: 'Fraunces', serif; font-style: italic; font-weight: 600; font-size: 18px; color: var(--ink); font-variation-settings: 'opsz' 144, 'SOFT' 100, 'WONK' 1; }
  .brand-dot { color: var(--saffron); }
  .live-dot { display: inline-block; width: 7px; height: 7px; background: var(--saffron); border-radius: 50%; animation: pulse 2s infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(184, 93, 18, 0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(184, 93, 18, 0); }
    100% { box-shadow: 0 0 0 0 rgba(184, 93, 18, 0); }
  }

  .back-link { color: var(--ink-mute); text-decoration: none; letter-spacing: 0.12em; text-transform: uppercase; font-size: 11px; transition: color 0.2s; }
  .back-link:hover { color: var(--saffron); }

  /* Almanac-scale ticker */
  .ticker-mega {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 300;
    font-size: clamp(6rem, 18vw, 13rem);
    line-height: 0.82;
    letter-spacing: -0.06em;
    color: var(--ink);
    font-variation-settings: 'opsz' 144, 'SOFT' 100, 'WONK' 1;
  }
  .price-display { font-family: 'IBM Plex Mono', monospace; font-size: clamp(2.5rem, 5vw, 4rem); font-weight: 300; letter-spacing: -0.04em; line-height: 1; color: var(--ink); }
  .quote-change { font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500; letter-spacing: 0.02em; padding: 3px 10px; border-radius: 0; border: 1px solid; display: inline-block; }
  .pos { color: var(--sage); border-color: var(--sage); background: var(--sage-soft); }
  .neg { color: var(--oxblood); border-color: var(--oxblood); background: var(--oxblood-soft); }
  .neu { color: var(--ink-mute); border-color: var(--rule); }

  .meta-line { color: var(--ink-mid); font-size: 13px; }
  .meta-line span + span::before { content: '·'; margin: 0 8px; color: var(--ink-faint); }

  /* KPI strip */
  .kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--rule); }
  @media (max-width: 720px) { .kpi-strip { grid-template-columns: repeat(2, 1fr); } }
  .kpi-cell { padding: 20px 24px; border-right: 1px solid var(--rule); position: relative; }
  .kpi-cell:nth-child(4) { border-right: 0; }
  @media (max-width: 720px) {
    .kpi-cell:nth-child(2) { border-right: 0; }
    .kpi-cell:nth-child(3) { border-top: 1px solid var(--rule); }
    .kpi-cell:nth-child(4) { border-top: 1px solid var(--rule); }
  }
  .kpi-cell .label { margin-bottom: 8px; }
  .kpi-val { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 400; color: var(--ink); letter-spacing: -0.02em; }

  /* Chart container */
  .chart-wrap { border: 1px solid var(--rule); padding: 28px; position: relative; }
  .chart-wrap canvas { max-height: 360px; }
  .range-toggle { display: inline-flex; gap: 0; border: 1px solid var(--rule); }
  .range-toggle button {
    background: none; border: 0;
    padding: 6px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-mute);
    cursor: pointer;
    transition: all 0.2s;
    border-right: 1px solid var(--rule);
  }
  .range-toggle button:last-child { border-right: 0; }
  .range-toggle button:hover { color: var(--ink); background: var(--saffron-soft); }
  .range-toggle button.on { color: var(--saffron); background: var(--saffron-soft); }

  /* Section heading */
  .section-h {
    font-family: 'Fraunces', serif; font-style: italic; font-weight: 400;
    font-size: clamp(2rem, 4vw, 3rem);
    letter-spacing: -0.03em;
    color: var(--ink);
    font-variation-settings: 'opsz' 144, 'SOFT' 80, 'WONK' 1;
    line-height: 1;
  }
  .section-rule { height: 1px; background: var(--rule); margin: 18px 0 24px; position: relative; }
  .section-rule::before { content: ''; position: absolute; left: 0; top: 0; width: 80px; height: 1px; background: var(--saffron); }

  /* Watchlist module */
  .module { border: 1px solid var(--rule); padding: 28px; }
  .module-good { border-left: 3px solid var(--saffron); }

  .dot-rating { display: inline-flex; gap: 8px; }
  .dot-rating button { background: none; border: 0; padding: 4px; cursor: pointer; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 1px solid var(--rule-strong); transition: all 0.2s; }
  .dot-on { background: var(--saffron); border-color: var(--saffron); box-shadow: 0 0 8px var(--saffron-soft); }
  .dot-rating button:hover .dot { transform: scale(1.3); }
  .remove-btn { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); background: none; border: 0; cursor: pointer; transition: color 0.2s; }
  .remove-btn:hover { color: var(--oxblood); }

  /* Add-to-watchlist form */
  .add-form { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .add-form input, .add-form select {
    background: var(--bg-2);
    border: 1px solid var(--rule);
    padding: 8px 12px;
    color: var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
  }
  .add-form input { flex: 1; min-width: 200px; }
  .add-form input:focus, .add-form select:focus { outline: 0; border-color: var(--saffron); }
  .btn-saffron {
    background: var(--saffron); color: var(--bg);
    border: 1px solid var(--saffron);
    padding: 8px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-saffron:hover { background: transparent; color: var(--saffron); }

  /* Discovery cards */
  .find-card { padding: 20px 0 20px 24px; border-left: 1px solid var(--rule); position: relative; }
  .find-card + .find-card { border-top: 1px dashed var(--rule); margin-top: 0; }
  .find-card::before {
    content: ''; position: absolute; left: -1px; top: 24px; width: 12px; height: 1px; background: var(--rule);
  }
  .find-thesis {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-style: normal; font-weight: 400;
    font-size: 15px; line-height: 1.85; color: var(--ink-mid);
    margin-top: 14px;
    max-width: 78ch;
  }
  .conf-pill { font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 2px 8px; border-radius: 0; border: 1px solid; }
  .conf-high { color: var(--saffron); border-color: var(--saffron); background: var(--saffron-soft); }
  .conf-mid  { color: var(--ink-mid); border-color: var(--rule-strong); }
  .conf-low  { color: var(--oxblood); border-color: var(--oxblood); background: var(--oxblood-soft); }
  .run-pill { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-mute); padding: 2px 8px; border: 1px solid var(--rule); }

  /* Holdings table */
  .htable { width: 100%; border-collapse: collapse; }
  .htable thead th { padding: 12px 14px; text-align: left; font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-mute); border-bottom: 1px solid var(--rule-strong); border-top: 1px solid var(--rule-strong); }
  .htable tbody td { padding: 14px; border-bottom: 1px solid var(--rule); }

  /* Reveal animations */
  @keyframes reveal { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .reveal { animation: reveal 0.7s cubic-bezier(.2,.7,.2,1) both; }
  .reveal-1 { animation-delay: 0.05s; }
  .reveal-2 { animation-delay: 0.15s; }
  .reveal-3 { animation-delay: 0.25s; }
  .reveal-4 { animation-delay: 0.35s; }
  .reveal-5 { animation-delay: 0.45s; }

  ::selection { background: var(--saffron); color: var(--bg); }
</style>
</head>
<body>

<div class="masthead">
  <div class="max-w-[1280px] mx-auto px-8">
    <div class="masthead-inner">
      <div class="flex items-center gap-8">
        <a href="{{ url_for('index') }}" class="brand" style="text-decoration: none;">Tradesheet<span class="brand-dot">.</span></a>
        <a href="{{ url_for('index') }}" class="back-link">← All Watchlists</a>
      </div>
      <div class="flex items-center gap-6 micro">
        <span><span class="live-dot"></span>LIVE</span>
        <span id="market-clock">—</span>
      </div>
    </div>
  </div>
</div>

<main class="max-w-[1280px] mx-auto px-8 pb-24">

  <!-- Hero -->
  <section class="pt-12 pb-10 reveal reveal-1">
    <div class="label mb-3">§ Equity File · {{ quote.sector or '—' }}</div>
    <div class="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-8 items-end">
      <div>
        <h1 class="ticker-mega">{{ ticker }}<span style="color: var(--saffron);">.</span></h1>
      </div>
      {% if quote.price is not none %}
      <div class="text-right">
        <div class="price-display">${{ '%.2f'|format(quote.price) }}</div>
        {% if quote.change_pct is not none %}
          {% set pcls = 'pos' if quote.change_pct >= 0 else 'neg' %}
          <div class="mt-3"><span class="quote-change {{ pcls }}">
            {% if quote.change_pct >= 0 %}▲{% else %}▼{% endif %} {{ '%+.2f'|format(quote.change_pct) }}% today
          </span></div>
        {% endif %}
      </div>
      {% endif %}
    </div>

    <div class="mt-8 flex justify-between items-baseline flex-wrap gap-4">
      <div class="meta-line serif-italic">
        {% if quote.name %}<span>{{ quote.name }}</span>{% endif %}
        {% if quote.industry %}<span>{{ quote.industry }}</span>{% endif %}
      </div>
      <div class="label" style="color: var(--ink-faint);">{{ ticker }} · NYSE/NASDAQ/TSX</div>
    </div>
  </section>

  <!-- KPI strip -->
  <section class="mb-12 reveal reveal-2">
    <div class="kpi-strip">
      <div class="kpi-cell">
        <div class="label">Market Cap</div>
        <div class="kpi-val">{% if quote.market_cap %}${{ '{:,.1f}'.format(quote.market_cap/1e9) }}B{% else %}—{% endif %}</div>
      </div>
      <div class="kpi-cell">
        <div class="label">Trailing P/E</div>
        <div class="kpi-val">{% if quote.pe %}{{ '%.1f'|format(quote.pe) }}x{% else %}—{% endif %}</div>
      </div>
      <div class="kpi-cell">
        <div class="label">52-week High</div>
        <div class="kpi-val">{% if quote.fifty_two_high %}${{ '%.2f'|format(quote.fifty_two_high) }}{% else %}—{% endif %}</div>
      </div>
      <div class="kpi-cell">
        <div class="label">52-week Low</div>
        <div class="kpi-val">{% if quote.fifty_two_low %}${{ '%.2f'|format(quote.fifty_two_low) }}{% else %}—{% endif %}</div>
      </div>
    </div>
  </section>

  <!-- Chart -->
  <section class="mb-16 reveal reveal-3">
    <div class="flex justify-between items-end mb-4 flex-wrap gap-4">
      <div>
        <div class="label">Price Trajectory</div>
        <h2 class="section-h">The Chart<span style="color: var(--saffron);">.</span></h2>
      </div>
      <div class="range-toggle" id="rangeToggle">
        <button data-range="1mo">1M</button>
        <button data-range="6mo">6M</button>
        <button data-range="1y" class="on">1Y</button>
        <button data-range="5y">5Y</button>
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="priceChart"></canvas>
    </div>
  </section>

  <!-- Watchlist module -->
  <section class="mb-16 reveal reveal-4">
    <div class="label mb-3">{% if watch_entry %}On Tradesheet{% else %}Add to Tradesheet{% endif %}</div>
    {% if watch_entry %}
      <div class="module module-good">
        <div class="flex justify-between items-start flex-wrap gap-6">
          <div>
            <div class="label" style="color: var(--saffron);">Added {{ watch_entry.added_at[:10] }}</div>
            {% if perf %}
              <div class="mt-3 flex items-center gap-4 flex-wrap">
                <span class="micro" style="color: var(--ink-mid);">Since added ({{ perf.days }}d):</span>
                {% set pcls = 'pos' if perf.return_pct >= 0 else 'neg' %}
                <span class="quote-change {{ pcls }}">
                  {% if perf.return_pct >= 0 %}▲{% else %}▼{% endif %} {{ '%+.2f'|format(perf.return_pct) }}%
                </span>
                <span class="mono micro">${{ '%.2f'|format(perf.entry_price) }} <span style="color: var(--ink-faint);">→</span> ${{ '%.2f'|format(perf.current_price) }}</span>
              </div>
            {% endif %}
          </div>
          <div class="flex items-center gap-6 flex-wrap">
            <form method="post" action="{{ url_for('set_interest', ticker=ticker) }}" class="dot-rating">
              {% for n in [1,2,3,4,5] %}
                <button type="submit" name="interest" value="{{ n }}" title="set interest {{ n }}">
                  <span class="dot {% if n <= watch_entry.interest_level %}dot-on{% endif %}"></span>
                </button>
              {% endfor %}
            </form>
            <form method="post" action="{{ url_for('remove_watch', ticker=ticker) }}">
              <button class="remove-btn" type="submit">— remove from list</button>
            </form>
          </div>
        </div>
        {% if watch_entry.notes %}
          <div class="mt-5 pt-5" style="border-top: 1px solid var(--rule);">
            <div style="font-family: 'IBM Plex Mono', monospace; font-style: normal; font-size: 15px; line-height: 1.75; color: var(--ink-mid); max-width: 78ch;">"{{ watch_entry.notes }}"</div>
          </div>
        {% endif %}
      </div>
    {% else %}
      <div class="module">
        <form method="post" action="{{ url_for('add_watch', ticker=ticker) }}" class="add-form">
          <input type="text" name="notes" placeholder="optional notes — a sentence on why">
          <select name="interest">
            {% for n in [1,2,3,4,5] %}<option value="{{ n }}" {% if n == 3 %}selected{% endif %}>{{ n }} ★</option>{% endfor %}
          </select>
          <button class="btn-saffron" type="submit">+ Add to Tradesheet</button>
        </form>
      </div>
    {% endif %}
  </section>

  {% if finds %}
  <!-- Discovery history -->
  <section class="mb-16 reveal reveal-5">
    <div class="label mb-3">§ Provenance</div>
    <h2 class="section-h">Discovery History<span style="color: var(--saffron);">.</span></h2>
    <div class="section-rule"></div>
    <div>
      {% for f in finds %}
      <div class="find-card">
        <div class="flex items-center justify-between flex-wrap gap-3 mb-1">
          <div class="flex items-center gap-3 flex-wrap">
            <span class="run-pill">Run #{{ f.analysis_run_id }}</span>
            <span class="label" style="letter-spacing: 0.1em;">{{ f.discovery_source or '—' }}</span>
            <span class="micro mono">{{ f.discovered_at[:10] if f.discovered_at else '' }}</span>
          </div>
          {% if f.confidence_score %}
            {% set cls = 'conf-high' if f.confidence_score >= 7 else ('conf-mid' if f.confidence_score >= 5 else 'conf-low') %}
            <span class="conf-pill {{ cls }}">conf {{ '%.1f'|format(f.confidence_score) }}</span>
          {% endif %}
        </div>
        {% if f.investment_thesis %}<div class="find-thesis">"{{ f.investment_thesis }}"</div>{% endif %}
      </div>
      {% endfor %}
    </div>
  </section>
  {% endif %}

  {% if holdings %}
  <!-- Holdings table -->
  <section class="mb-16 reveal reveal-5">
    <div class="label mb-3">§ Portfolio Tracking</div>
    <h2 class="section-h">Holdings<span style="color: var(--saffron);">.</span></h2>
    <div class="section-rule"></div>
    <div style="overflow-x: auto;">
      <table class="htable mono">
        <thead>
          <tr>
            <th>Entry Date</th>
            <th style="text-align: right;">Entry</th>
            <th style="text-align: right;">Current</th>
            <th style="text-align: right;">Return</th>
            <th style="text-align: right;">Days</th>
            <th style="text-align: right;">Max Gain</th>
            <th style="text-align: right;">Max DD</th>
          </tr>
        </thead>
        <tbody>
          {% for h in holdings %}
          <tr>
            <td>{{ h.entry_date[:10] if h.entry_date else '—' }}</td>
            <td style="text-align: right;">${{ '%.2f'|format(h.entry_price) if h.entry_price else '—' }}</td>
            <td style="text-align: right;">${{ '%.2f'|format(h.current_price) if h.current_price else '—' }}</td>
            <td style="text-align: right;">
              {% if h.total_return_pct is not none %}
                {% set cls = 'pos' if h.total_return_pct >= 0 else 'neg' %}
                <span class="quote-change {{ cls }}">{% if h.total_return_pct >= 0 %}▲{% else %}▼{% endif %} {{ '%+.2f'|format(h.total_return_pct) }}%</span>
              {% endif %}
            </td>
            <td style="text-align: right;">{{ h.holding_days or 0 }}</td>
            <td style="text-align: right; color: var(--sage);">{% if h.max_gain_pct %}{{ '%+.1f'|format(h.max_gain_pct) }}%{% else %}—{% endif %}</td>
            <td style="text-align: right; color: var(--oxblood);">{% if h.max_drawdown_pct %}{{ '%+.1f'|format(h.max_drawdown_pct) }}%{% else %}—{% endif %}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>
  {% endif %}

  <footer class="mt-24 pt-8 border-t border-[#c5b48d] flex justify-between micro">
    <div>Tradesheet · {{ ticker }}</div>
    <div class="label">quote refreshed {{ quote.ticker }}</div>
  </footer>
</main>

<script>
  function updateClock() {
    const now = new Date();
    const opts = { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false };
    document.getElementById('market-clock').textContent = now.toLocaleTimeString('en-US', opts) + ' ET';
  }
  updateClock();
  setInterval(updateClock, 30000);

  let priceChart = null;
  function drawChart(range) {
    fetch('{{ url_for("api_prices", ticker=ticker) }}?range=' + range)
      .then(r => r.json())
      .then(data => {
        if (!data.dates || data.dates.length === 0) return;
        const ctx = document.getElementById('priceChart');
        if (priceChart) priceChart.destroy();
        // Compute a softer min/max for cleaner y-axis
        const minClose = Math.min(...data.closes);
        const maxClose = Math.max(...data.closes);
        const padding = (maxClose - minClose) * 0.1;
        priceChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: data.dates,
            datasets: [{
              label: '{{ ticker }} close',
              data: data.closes,
              borderColor: '#b85d12',
              backgroundColor: (ctx) => {
                const chart = ctx.chart;
                const { ctx: c, chartArea } = chart;
                if (!chartArea) return 'rgba(184, 93, 18, 0.06)';
                const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                gradient.addColorStop(0, 'rgba(184, 93, 18, 0.22)');
                gradient.addColorStop(1, 'rgba(184, 93, 18, 0)');
                return gradient;
              },
              fill: true,
              tension: 0.25,
              pointRadius: 0,
              pointHoverRadius: 4,
              pointHoverBackgroundColor: '#b85d12',
              pointHoverBorderColor: '#f1e8d3',
              borderWidth: 1.75,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: '#1c1610',
                borderColor: '#1c1610',
                borderWidth: 1,
                titleColor: '#f1e8d3',
                bodyColor: '#b85d12',
                titleFont: { family: 'IBM Plex Mono', size: 11, weight: '400' },
                bodyFont: { family: 'IBM Plex Mono', size: 13, weight: '500' },
                padding: 12,
                displayColors: false,
                callbacks: {
                  label: (ctx) => '$' + ctx.parsed.y.toFixed(2),
                }
              }
            },
            scales: {
              x: {
                ticks: {
                  color: '#7a6e51',
                  font: { family: 'IBM Plex Mono', size: 10 },
                  maxTicksLimit: 8,
                  maxRotation: 0,
                },
                grid: { display: false },
                border: { color: '#94835a' },
              },
              y: {
                position: 'right',
                min: minClose - padding,
                max: maxClose + padding,
                ticks: {
                  color: '#7a6e51',
                  font: { family: 'IBM Plex Mono', size: 10 },
                  callback: v => '$' + v.toFixed(0),
                  padding: 8,
                },
                grid: { color: 'rgba(197, 180, 141, 0.5)', drawTicks: false },
                border: { display: false },
              }
            }
          }
        });
      });
  }
  drawChart('1y');

  document.querySelectorAll('#rangeToggle button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#rangeToggle button').forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
      drawChart(btn.dataset.range);
    });
  });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    watchlist_entries = list_watchlist()
    enriched = []
    for w in watchlist_entries:
        quote = _quote(w.ticker)
        perf = _performance_since(w.ticker, w.added_at)
        enriched.append({
            "ticker": w.ticker,
            "added_at": w.added_at,
            "interest_level": w.interest_level,
            "notes": w.notes,
            "quote": quote,
            "perf": perf,
        })
    saved = _saved_stocks(limit=300)
    watchlist_tickers = {w.ticker for w in watchlist_entries}
    return render_template_string(
        INDEX_TEMPLATE,
        watchlist=enriched,
        saved=saved,
        watchlist_tickers=watchlist_tickers,
        db_path=str(DB_PATH),
    )


@app.route("/stock/<ticker>")
def stock_detail(ticker: str):
    ticker = ticker.upper()
    from stockbot.web.watchlist import get_entry
    watch_entry = get_entry(ticker)
    quote = _quote(ticker)
    finds = _finds_by_ticker(ticker)
    holdings = _holdings_by_ticker(ticker)
    perf = _performance_since(ticker, watch_entry.added_at) if watch_entry else None
    return render_template_string(
        STOCK_TEMPLATE,
        ticker=ticker,
        quote=quote,
        finds=finds,
        holdings=holdings,
        watch_entry=watch_entry,
        perf=perf,
    )


@app.route("/watchlist/<ticker>", methods=["POST"])
def add_watch(ticker: str):
    interest = int(request.form.get("interest", 3))
    notes = (request.form.get("notes") or "").strip()
    add_to_watchlist(ticker, interest_level=interest, notes=notes)
    return redirect(request.referrer or url_for("index"))


@app.route("/watchlist/<ticker>/remove", methods=["POST"])
def remove_watch(ticker: str):
    remove_from_watchlist(ticker)
    return redirect(request.referrer or url_for("index"))


@app.route("/watchlist/<ticker>/interest", methods=["POST"])
def set_interest(ticker: str):
    interest = int(request.form.get("interest", 3))
    update_interest(ticker, interest)
    return redirect(request.referrer or url_for("index"))


@app.route("/api/prices/<ticker>")
def api_prices(ticker: str):
    period = request.args.get("range", "1y")
    return jsonify(_price_history(ticker, period=period))


@app.route("/api/snapshot/<ticker>")
def api_snapshot(ticker: str):
    return jsonify(_quote(ticker))


@app.route("/api/perf/<ticker>")
def api_perf(ticker: str):
    """Return performance since a given ISO date (?since=YYYY-MM-DD or full ISO)."""
    since = request.args.get("since")
    if not since:
        return jsonify({"error": "missing 'since' query param"}), 400
    perf = _performance_since(ticker, since)
    if perf is None:
        return jsonify({"ticker": ticker.upper(), "error": "no history available"})
    return jsonify({"ticker": ticker.upper(), **perf})


def main():
    """Run the Flask dev server on localhost:5050."""
    print("Watchlist UI: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
