"""Stock universe loader for the quant funnel.

Sources:
    - S&P 500       Wikipedia table (~500 large-cap US)
    - S&P 600       Wikipedia table (~600 small-cap US)
    - TSX Composite Wikipedia table (~230 Canadian)

Cached as JSON in ``state/universe_cache.json`` for 24 hours. The cache is
rebuilt automatically when stale; call ``rebuild_universe()`` to force.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import requests

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "universe_cache.json"
CACHE_TTL_SECONDS = 24 * 3600
SOURCE_MINIMUM_COUNTS = {"sp500": 450, "sp600": 500, "tsx_composite": 150}

_HEADERS = {"User-Agent": "stockbot-universe-loader/1.0"}


@dataclass(frozen=True)
class Ticker:
    """One row of the universe."""
    symbol: str          # yfinance-resolvable (e.g. "AAPL", "SHOP.TO")
    exchange: str        # "US" or "TSX"
    source: str          # "sp500", "sp600", "tsx_composite"


# ---------------------------------------------------------------------------
# Wikipedia table parsing — regex over the table cells to stay dep-free.
# ---------------------------------------------------------------------------
def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _extract_first_column_links(html: str, table_idx: int = 0) -> List[str]:
    """Return text of the first cell of each tbody row of the Nth table.

    Wikipedia's constituent tables always put the ticker in column 1.
    """
    # Find each table block
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.DOTALL)
    if table_idx >= len(tables):
        return []
    table = tables[table_idx]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.DOTALL)
    out: List[str] = []
    for row in rows[1:]:  # skip header
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        if not cells:
            continue
        first = _strip_html(cells[0])
        if first:
            out.append(first)
    return out


def _fetch(url: str) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def _load_sp500() -> List[Ticker]:
    html = _fetch("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    symbols = _extract_first_column_links(html, table_idx=0)
    # Some tickers contain "." (e.g. "BRK.B") which yfinance wants as "BRK-B"
    return [
        Ticker(symbol=s.replace(".", "-"), exchange="US", source="sp500")
        for s in symbols
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,5}", s)
    ]


def _load_sp600() -> List[Ticker]:
    html = _fetch("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
    symbols = _extract_first_column_links(html, table_idx=0)
    return [
        Ticker(symbol=s.replace(".", "-"), exchange="US", source="sp600")
        for s in symbols
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,5}", s)
    ]


def _load_tsx_composite() -> List[Ticker]:
    html = _fetch("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index")
    # The constituents table on this page isn't always table index 0; scan all.
    candidates: List[str] = []
    for idx in range(5):
        symbols = _extract_first_column_links(html, table_idx=idx)
        # Constituent table has ~230 entries; pick whichever table has the most short-uppercase entries.
        cleaned = [
            s for s in symbols
            if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,5}", s)
        ]
        if len(cleaned) > len(candidates):
            candidates = cleaned
    # TSX yfinance suffix is ".TO"
    return [
        Ticker(symbol=f"{s}.TO" if "." not in s else s, exchange="TSX", source="tsx_composite")
        for s in candidates
    ]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    age = time.time() - CACHE_PATH.stat().st_mtime
    return age < CACHE_TTL_SECONDS


def _read_cache() -> List[Ticker]:
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    tickers = [Ticker(**t) for t in payload["tickers"]]
    if payload.get("count") != len(tickers):
        raise ValueError("universe cache count mismatch")
    _validate_universe(tickers)
    return tickers


def _validate_universe(tickers: List[Ticker]) -> None:
    counts = universe_breakdown(tickers)
    for source, minimum in SOURCE_MINIMUM_COUNTS.items():
        actual = counts.get(source, 0)
        if actual < minimum:
            raise ValueError(
                f"universe source {source} has {actual} constituents; minimum is {minimum}"
            )


def _write_cache(tickers: List[Ticker]) -> None:
    _validate_universe(tickers)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "count": len(tickers),
        "tickers": [t.__dict__ for t in tickers],
    }
    contents = json.dumps(payload, indent=2)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=CACHE_PATH.parent, prefix=f".{CACHE_PATH.name}."
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, CACHE_PATH)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def rebuild_universe() -> List[Ticker]:
    """Fetch all three sources, dedupe, write to cache."""
    sources = [_load_sp500, _load_sp600, _load_tsx_composite]
    all_tickers: List[Ticker] = []
    seen: set[str] = set()
    for loader in sources:
        loaded = loader()
        expected_source = loaded[0].source if loaded else loader.__name__.removeprefix("_load_")
        minimum = SOURCE_MINIMUM_COUNTS.get(expected_source)
        if minimum is None or len(loaded) < minimum:
            raise RuntimeError(
                f"universe source {expected_source} returned {len(loaded)}; minimum is {minimum}"
            )
        for t in loaded:
            if t.symbol in seen:
                continue
            seen.add(t.symbol)
            all_tickers.append(t)
    _validate_universe(all_tickers)
    _write_cache(all_tickers)
    return all_tickers


def load_universe(force_refresh: bool = False) -> List[Ticker]:
    """Return the cached universe, refreshing if older than 24 hours."""
    if not force_refresh and _cache_is_fresh():
        try:
            return _read_cache()
        except Exception:
            pass
    try:
        return rebuild_universe()
    except Exception:
        if CACHE_PATH.exists():
            try:
                return _read_cache()
            except Exception:
                pass
        raise


def universe_breakdown(tickers: List[Ticker]) -> Dict[str, int]:
    """Count tickers by source — useful for logs."""
    out: Dict[str, int] = {}
    for t in tickers:
        out[t.source] = out.get(t.source, 0) + 1
    out["_total"] = len(tickers)
    return out


if __name__ == "__main__":
    print("Building universe...")
    tickers = rebuild_universe()
    print(f"Loaded {len(tickers)} tickers")
    print(f"Breakdown: {universe_breakdown(tickers)}")
    print(f"Sample: {[t.symbol for t in tickers[:5]]} ... {[t.symbol for t in tickers[-5:]]}")
