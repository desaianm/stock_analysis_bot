"""Parallel numeric screening over the universe.

Per ticker, fetches yfinance `info` (one HTTP call) and extracts the metrics
needed for the funnel's deterministic stages. Runs 10 concurrent threads to
keep the full ~1,300-ticker scan under 4 minutes.

Results cache to ``state/numeric_screen_cache.json`` for 24h. Reuse requires
the same cache schema and exact requested ticker universe.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yfinance as yf

from stockbot.screening.universe import Ticker, load_universe

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "numeric_screen_cache.json"
CACHE_TTL_SECONDS = 24 * 3600
CACHE_SCHEMA_VERSION = 2


@dataclass
class StockSnapshot:
    """Numeric metrics needed for funnel stages 2-4."""
    symbol: str
    exchange: str
    source: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    price: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    enterprise_value: Optional[float] = None
    revenue_growth: Optional[float] = None  # most recent yoy
    earnings_growth: Optional[float] = None
    gross_margins: Optional[float] = None
    return_on_equity: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    target_mean_price: Optional[float] = None
    historical_fcf_growth: Optional[float] = None  # for DCF margin-of-safety
    fetched_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "StockSnapshot":
        return cls(**d)

    @property
    def fcf_yield(self) -> Optional[float]:
        if self.free_cash_flow and self.market_cap and self.market_cap > 0:
            return self.free_cash_flow / self.market_cap
        return None

    @property
    def price_vs_52w_high(self) -> Optional[float]:
        if self.price and self.fifty_two_week_high and self.fifty_two_week_high > 0:
            return self.price / self.fifty_two_week_high
        return None

    @property
    def analyst_upside(self) -> Optional[float]:
        if self.target_mean_price and self.price and self.price > 0:
            return (self.target_mean_price - self.price) / self.price
        return None


def _historical_fcf_growth(stock: yf.Ticker) -> Optional[float]:
    """CAGR of free cash flow across available annual cash-flow statements."""
    try:
        cf = stock.cash_flow
        if cf is None or getattr(cf, "empty", True):
            return None
        # yfinance returns columns left-to-right newest-to-oldest
        row_key = None
        for name in ["Free Cash Flow"]:
            for idx in cf.index:
                if str(idx).strip() == name:
                    row_key = idx
                    break
            if row_key is not None:
                break
        if row_key is None:
            return None
        values = [float(v) for v in cf.loc[row_key].tolist() if v == v]  # drop NaN
        # We want oldest -> newest order
        values = list(reversed(values))
        if len(values) < 2 or values[0] <= 0:
            return None
        years = len(values) - 1
        return (values[-1] / values[0]) ** (1 / years) - 1
    except Exception:
        return None


def fetch_snapshot(t: Ticker) -> StockSnapshot:
    """Fetch one ticker's numeric snapshot from yfinance."""
    try:
        stock = yf.Ticker(t.symbol)
        info = stock.info or {}

        d2e = info.get("debtToEquity")
        if d2e is not None:
            # Yahoo defines debtToEquity in percentage points (e.g. 125 = 1.25x).
            d2e = d2e / 100

        snap = StockSnapshot(
            symbol=t.symbol,
            exchange=t.exchange,
            source=t.source,
            sector=info.get("sector"),
            industry=info.get("industry"),
            price=info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"),
            market_cap=info.get("marketCap"),
            volume=info.get("averageVolume") or info.get("volume"),
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            price_to_book=info.get("priceToBook"),
            debt_to_equity=d2e,
            current_ratio=info.get("currentRatio"),
            free_cash_flow=info.get("freeCashflow"),
            enterprise_value=info.get("enterpriseValue"),
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            gross_margins=info.get("grossMargins"),
            return_on_equity=info.get("returnOnEquity"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            target_mean_price=info.get("targetMeanPrice"),
            historical_fcf_growth=_historical_fcf_growth(stock),
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        return snap
    except Exception as exc:  # noqa: BLE001
        return StockSnapshot(
            symbol=t.symbol,
            exchange=t.exchange,
            source=t.source,
            error=str(exc),
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    return (time.time() - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS


def _universe_identity(tickers: Iterable[Ticker]) -> List[Dict[str, str]]:
    """Return a stable identity for the complete requested ticker universe."""
    return [
        {"symbol": ticker.symbol, "exchange": ticker.exchange, "source": ticker.source}
        for ticker in sorted(tickers, key=lambda item: (item.symbol, item.exchange, item.source))
    ]


def _read_cache(tickers: Iterable[Ticker]) -> List[StockSnapshot]:
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("numeric cache schema mismatch")
    if payload.get("ticker_universe") != _universe_identity(tickers):
        raise ValueError("numeric cache universe mismatch")
    return [StockSnapshot.from_dict(d) for d in payload["snapshots"]]


def _write_cache(snaps: List[StockSnapshot], tickers: Iterable[Ticker]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "ticker_universe": _universe_identity(tickers),
        "generated_at": time.time(),
        "count": len(snaps),
        "snapshots": [s.to_dict() for s in snaps],
    }
    contents = json.dumps(payload, indent=2, default=str)
    fd, temporary_name = tempfile.mkstemp(dir=CACHE_PATH.parent, prefix=f".{CACHE_PATH.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, CACHE_PATH)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def scan_universe(
    tickers: Optional[Iterable[Ticker]] = None,
    workers: int = 10,
    force_refresh: bool = False,
    progress_every: int = 100,
) -> List[StockSnapshot]:
    """Parallel-fetch metrics for the full universe. Caches 24h."""
    if tickers is None:
        tickers = load_universe()
    tickers = list(tickers)
    if not force_refresh and _cache_is_fresh():
        try:
            return _read_cache(tickers)
        except Exception:
            pass

    print(f"Scanning {len(tickers)} tickers with {workers} workers...")
    snaps: List[StockSnapshot] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_snapshot, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            snaps.append(future.result())
            if i % progress_every == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(tickers) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(tickers)} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    ok = [s for s in snaps if not s.error]
    failed = [s for s in snaps if s.error]
    print(f"Scan complete: {len(ok)} ok, {len(failed)} failed, total {time.time() - start:.0f}s")

    _write_cache(snaps, tickers)
    return snaps


@dataclass
class ScreeningGates:
    """Hard numeric filters applied after the scan."""
    max_price: float = 500.0
    min_price: float = 3.0
    min_volume: float = 200_000
    max_pe: Optional[float] = 30.0
    min_market_cap: float = 100_000_000
    max_debt_equity: Optional[float] = 3.0
    min_current_ratio: Optional[float] = 1.0
    max_decline_from_high: Optional[float] = None
    require_positive_fcf: bool = True


def apply_gates(snaps: Iterable[StockSnapshot], gates: ScreeningGates) -> List[StockSnapshot]:
    """Filter snapshots through the numeric gates; required missing values fail closed."""
    out: List[StockSnapshot] = []
    for s in snaps:
        if s.error:
            continue
        if s.price is None or s.price < gates.min_price or s.price > gates.max_price:
            continue
        if s.volume is None or s.volume < gates.min_volume:
            continue
        if s.market_cap is None or s.market_cap < gates.min_market_cap:
            continue
        pe = s.trailing_pe if s.trailing_pe is not None else s.forward_pe
        if gates.max_pe is not None:
            if pe is None or pe <= 0 or pe > gates.max_pe:
                continue
        if gates.max_debt_equity is not None:
            if s.debt_to_equity is None or s.debt_to_equity > gates.max_debt_equity:
                continue
        if gates.min_current_ratio is not None:
            if s.current_ratio is None or s.current_ratio < gates.min_current_ratio:
                continue
        if gates.max_decline_from_high is not None:
            if (
                s.price_vs_52w_high is None
                or 1 - s.price_vs_52w_high > gates.max_decline_from_high
            ):
                continue
        if gates.require_positive_fcf:
            if s.free_cash_flow is None or s.free_cash_flow <= 0:
                continue
        out.append(s)
    return out


if __name__ == "__main__":
    from stockbot.screening.universe import load_universe
    # Smoke: scan 20 tickers
    universe = load_universe()[:20]
    print(f"Smoke scan of {len(universe)} tickers")
    snaps = scan_universe(universe, workers=10, force_refresh=True)
    print(f"\nSurvivors after gates:")
    survivors = apply_gates(snaps, ScreeningGates())
    for s in survivors[:10]:
        print(f"  {s.symbol:8s} {s.sector or '?':20s} price=${s.price:.2f} pe={s.trailing_pe or s.forward_pe:.1f} fcf_yield={s.fcf_yield:.2%}" if s.fcf_yield else f"  {s.symbol:8s} price=${s.price}")
