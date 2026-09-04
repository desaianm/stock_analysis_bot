"""Parallel numeric screening over the universe.

Per ticker, fetches yfinance `info` (one HTTP call) and extracts the metrics
needed for the funnel's deterministic stages. The initial pass uses five
threads, followed by two bounded low-concurrency retry passes for failures.

Results cache to ``state/numeric_screen_cache.json`` for 24h. Reuse requires
the same cache schema, exact requested ticker universe, and an acceptable
recorded failure fraction.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import yfinance as yf

from stockbot.screening.universe import Ticker, load_universe

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "numeric_screen_cache.json"
CACHE_TTL_SECONDS = 24 * 3600
CACHE_SCHEMA_VERSION = 4
MAX_FAILURE_FRACTION = 0.20
CACHE_CLOCK_SKEW_SECONDS = 1.0
DEFAULT_WORKERS = 5
DEFAULT_RETRY_WORKERS = 2
DEFAULT_RETRY_DELAYS = (2.0, 5.0)


@dataclass
class StockSnapshot:
    """Numeric metrics needed for funnel stages 2-4."""
    symbol: str
    exchange: str
    source: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    company_name: Optional[str] = None
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


REQUIRED_NUMERIC_FIELDS = (
    "price", "volume", "market_cap", "debt_to_equity", "current_ratio",
    "free_cash_flow",
)


def _finite_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _snapshot_completeness_error(snapshot: StockSnapshot) -> Optional[str]:
    missing = [
        field_name for field_name in REQUIRED_NUMERIC_FIELDS
        if not _finite_real(getattr(snapshot, field_name))
    ]
    effective_pe = snapshot.trailing_pe if snapshot.trailing_pe is not None else snapshot.forward_pe
    if not _finite_real(effective_pe):
        missing.append("effective_pe")
    if missing:
        return "missing or non-finite required metrics: " + ", ".join(missing)
    return None


def _aware_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


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
        values = [float(value) for value in cf.loc[row_key].tolist()]
        # We want oldest -> newest order
        values = list(reversed(values))
        if (
            len(values) < 2
            or not all(math.isfinite(value) for value in values)
            or any(value <= 0 for value in values)
        ):
            return None
        years = len(values) - 1
        growth = (values[-1] / values[0]) ** (1 / years) - 1
        if isinstance(growth, complex) or not math.isfinite(growth):
            return None
        return growth
    except Exception:
        return None


def fetch_snapshot(t: Ticker) -> StockSnapshot:
    """Fetch one ticker's numeric snapshot from yfinance."""
    try:
        stock = yf.Ticker(t.symbol)
        info = stock.info or {}

        if not info:
            raise ValueError("Yahoo info is empty")
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
            company_name=info.get("longName") or info.get("shortName"),
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
        snap.error = _snapshot_completeness_error(snap)
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
    now = time.time()
    mtime_age = now - CACHE_PATH.stat().st_mtime
    if mtime_age < -CACHE_CLOCK_SKEW_SECONDS or mtime_age >= CACHE_TTL_SECONDS:
        return False
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        generated_at = _aware_datetime(payload.get("generated_at"), "generated_at")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    payload_age = now - generated_at.timestamp()
    return -CACHE_CLOCK_SKEW_SECONDS <= payload_age < CACHE_TTL_SECONDS


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
    snapshots = [StockSnapshot.from_dict(d) for d in payload["snapshots"]]
    generated_at = _aware_datetime(payload.get("generated_at"), "generated_at")
    now = datetime.now(timezone.utc)
    if generated_at > now or now - generated_at >= timedelta(seconds=CACHE_TTL_SECONDS):
        raise ValueError("numeric cache generated_at is future or stale")
    if any(snapshot.error or _snapshot_completeness_error(snapshot) for snapshot in snapshots):
        raise ValueError("numeric cache contains failed or incomplete snapshots")
    for snapshot in snapshots:
        fetched_at = _aware_datetime(snapshot.fetched_at, "fetched_at")
        if fetched_at > now or now - fetched_at >= timedelta(seconds=CACHE_TTL_SECONDS):
            raise ValueError("numeric cache snapshot fetched_at is future or stale")
    success_count = sum(snapshot.error is None for snapshot in snapshots)
    failure_count = len(snapshots) - success_count
    if payload.get("count") != len(snapshots):
        raise ValueError("numeric cache count mismatch")
    if payload.get("success_count") != success_count:
        raise ValueError("numeric cache success count mismatch")
    if payload.get("failure_count") != failure_count:
        raise ValueError("numeric cache failure count mismatch")
    failure_fraction = failure_count / len(snapshots) if snapshots else 0.0
    if failure_fraction > MAX_FAILURE_FRACTION:
        raise ValueError("numeric cache failure fraction exceeds maximum")
    snapshot_by_identity = {
        (snapshot.symbol, snapshot.exchange, snapshot.source): snapshot
        for snapshot in snapshots
    }
    try:
        return [
            snapshot_by_identity[(ticker.symbol, ticker.exchange, ticker.source)]
            for ticker in tickers
        ]
    except KeyError as exc:
        raise ValueError("numeric cache snapshot identity mismatch") from exc


def _write_cache(snaps: List[StockSnapshot], tickers: Iterable[Ticker]) -> None:
    success_count = sum(snapshot.error is None for snapshot in snaps)
    failure_count = len(snaps) - success_count
    if failure_count or any(_snapshot_completeness_error(snapshot) for snapshot in snaps):
        raise ValueError("refusing to write failed or incomplete numeric cache")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "ticker_universe": _universe_identity(tickers),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(snaps),
        "success_count": success_count,
        "failure_count": failure_count,
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
    workers: int = DEFAULT_WORKERS,
    force_refresh: bool = False,
    progress_every: int = 100,
    retry_workers: int = DEFAULT_RETRY_WORKERS,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> List[StockSnapshot]:
    """Fetch metrics with bounded retries and reject low-quality results."""
    if tickers is None:
        tickers = load_universe()
    tickers = list(tickers)
    if not force_refresh and _cache_is_fresh():
        try:
            return _read_cache(tickers)
        except Exception:
            pass

    print(f"Scanning {len(tickers)} tickers with {workers} workers...")
    snaps: List[Optional[StockSnapshot]] = [None] * len(tickers)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_snapshot, ticker): index for index, ticker in enumerate(tickers)}
        for i, future in enumerate(as_completed(futures), 1):
            snaps[futures[future]] = future.result()
            if i % progress_every == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(tickers) - i) / rate if rate > 0 else 0
                print(f"  {i}/{len(tickers)} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    for retry_number, delay in enumerate(retry_delays, 1):
        failed_indexes = [
            index for index, snapshot in enumerate(snaps)
            if snapshot is not None and snapshot.error
        ]
        if not failed_indexes:
            break
        print(
            f"Retry {retry_number}/{len(retry_delays)}: "
            f"{len(failed_indexes)} failed tickers with {retry_workers} workers"
        )
        sleep(delay)
        with ThreadPoolExecutor(max_workers=retry_workers) as pool:
            futures = {
                pool.submit(fetch_snapshot, tickers[index]): index
                for index in failed_indexes
            }
            for future in as_completed(futures):
                snaps[futures[future]] = future.result()

    complete_snaps = [snapshot for snapshot in snaps if snapshot is not None]
    if len(complete_snaps) != len(tickers):
        raise RuntimeError("numeric scan did not produce a snapshot for every requested ticker")
    ok = [snapshot for snapshot in complete_snaps if not snapshot.error]
    failed = [snapshot for snapshot in complete_snaps if snapshot.error]
    print(f"Scan complete: {len(ok)} ok, {len(failed)} failed, total {time.time() - start:.0f}s")

    failure_fraction = len(failed) / len(complete_snaps) if complete_snaps else 0.0
    if failure_fraction > MAX_FAILURE_FRACTION:
        raise RuntimeError(
            "numeric scan quality below threshold: "
            f"{len(failed)} failed of {len(complete_snaps)} "
            f"({failure_fraction:.1%} > {MAX_FAILURE_FRACTION:.1%})"
        )

    if not failed and all(not _snapshot_completeness_error(snapshot) for snapshot in complete_snaps):
        _write_cache(complete_snaps, tickers)
    return complete_snaps


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
