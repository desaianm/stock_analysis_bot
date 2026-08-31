import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from stockbot.screening import numeric_screen
from stockbot.screening.numeric_screen import StockSnapshot
from stockbot.screening.universe import Ticker


def _ticker(symbol, exchange=None, source="test"):
    return Ticker(symbol=symbol, exchange=exchange or ("TSX" if symbol.endswith(".TO") else "US"), source=source)


def _clean_snapshot(ticker):
    return StockSnapshot(
        ticker.symbol, ticker.exchange, ticker.source, price=10.0,
        volume=1_000_000.0, market_cap=1_000_000_000.0, trailing_pe=10.0,
        debt_to_equity=1.0, current_ratio=2.0, free_cash_flow=1_000_000.0,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def test_scan_does_not_reuse_fresh_cache_for_different_universe(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    cache.write_text(json.dumps({"generated_at": 1, "count": 1, "snapshots": [StockSnapshot("AAA", "US", "test").to_dict()]}))
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    monkeypatch.setattr(numeric_screen, "fetch_snapshot", lambda ticker: StockSnapshot(ticker.symbol, ticker.exchange, ticker.source))

    result = numeric_screen.scan_universe([_ticker("BBB")], workers=1)

    assert [snapshot.symbol for snapshot in result] == ["BBB"]


def test_scan_reuses_matching_schema_and_exchange_qualified_universe(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    requested = [_ticker("SHOP.TO")]
    cache.write_text(json.dumps({
        "schema_version": numeric_screen.CACHE_SCHEMA_VERSION,
        "ticker_universe": numeric_screen._universe_identity(requested),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": 1,
        "success_count": 1,
        "failure_count": 0,
        "snapshots": [_clean_snapshot(requested[0]).to_dict()],
    }))
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    monkeypatch.setattr(numeric_screen, "fetch_snapshot", lambda ticker: (_ for _ in ()).throw(AssertionError("cache not reused")))

    result = numeric_screen.scan_universe(requested, workers=1)

    assert [snapshot.symbol for snapshot in result] == ["SHOP.TO"]


def test_scan_reuses_cache_when_requested_universe_is_reordered(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    original = [_ticker("AAA"), _ticker("BBB")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    numeric_screen._write_cache(
        [_clean_snapshot(t) for t in original], original
    )
    monkeypatch.setattr(
        numeric_screen, "fetch_snapshot",
        lambda ticker: (_ for _ in ()).throw(AssertionError("cache not reused")),
    )

    result = numeric_screen.scan_universe(list(reversed(original)), workers=1)

    assert [snapshot.symbol for snapshot in result] == ["BBB", "AAA"]


@pytest.mark.parametrize(
    "changed",
    [_ticker("AAA", exchange="TSX"), _ticker("AAA", source="another-source")],
)
def test_scan_rejects_cache_when_exchange_or_source_differs(tmp_path, monkeypatch, changed):
    cache = tmp_path / "numeric.json"
    original = [_ticker("AAA")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    numeric_screen._write_cache([_clean_snapshot(original[0])], original)
    fetched = []
    monkeypatch.setattr(
        numeric_screen, "fetch_snapshot",
        lambda ticker: fetched.append(ticker) or _clean_snapshot(ticker),
    )

    numeric_screen.scan_universe([changed], workers=1)

    assert fetched == [changed]


def test_snapshot_timestamp_is_timezone_aware_utc(monkeypatch):
    class FakeYahooTicker:
        info = {}
        cash_flow = None

    monkeypatch.setattr(numeric_screen.yf, "Ticker", lambda symbol: FakeYahooTicker())

    snapshot = numeric_screen.fetch_snapshot(_ticker("AAA"))

    assert snapshot.fetched_at.endswith("+00:00")


def test_scan_retries_only_failures_and_preserves_requested_order(tmp_path, monkeypatch):
    requested = [_ticker("AAA"), _ticker("BBB"), _ticker("CCC")]
    attempts = {ticker.symbol: 0 for ticker in requested}
    delays = []
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", tmp_path / "numeric.json")

    def fetch(ticker):
        attempts[ticker.symbol] += 1
        if ticker.symbol == "BBB" and attempts[ticker.symbol] == 1:
            return StockSnapshot(ticker.symbol, ticker.exchange, ticker.source, error="429")
        return StockSnapshot(ticker.symbol, ticker.exchange, ticker.source)

    monkeypatch.setattr(numeric_screen, "fetch_snapshot", fetch)
    result = numeric_screen.scan_universe(
        requested, workers=3, retry_workers=1, retry_delays=(0.25,), sleep=delays.append
    )

    assert [snapshot.symbol for snapshot in result] == ["AAA", "BBB", "CCC"]
    assert [(s.exchange, s.source) for s in result] == [(t.exchange, t.source) for t in requested]
    assert [snapshot.error for snapshot in result] == [None, None, None]
    assert attempts == {"AAA": 1, "BBB": 2, "CCC": 1}
    assert delays == [0.25]


def test_cache_records_quality_counts_and_schema(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    requested = [_ticker("AAA"), _ticker("BBB")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    monkeypatch.setattr(
        numeric_screen,
        "fetch_snapshot",
        lambda ticker: _clean_snapshot(ticker),
    )

    numeric_screen.scan_universe(requested, workers=1, retry_delays=(), sleep=lambda _: None)
    payload = json.loads(cache.read_text())

    assert payload["schema_version"] == numeric_screen.CACHE_SCHEMA_VERSION
    assert payload["success_count"] == 2
    assert payload["failure_count"] == 0


def test_failure_dominated_scan_raises_and_writes_no_cache(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    requested = [_ticker(symbol) for symbol in ["AAA", "BBB", "CCC", "DDD"]]
    attempts = []
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    monkeypatch.setattr(
        numeric_screen,
        "fetch_snapshot",
        lambda ticker: attempts.append(ticker.symbol)
        or StockSnapshot(ticker.symbol, ticker.exchange, ticker.source, error="429"),
    )

    with pytest.raises(RuntimeError, match=r"4 failed of 4.*100\.0%.*20\.0%"):
        numeric_screen.scan_universe(
            requested, workers=2, retry_workers=1, retry_delays=(0, 0), sleep=lambda _: None
        )

    assert attempts == ["AAA", "BBB", "CCC", "DDD"] * 3
    assert not cache.exists()


def test_failure_dominated_cache_is_never_served(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    requested = [_ticker(symbol) for symbol in ["AAA", "BBB", "CCC", "DDD", "EEE"]]
    snapshots = [
        StockSnapshot(t.symbol, t.exchange, t.source, error="old failure")
        for t in requested
    ]
    cache.write_text(json.dumps({
        "schema_version": numeric_screen.CACHE_SCHEMA_VERSION,
        "ticker_universe": numeric_screen._universe_identity(requested),
        "generated_at": 1,
        "count": 5,
        "success_count": 0,
        "failure_count": 5,
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
    }))
    fetched = []
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    monkeypatch.setattr(
        numeric_screen,
        "fetch_snapshot",
        lambda ticker: fetched.append(ticker.symbol)
        or StockSnapshot(ticker.symbol, ticker.exchange, ticker.source),
    )

    result = numeric_screen.scan_universe(requested, workers=1, retry_delays=(), sleep=lambda _: None)

    assert fetched == [ticker.symbol for ticker in requested]
    assert all(snapshot.error is None for snapshot in result)


def test_any_failed_scan_is_returned_but_never_cached(tmp_path, monkeypatch):
    requested = [_ticker(symbol) for symbol in ["AAA", "BBB", "CCC", "DDD", "EEE"]]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", tmp_path / "numeric.json")
    monkeypatch.setattr(
        numeric_screen,
        "fetch_snapshot",
        lambda ticker: StockSnapshot(
            ticker.symbol,
            ticker.exchange,
            ticker.source,
            error="still throttled" if ticker.symbol == "EEE" else None,
        ),
    )

    result = numeric_screen.scan_universe(
        requested, workers=1, retry_delays=(), sleep=lambda _: None
    )

    assert result[-1].symbol == "EEE"
    assert result[-1].error == "still throttled"
    assert not numeric_screen.CACHE_PATH.exists()


def test_cache_with_one_failed_snapshot_is_retried_not_served(tmp_path, monkeypatch):
    cache = tmp_path / "numeric.json"
    requested = [_ticker("AAA"), _ticker("BBB")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    snapshots = [_clean_snapshot(requested[0]), _clean_snapshot(requested[1])]
    snapshots[1].error = "old failure"
    now = datetime.now(timezone.utc)
    cache.write_text(json.dumps({
        "schema_version": numeric_screen.CACHE_SCHEMA_VERSION,
        "ticker_universe": numeric_screen._universe_identity(requested),
        "generated_at": now.isoformat(), "count": 2,
        "success_count": 1, "failure_count": 1,
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
    }))
    fetched = []
    monkeypatch.setattr(
        numeric_screen, "fetch_snapshot",
        lambda ticker: fetched.append(ticker.symbol) or _clean_snapshot(ticker),
    )

    result = numeric_screen.scan_universe(requested, workers=1, retry_delays=())

    assert fetched == ["AAA", "BBB"]
    assert all(snapshot.error is None for snapshot in result)


@pytest.mark.parametrize("timestamp_kind", ["stale_payload", "future_payload", "stale_mtime", "future_mtime"])
def test_copied_or_touched_cache_with_invalid_age_is_not_served(
    tmp_path, monkeypatch, timestamp_kind
):
    cache = tmp_path / "numeric.json"
    requested = [_ticker("AAA")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    snapshot = _clean_snapshot(requested[0])
    now = datetime.now(timezone.utc)
    generated = now
    mtime = now.timestamp()
    if timestamp_kind == "stale_payload":
        generated = now - timedelta(seconds=numeric_screen.CACHE_TTL_SECONDS + 10)
    elif timestamp_kind == "future_payload":
        generated = now + timedelta(minutes=5)
    elif timestamp_kind == "stale_mtime":
        mtime -= numeric_screen.CACHE_TTL_SECONDS + 10
    else:
        mtime += 300
    cache.write_text(json.dumps({
        "schema_version": numeric_screen.CACHE_SCHEMA_VERSION,
        "ticker_universe": numeric_screen._universe_identity(requested),
        "generated_at": generated.isoformat(), "count": 1,
        "success_count": 1, "failure_count": 0,
        "snapshots": [snapshot.to_dict()],
    }))
    os.utime(cache, (mtime, mtime))
    fetched = []
    monkeypatch.setattr(
        numeric_screen, "fetch_snapshot",
        lambda ticker: fetched.append(ticker.symbol) or _clean_snapshot(ticker),
    )

    numeric_screen.scan_universe(requested, workers=1, retry_delays=())

    assert fetched == ["AAA"]


@pytest.mark.parametrize("fetched_at", [None, "2026-08-31T12:00:00", "not-a-date"])
def test_cache_requires_timezone_aware_snapshot_timestamps(
    tmp_path, monkeypatch, fetched_at
):
    cache = tmp_path / "numeric.json"
    requested = [_ticker("AAA")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    snapshot = _clean_snapshot(requested[0])
    snapshot.fetched_at = fetched_at
    now = datetime.now(timezone.utc)
    cache.write_text(json.dumps({
        "schema_version": numeric_screen.CACHE_SCHEMA_VERSION,
        "ticker_universe": numeric_screen._universe_identity(requested),
        "generated_at": now.isoformat(), "count": 1,
        "success_count": 1, "failure_count": 0,
        "snapshots": [snapshot.to_dict()],
    }))
    monkeypatch.setattr(numeric_screen, "fetch_snapshot", lambda ticker: _clean_snapshot(ticker))

    numeric_screen.scan_universe(requested, workers=1, retry_delays=())

    assert json.loads(cache.read_text())["snapshots"][0]["fetched_at"] != fetched_at
