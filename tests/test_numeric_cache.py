import json

import pytest

from stockbot.screening import numeric_screen
from stockbot.screening.numeric_screen import StockSnapshot
from stockbot.screening.universe import Ticker


def _ticker(symbol, exchange=None, source="test"):
    return Ticker(symbol=symbol, exchange=exchange or ("TSX" if symbol.endswith(".TO") else "US"), source=source)


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
        "generated_at": 1,
        "count": 1,
        "snapshots": [StockSnapshot("SHOP.TO", "TSX", "test").to_dict()],
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
        [StockSnapshot(t.symbol, t.exchange, t.source) for t in original], original
    )
    monkeypatch.setattr(
        numeric_screen, "fetch_snapshot",
        lambda ticker: (_ for _ in ()).throw(AssertionError("cache not reused")),
    )

    result = numeric_screen.scan_universe(list(reversed(original)), workers=1)

    assert {snapshot.symbol for snapshot in result} == {"AAA", "BBB"}


@pytest.mark.parametrize(
    "changed",
    [_ticker("AAA", exchange="TSX"), _ticker("AAA", source="another-source")],
)
def test_scan_rejects_cache_when_exchange_or_source_differs(tmp_path, monkeypatch, changed):
    cache = tmp_path / "numeric.json"
    original = [_ticker("AAA")]
    monkeypatch.setattr(numeric_screen, "CACHE_PATH", cache)
    numeric_screen._write_cache([StockSnapshot("AAA", "US", "test")], original)
    fetched = []
    monkeypatch.setattr(
        numeric_screen, "fetch_snapshot",
        lambda ticker: fetched.append(ticker) or StockSnapshot(ticker.symbol, ticker.exchange, ticker.source),
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
