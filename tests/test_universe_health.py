import json
import time

import pytest

from stockbot.screening import universe
from stockbot.screening.universe import Ticker


MINIMUMS = {"sp500": 450, "sp600": 500, "tsx_composite": 150}


def _source(source, count):
    exchange = "TSX" if source == "tsx_composite" else "US"
    suffix = ".TO" if exchange == "TSX" else ""
    prefix = {"sp500": "L", "sp600": "S", "tsx_composite": "T"}[source]
    return [Ticker(f"{prefix}{index:04d}{suffix}", exchange, source) for index in range(count)]


def _healthy():
    return [ticker for source, count in MINIMUMS.items() for ticker in _source(source, count)]


def test_partial_source_refresh_never_overwrites_last_known_good(tmp_path, monkeypatch):
    cache = tmp_path / "universe.json"
    monkeypatch.setattr(universe, "CACHE_PATH", cache)
    universe._write_cache(_healthy())
    original = cache.read_text()
    monkeypatch.setattr(universe, "_load_sp500", lambda: _source("sp500", 449))
    monkeypatch.setattr(universe, "_load_sp600", lambda: _source("sp600", 500))
    monkeypatch.setattr(universe, "_load_tsx_composite", lambda: _source("tsx_composite", 150))

    result = universe.load_universe(force_refresh=True)

    assert len(result) == 1100
    assert cache.read_text() == original


def test_refresh_failure_raises_without_valid_stale_cache(tmp_path, monkeypatch):
    cache = tmp_path / "universe.json"
    monkeypatch.setattr(universe, "CACHE_PATH", cache)
    monkeypatch.setattr(universe, "_load_sp500", lambda: _source("sp500", 449))
    monkeypatch.setattr(universe, "_load_sp600", lambda: _source("sp600", 500))
    monkeypatch.setattr(universe, "_load_tsx_composite", lambda: _source("tsx_composite", 150))

    with pytest.raises(RuntimeError, match="sp500.*450"):
        universe.load_universe(force_refresh=True)


@pytest.mark.parametrize("source,minimum", MINIMUMS.items())
def test_source_health_minimums_are_enforced(source, minimum):
    tickers = _healthy()
    tickers = [ticker for ticker in tickers if ticker.source != source]
    tickers.extend(_source(source, minimum - 1))

    with pytest.raises(ValueError, match=source):
        universe._validate_universe(tickers)
