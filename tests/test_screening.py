import math

import pandas as pd
import pytest

from stockbot.screening import numeric_screen
from stockbot.screening.numeric_screen import (
    ScreeningGates,
    StockSnapshot,
    apply_gates,
    fetch_snapshot,
)
from stockbot.screening.universe import Ticker


def _snapshot(**overrides):
    values = dict(symbol="AAA", exchange="US", source="test", price=50, volume=1_000_000,
                  market_cap=1_000_000_000, trailing_pe=10, forward_pe=12,
                  debt_to_equity=1, current_ratio=2, free_cash_flow=1_000_000,
                  fifty_two_week_high=80)
    values.update(overrides)
    return StockSnapshot(**values)


def test_configured_balance_sheet_and_high_decline_gates_fail_closed():
    gates = ScreeningGates(max_debt_equity=3, min_current_ratio=1, max_decline_from_high=0.4)
    snapshots = [
        _snapshot(symbol="NO_DEBT", debt_to_equity=None),
        _snapshot(symbol="NO_RATIO", current_ratio=None),
        _snapshot(symbol="TOO_LOW", price=40, fifty_two_week_high=80),
        _snapshot(symbol="PASS", price=50, fifty_two_week_high=80),
    ]

    assert [s.symbol for s in apply_gates(snapshots, gates)] == ["PASS"]


def test_zero_trailing_pe_is_not_replaced_by_forward_pe():
    assert apply_gates([_snapshot(trailing_pe=0, forward_pe=10)], ScreeningGates()) == []


def test_yfinance_debt_to_equity_is_converted_from_percent(monkeypatch):
    class FakeYFinanceTicker:
        info = {"debtToEquity": 8.0}
        cash_flow = None

    monkeypatch.setattr("stockbot.screening.numeric_screen.yf.Ticker", lambda symbol: FakeYFinanceTicker())

    snapshot = fetch_snapshot(Ticker("AAA", "US", "test"))

    assert snapshot.debt_to_equity == 0.08


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("currentPrice", math.nan),
        ("currentPrice", math.inf),
        ("averageVolume", math.nan),
        ("averageVolume", -math.inf),
        ("marketCap", math.nan),
        ("marketCap", math.inf),
        ("trailingPE", math.nan),
        ("trailingPE", math.inf),
        ("forwardPE", math.nan),
        ("forwardPE", -math.inf),
        ("debtToEquity", math.nan),
        ("debtToEquity", math.inf),
        ("currentRatio", math.nan),
        ("currentRatio", math.inf),
        ("freeCashflow", math.nan),
        ("freeCashflow", -math.inf),
    ],
)
def test_fetch_snapshot_rejects_nonfinite_required_yahoo_metrics(
    monkeypatch, field, value
):
    valid = {
        "currentPrice": 50.0,
        "averageVolume": 1_000_000.0,
        "marketCap": 1_000_000_000.0,
        "trailingPE": 10.0,
        "forwardPE": 12.0,
        "debtToEquity": 100.0,
        "currentRatio": 2.0,
        "freeCashflow": 10_000_000.0,
    }
    if field == "forwardPE":
        valid["trailingPE"] = None
    valid[field] = value

    class FakeYahooTicker:
        info = valid
        cash_flow = None

    monkeypatch.setattr(numeric_screen.yf, "Ticker", lambda symbol: FakeYahooTicker())

    snapshot = fetch_snapshot(Ticker("AAA", "US", "test"))

    assert snapshot.error


@pytest.mark.parametrize("info", [{}, {"currentPrice": 10.0}])
def test_fetch_snapshot_rejects_empty_or_incomplete_info(monkeypatch, info):
    class FakeYahooTicker:
        cash_flow = None

    FakeYahooTicker.info = info
    monkeypatch.setattr(numeric_screen.yf, "Ticker", lambda symbol: FakeYahooTicker())

    assert fetch_snapshot(Ticker("AAA", "US", "test")).error


@pytest.mark.parametrize(
    "values",
    [
        [-100.0, 121.0],
        [100.0, -121.0],
        [100.0, 0.0],
        [100.0, -50.0, 121.0],
        [math.inf, 121.0],
        [100.0, math.nan, 121.0],
        [100.0, math.nan, math.inf],
    ],
)
def test_historical_fcf_growth_rejects_invalid_endpoints(values):
    class FakeStock:
        # Yahoo presents newest first; the helper reverses this row.
        cash_flow = pd.DataFrame(
            [list(reversed(values))], index=["Free Cash Flow"]
        )

    assert numeric_screen._historical_fcf_growth(FakeStock()) is None


def test_historical_fcf_growth_returns_real_finite_cagr():
    class FakeStock:
        cash_flow = pd.DataFrame(
            [[121.0, 110.0, 100.0]], index=["Free Cash Flow"]
        )

    assert numeric_screen._historical_fcf_growth(FakeStock()) == pytest.approx(0.1)
