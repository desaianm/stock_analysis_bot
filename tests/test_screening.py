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
