import asyncio
from types import SimpleNamespace

from stockbot.flows.performance_tracker import PerformanceTrackerFlow


def test_snapshots_use_reloaded_prices_and_exclude_failed_tickers(monkeypatch):
    flow = PerformanceTrackerFlow.__new__(PerformanceTrackerFlow)
    stale = [
        {"id": 1, "ticker": "AAA", "current_price": 10, "total_return_pct": 0, "holding_days": 1},
        {"id": 2, "ticker": "BAD", "current_price": 20, "total_return_pct": 0, "holding_days": 1},
    ]
    fresh = [
        {"id": 1, "ticker": "AAA", "current_price": 12, "total_return_pct": 20, "holding_days": 2},
        {"id": 2, "ticker": "BAD", "current_price": 20, "total_return_pct": 0, "holding_days": 2},
    ]
    flow.db = SimpleNamespace(get_active_holdings=lambda: fresh)
    flow.performance_agent = SimpleNamespace(arun=lambda prompt: _async_value("ok"))
    flow._verify_prices = lambda tickers: {"ok": ["AAA"], "failed": {"BAD": "no quote"}}
    flow._get_pending_catalysts_summary = lambda: "none"
    captured = []

    async def save(rows):
        captured.extend(rows)

    flow._save_performance_snapshots = save
    calls = iter([stale, fresh])
    flow.db.get_active_holdings = lambda: next(calls)

    asyncio.run(flow.update_all_holdings())

    assert [(row["ticker"], row["current_price"], row["total_return_pct"]) for row in captured] == [
        ("AAA", 12, 20)
    ]


async def _async_value(value):
    return value
