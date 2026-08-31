import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from stockbot.scheduler.daily_tracker import DailyPortfolioScheduler


@pytest.mark.parametrize(("date", "weekly_calls"), [(datetime(2026, 8, 30), 1), (datetime(2026, 8, 31), 0)])
def test_daily_updates_run_daily_but_learning_runs_only_sunday(date, weekly_calls):
    calls = {"daily": 0, "weekly": 0}

    async def daily():
        calls["daily"] += 1

    async def weekly():
        calls["weekly"] += 1

    scheduler = DailyPortfolioScheduler.__new__(DailyPortfolioScheduler)
    scheduler.tracker = SimpleNamespace(update_all_holdings=daily, generate_weekly_learning_insights=weekly)
    scheduler._now = lambda: date.replace(tzinfo=ZoneInfo("America/New_York"))

    asyncio.run(scheduler.daily_portfolio_update())

    assert calls == {"daily": 1, "weekly": weekly_calls}
