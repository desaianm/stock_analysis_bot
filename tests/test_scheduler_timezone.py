from datetime import date, datetime
from zoneinfo import ZoneInfo

from stockbot.scheduler.daily_tracker import DailyPortfolioScheduler


def test_discord_loop_uses_dst_aware_new_york_zoneinfo():
    scheduler = DailyPortfolioScheduler.__new__(DailyPortfolioScheduler)

    loop = scheduler.create_task()
    scheduled_time = loop.time[0]

    assert isinstance(scheduled_time.tzinfo, ZoneInfo)
    assert scheduled_time.tzinfo.key == "America/New_York"
    winter = datetime.combine(date(2026, 1, 15), scheduled_time)
    summer = datetime.combine(date(2026, 7, 15), scheduled_time)
    assert winter.utcoffset().total_seconds() == -5 * 3600
    assert summer.utcoffset().total_seconds() == -4 * 3600
