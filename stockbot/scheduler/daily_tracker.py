"""Daily portfolio tracking using Discord task loops."""

import pytz
from datetime import datetime, time
from discord.ext import tasks

from stockbot.flows.performance_tracker import PerformanceTrackerFlow

ny_timezone = pytz.timezone("America/New_York")


class DailyPortfolioScheduler:
    """Schedules daily portfolio performance updates using Discord task loops."""

    def __init__(self):
        """Initialize scheduler with NY timezone."""
        self.tracker = PerformanceTrackerFlow()
        self._task_started = False

    async def daily_portfolio_update(self):
        """
        Run daily portfolio update at 5 PM ET (after market close).

        Updates all active holdings with current prices, validates catalysts,
        and generates learning insights daily.
        """
        print(f"\n[{datetime.now(ny_timezone)}] Starting daily portfolio update...")

        try:
            # Update all holdings with current prices
            await self.tracker.update_all_holdings()

            # Generate learning insights daily (after price updates)
            print("  Generating daily learning insights...")
            await self.tracker.generate_weekly_learning_insights()

            print(f"[{datetime.now(ny_timezone)}] Daily portfolio update completed")

        except Exception as e:
            print(f"[{datetime.now(ny_timezone)}] Error during daily update: {e}")

    def create_task(self):
        """Create and return the Discord task loop.

        Returns:
            discord.ext.tasks.Loop: The task loop object
        """
        # Create task that runs daily at 5 PM ET
        @tasks.loop(time=time(hour=17, minute=0, tzinfo=ny_timezone))
        async def _daily_update_task():
            await self.daily_portfolio_update()

        return _daily_update_task

    def start(self, task_loop):
        """Start the daily portfolio task loop.

        Args:
            task_loop: The Discord task loop to start
        """
        if not self._task_started:
            task_loop.start()
            self._task_started = True
            print("Daily portfolio scheduler started (runs at 5 PM ET)")

    def stop(self, task_loop):
        """Stop the task loop gracefully.

        Args:
            task_loop: The Discord task loop to stop
        """
        if self._task_started and task_loop.is_running():
            task_loop.cancel()
            self._task_started = False
            print("Daily portfolio scheduler stopped")
