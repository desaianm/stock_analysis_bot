"""Portfolio performance tracking and learning flow using Agno framework."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pytz
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from stockbot.database.performance_manager import PerformanceTrackingManager
from stockbot.tools.performance_tools import PerformanceTools

ny_timezone = pytz.timezone("America/New_York")


def load_prompt(prompt_name: str) -> str:
    """Load prompt template from prompts/performance_tracker/ directory."""
    prompt_path = (
        Path(__file__).parent.parent.parent
        / "prompts"
        / "performance_tracker"
        / f"{prompt_name}.txt"
    )
    return prompt_path.read_text(encoding="utf-8")


class PerformanceTrackerFlow:
    """Agno-based flow for portfolio performance tracking and learning."""

    def __init__(self, db_path: str = "stock_analysis.db"):
        """Initialize performance tracker with database and tools.

        Args:
            db_path: Path to database file (default: stock_analysis.db)
        """
        self.db = PerformanceTrackingManager(db_path)
        self.tools_instance = PerformanceTools(db_path)

        # Initialize Agno agent (following undervalued.py pattern)
        self.performance_agent = Agent(
            name="Portfolio Performance Analyst",
            model=OpenAIChat(
                id="gpt-5.4-nano-2026-03-17",
                temperature=1,
                max_completion_tokens=10000,
            ),
            tools=[
                self.update_prices,
                self.calculate_metrics,
                self.validate_catalysts,
                self.generate_insights,
                self.get_summary,
            ],
            instructions=load_prompt("performance_agent_instructions")
            .strip()
            .split("\n"),
            markdown=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

    # -------------------------------------------------------------------------
    # Tool Wrappers (return JSON strings for Agno agent)
    # -------------------------------------------------------------------------

    def update_prices(self, tickers: list) -> str:
        """Update portfolio prices for given tickers."""
        return self.tools_instance.update_portfolio_prices(tickers)

    def calculate_metrics(self, ticker: str, entry_date: str) -> str:
        """Calculate performance metrics for a ticker."""
        return self.tools_instance.calculate_performance_metrics(ticker, entry_date)

    def validate_catalysts(self, ticker: str, catalysts: list) -> str:
        """Validate catalyst realization for a ticker."""
        return self.tools_instance.validate_catalyst_realization(ticker, catalysts)

    def generate_insights(self, time_period: int = 90) -> str:
        """Generate learning insights from performance data."""
        return self.tools_instance.generate_learning_insights(time_period)

    def get_summary(self) -> str:
        """Get portfolio summary statistics."""
        return self.tools_instance.get_portfolio_summary()

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    async def initialize_holdings_from_run(self, run_id: int):
        """
        Create portfolio holdings for all stocks from an analysis run.

        Args:
            run_id: The analysis run ID to create holdings from
        """
        stock_finds = self.db.get_stock_finds_by_run_id(run_id)

        if not stock_finds:
            print(f"  No stock finds found for run #{run_id}")
            return

        holdings_created = 0
        catalysts_tracked = 0

        for find in stock_finds:
            try:
                # Create portfolio holding
                holding_id = self.db.create_portfolio_holding(
                    stock_find_id=find["id"],
                    ticker=find["ticker"],
                    entry_price=find["current_price"] or 0.0,
                    entry_date=find["discovered_at"],
                )
                holdings_created += 1

                # Create catalyst tracking for each predicted catalyst
                if find.get("catalysts"):
                    catalysts = (
                        find["catalysts"]
                        if isinstance(find["catalysts"], list)
                        else json.loads(find["catalysts"])
                    )

                    for catalyst in catalysts:
                        if isinstance(catalyst, str) and catalyst.strip():
                            self.db.create_catalyst_tracking(
                                stock_find_id=find["id"],
                                ticker=find["ticker"],
                                catalyst=catalyst,
                                catalyst_type=self._infer_catalyst_type(catalyst),
                                confidence_at_prediction=find.get("confidence_score"),
                            )
                            catalysts_tracked += 1

            except Exception as e:
                print(f"  Error creating holding for {find['ticker']}: {e}")
                continue

        print(
            f"  Initialized {holdings_created} portfolio holdings and {catalysts_tracked} catalyst trackers from run #{run_id}"
        )

    async def update_all_holdings(self):
        """Daily price update and analysis for all active holdings."""
        holdings = self.db.get_active_holdings()
        tickers = [h["ticker"] for h in holdings]

        if not tickers:
            print("No active holdings to update")
            return

        print(f"\n[Performance Tracker] Updating {len(tickers)} active holdings...")

        # Build daily review prompt
        prompt = load_prompt("daily_review_prompt").format(
            current_date=datetime.now(ny_timezone).isoformat(),
            active_holdings_count=len(holdings),
            ticker_list=", ".join(tickers),
            pending_catalysts=self._get_pending_catalysts_summary(),
        )

        # Run agent
        try:
            result = await self.performance_agent.arun(prompt)
            print(f"  Daily update completed: {len(tickers)} holdings processed")

            # Extract and save performance snapshots
            await self._save_performance_snapshots(holdings)

        except Exception as e:
            print(f"  Error during daily update: {e}")

    async def generate_weekly_learning_insights(self):
        """Weekly learning synthesis (runs every Sunday)."""
        holdings = self.db.get_active_holdings()

        if len(holdings) < 5:
            print("  Insufficient data for learning insights (minimum 5 holdings required)")
            return

        print("\n[Performance Tracker] Generating weekly learning insights...")

        # Build learning synthesis prompt
        prompt = load_prompt("learning_synthesis_prompt").format(
            start_date=(datetime.now(ny_timezone) - timedelta(days=90)).isoformat(),
            end_date=datetime.now(ny_timezone).isoformat(),
            total_holdings=len(holdings),
        )

        # Run agent
        try:
            result = await self.performance_agent.arun(prompt)

            # Parse insights from agent output and save to database
            await self._extract_and_save_insights(result)

            print("  Weekly learning insights generated successfully")

        except Exception as e:
            print(f"  Error during learning synthesis: {e}")

    async def generate_performance_report(self) -> str:
        """
        Generate on-demand performance summary for Discord command.

        Returns:
            Markdown-formatted performance report
        """
        try:
            summary_json = self.get_summary()
            summary = json.loads(summary_json)

            # Build report
            report = f"""# Portfolio Performance Summary

**Active Holdings**: {summary['active_holdings']}
**Average Return**: {summary['avg_return']}%
**Win Rate**: {summary['win_rate']}%
**Average Holding Period**: {summary['avg_holding_days']} days

## Best Performer
{summary['best_performer']['ticker']}: {summary['best_performer']['return']:+.2f}%

## Worst Performer
{summary['worst_performer']['ticker']}: {summary['worst_performer']['return']:+.2f}%

## Recent Learning Insights
{self._format_recent_insights()}

---
*Last updated: {datetime.now(ny_timezone).strftime('%Y-%m-%d %H:%M ET')}*
"""
            return report

        except Exception as e:
            return f"Error generating performance report: {e}"

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _infer_catalyst_type(self, catalyst_text: str) -> str:
        """Infer catalyst type from text using keyword matching."""
        lower = catalyst_text.lower()

        if any(kw in lower for kw in ["earnings", "quarterly", "annual", "eps"]):
            return "earnings"
        elif any(kw in lower for kw in ["product", "launch", "release", "unveil"]):
            return "product_launch"
        elif any(
            kw in lower for kw in ["regulatory", "approval", "fda", "compliance"]
        ):
            return "regulatory"
        elif any(kw in lower for kw in ["merger", "acquisition", "buyout", "deal"]):
            return "merger"
        elif any(kw in lower for kw in ["management", "ceo", "cfo", "appointment"]):
            return "management_change"
        else:
            return "other"

    def _get_pending_catalysts_summary(self) -> str:
        """Get summary of pending catalysts for daily review prompt."""
        pending = self.db.get_pending_catalysts(limit=10)

        if not pending:
            return "None pending validation"

        summary_items = []
        for catalyst in pending[:5]:  # Top 5
            summary_items.append(f"{catalyst['ticker']}: {catalyst['predicted_catalyst']}")

        more_count = len(pending) - 5
        if more_count > 0:
            summary_items.append(f"...and {more_count} more")

        return ", ".join(summary_items)

    def _format_recent_insights(self) -> str:
        """Format recent learning insights for report display."""
        insights_list = self.db.get_recent_learning_insights(days=30, min_confidence="medium")

        if not insights_list:
            return "No recent insights available (minimum 5 holdings required)"

        formatted = []
        for insight in insights_list[:3]:  # Top 3 most recent
            formatted.append(f"- **{insight['insight_type'].replace('_', ' ').title()}**: {insight['insight_summary']}")

        return "\n".join(formatted)

    async def _save_performance_snapshots(self, holdings: List[dict]):
        """Save performance snapshots after price updates."""
        for holding in holdings:
            try:
                if holding["current_price"] and holding["current_price"] > 0:
                    self.db.save_performance_snapshot(
                        holding_id=holding["id"],
                        ticker=holding["ticker"],
                        price=holding["current_price"],
                        return_pct=holding["total_return_pct"] or 0.0,
                        days_held=holding["holding_days"] or 0,
                    )
            except Exception as e:
                print(f"  Error saving snapshot for {holding['ticker']}: {e}")

    async def _extract_and_save_insights(self, agent_output: str):
        """
        Extract learning insights from agent output and save to database.

        This is a simple implementation that calls generate_insights tool
        which already saves to the database.
        """
        try:
            # Generate insights using the tool (which saves to DB)
            insights_json = self.generate_insights(time_period=90)
            insights_data = json.loads(insights_json)

            if "insights" in insights_data and insights_data["insights"]:
                # Save each insight to database
                for insight in insights_data["insights"]:
                    self.db.save_learning_insight(
                        insight_type=insight["type"],
                        metric_name=insight.get("type", "general"),
                        metric_value=0.0,  # Placeholder
                        insight_summary=insight["summary"],
                        actionable_recommendation=insight["action"],
                        sample_size=insight.get("sample_size", 0),
                        time_period_days=90,
                        confidence_level=insight.get("confidence_level", "medium"),
                    )

                print(f"  Saved {len(insights_data['insights'])} insights to database")

        except Exception as e:
            print(f"  Error extracting insights: {e}")
