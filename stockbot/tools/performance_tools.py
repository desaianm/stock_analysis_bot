"""Performance tracking tools for portfolio monitoring and learning."""

import json
from datetime import datetime, timedelta
from typing import List, Optional

import pytz
import yfinance as yf

from stockbot.database.performance_manager import PerformanceTrackingManager

ny_timezone = pytz.timezone("America/New_York")


class PerformanceTools:
    """Tools for tracking portfolio performance and generating learning insights."""

    def __init__(self, db_path: str = "stock_analysis.db"):
        """Initialize performance tools with database connection.

        Args:
            db_path: Path to database file (default: stock_analysis.db)
        """
        self.db = PerformanceTrackingManager(db_path)

    def update_portfolio_prices(self, tickers: List[str]) -> str:
        """
        Fetch current prices for multiple tickers and update database.

        Args:
            tickers: List of ticker symbols to update

        Returns:
            JSON string with updated prices and statistics
        """
        if not tickers:
            return json.dumps(
                {"error": "No tickers provided", "updated_count": 0}, indent=2
            )

        results = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                current_price = info.get("currentPrice") or info.get(
                    "regularMarketPrice"
                )
                if not current_price:
                    results.append({"ticker": ticker, "error": "Price not available"})
                    continue

                prev_close = info.get("previousClose")
                change_pct = (
                    ((current_price - prev_close) / prev_close * 100)
                    if prev_close
                    else 0
                )

                results.append(
                    {
                        "ticker": ticker,
                        "price": round(current_price, 2),
                        "change_pct": round(change_pct, 2),
                        "volume": info.get("volume"),
                        "market_cap": info.get("marketCap"),
                    }
                )

                # Update database
                update_result = self.db.update_holding_price(ticker, current_price)
                if update_result:
                    results[-1]["db_updated"] = True
                    results[-1]["return_pct"] = update_result["return_pct"]

            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})

        successful_updates = len([r for r in results if "price" in r])

        return json.dumps(
            {
                "updated_count": successful_updates,
                "prices": results,
                "timestamp": datetime.now(ny_timezone).isoformat(),
            },
            indent=2,
        )

    def calculate_performance_metrics(self, ticker: str, entry_date: str) -> str:
        """
        Calculate comprehensive performance metrics for a holding.

        Args:
            ticker: Stock ticker symbol
            entry_date: Entry date in ISO format

        Returns:
            JSON string with performance metrics
        """
        try:
            holding = self.db.get_holding_by_ticker(ticker)
            if not holding:
                return json.dumps({"error": f"No holding found for {ticker}"}, indent=2)

            entry_price = holding["entry_price"]
            current_price = holding["current_price"]
            holding_days = holding["holding_days"]

            # Calculate returns
            total_return = ((current_price - entry_price) / entry_price) * 100
            annualized_return = (
                (total_return / holding_days) * 365 if holding_days > 0 else 0
            )

            # Fetch price history for volatility calculation
            stock = yf.Ticker(ticker)
            hist = stock.history(start=entry_date, interval="1d")

            if len(hist) > 1:
                daily_returns = hist["Close"].pct_change().dropna()
                volatility = daily_returns.std() * (252**0.5) * 100  # Annualized

                max_price = hist["Close"].max()
                min_price = hist["Close"].min()
                max_gain = ((max_price - entry_price) / entry_price) * 100
                max_drawdown = ((min_price - entry_price) / entry_price) * 100
            else:
                volatility = 0
                max_gain = holding["max_gain_pct"] or 0
                max_drawdown = holding["max_drawdown_pct"] or 0

            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0

            return json.dumps(
                {
                    "ticker": ticker,
                    "entry_price": round(entry_price, 2),
                    "current_price": round(current_price, 2),
                    "total_return": round(total_return, 2),
                    "annualized_return": round(annualized_return, 2),
                    "holding_days": holding_days,
                    "volatility": round(volatility, 2),
                    "max_gain": round(max_gain, 2),
                    "max_drawdown": round(max_drawdown, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"error": str(e), "ticker": ticker}, indent=2)

    def validate_catalyst_realization(
        self, ticker: str, predicted_catalysts: List[str]
    ) -> str:
        """
        Check if predicted catalysts have materialized using news search.

        Args:
            ticker: Stock ticker symbol
            predicted_catalysts: List of predicted catalyst descriptions

        Returns:
            JSON string with catalyst validation results
        """
        if not predicted_catalysts:
            return json.dumps(
                {"ticker": ticker, "catalysts_checked": 0, "details": []}, indent=2
            )

        try:
            # Import here to avoid circular dependency
            from stockbot.tools.data import TavilySearchTool

            search_tool = TavilySearchTool()
            results = []

            for catalyst in predicted_catalysts:
                # Search for news about this catalyst
                query = f"{ticker} {catalyst}"
                try:
                    news_results = search_tool._run(query)

                    # Simple heuristic: if recent news mentions catalyst, likely realized
                    realized = len(news_results) > 0
                    evidence_count = len(news_results)

                    results.append(
                        {
                            "catalyst": catalyst,
                            "status": "realized" if realized else "pending",
                            "evidence_count": evidence_count,
                            "recent_news": (
                                [
                                    {"title": n.get("title"), "url": n.get("url")}
                                    for n in news_results[:2]
                                ]
                                if news_results
                                else []
                            ),
                        }
                    )

                except Exception as e:
                    results.append(
                        {"catalyst": catalyst, "status": "error", "error": str(e)}
                    )

            realized_count = len([r for r in results if r.get("status") == "realized"])
            pending_count = len([r for r in results if r.get("status") == "pending"])

            return json.dumps(
                {
                    "ticker": ticker,
                    "catalysts_checked": len(predicted_catalysts),
                    "realized": realized_count,
                    "pending": pending_count,
                    "details": results,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {"error": str(e), "ticker": ticker, "catalysts_checked": 0}, indent=2
            )

    def generate_learning_insights(self, time_period: int = 90) -> str:
        """
        Generate learning insights from performance data over time period.

        Args:
            time_period: Lookback period in days (default 90)

        Returns:
            JSON string with actionable insights
        """
        try:
            insights = []

            # 1. Confidence calibration
            calibration = self.db.get_confidence_calibration_stats(time_period)
            if calibration and calibration["total_analyzed"] >= 5:
                high_conf = calibration["high_conf_avg_return"] or 0
                med_conf = calibration["med_conf_avg_return"] or 0

                gap = high_conf - med_conf

                if gap < 3:  # High and medium performing similarly
                    action = "Review confidence scoring - high confidence picks not outperforming medium"
                elif gap > 10:  # High confidence significantly better
                    action = "Confidence scoring is well-calibrated - maintain current methodology"
                else:
                    action = "Confidence scoring showing moderate differentiation"

                insights.append(
                    {
                        "type": "confidence_calibration",
                        "summary": f"High-confidence picks average {round(high_conf, 1)}% return vs {round(med_conf, 1)}% for medium confidence",
                        "action": action,
                        "sample_size": calibration["total_analyzed"],
                        "confidence_level": (
                            "high"
                            if calibration["total_analyzed"] >= 20
                            else (
                                "medium"
                                if calibration["total_analyzed"] >= 10
                                else "low"
                            )
                        ),
                    }
                )

            # 2. Source reliability
            source_stats = self.db.get_source_performance_stats(time_period)
            if source_stats and len(source_stats) >= 2:
                best_source = max(source_stats, key=lambda x: x["avg_return"] or 0)

                insights.append(
                    {
                        "type": "source_reliability",
                        "summary": f"{best_source['source']} performs best ({round(best_source['avg_return'], 1)}% avg return, {round(best_source['win_rate'], 1)}% win rate)",
                        "action": f"Prioritize {best_source['source']} discoveries in future screening",
                        "sample_size": best_source["pick_count"],
                        "confidence_level": (
                            "high"
                            if best_source["pick_count"] >= 15
                            else ("medium" if best_source["pick_count"] >= 8 else "low")
                        ),
                    }
                )

            # 3. Sector timing
            sector_stats = self.db.get_sector_performance_stats(time_period)
            if sector_stats:
                top_sector = sector_stats[0]

                tsx_better = (
                    (top_sector["tsx_avg_return"] or 0)
                    > (top_sector["us_avg_return"] or 0)
                )

                insights.append(
                    {
                        "type": "sector_timing",
                        "summary": f"Top sector: {top_sector['sector']} ({round(top_sector['avg_return'], 1)}% avg return)",
                        "action": f"{'TSX' if tsx_better else 'US'} stocks in {top_sector['sector']} performing better - adjust geographic focus",
                        "sample_size": top_sector["pick_count"],
                        "confidence_level": (
                            "high"
                            if top_sector["pick_count"] >= 10
                            else ("medium" if top_sector["pick_count"] >= 5 else "low")
                        ),
                    }
                )

            # 4. Catalyst accuracy
            catalyst_stats = self.db.get_catalyst_stats(time_period)
            if catalyst_stats:
                # Find best catalyst type
                best_type = None
                best_rate = 0
                for cat_type, stats in catalyst_stats.items():
                    if stats["total"] >= 3 and stats["realization_rate"] > best_rate:
                        best_type = cat_type
                        best_rate = stats["realization_rate"]

                if best_type:
                    insights.append(
                        {
                            "type": "catalyst_accuracy",
                            "summary": f"{best_type} catalysts have {round(best_rate * 100, 1)}% realization rate",
                            "action": f"Weight {best_type} catalysts higher in investment theses",
                            "sample_size": catalyst_stats[best_type]["total"],
                            "confidence_level": (
                                "high"
                                if catalyst_stats[best_type]["total"] >= 10
                                else (
                                    "medium"
                                    if catalyst_stats[best_type]["total"] >= 5
                                    else "low"
                                )
                            ),
                        }
                    )

            return json.dumps(
                {
                    "insights": insights,
                    "time_period_days": time_period,
                    "total_insights": len(insights),
                    "generated_at": datetime.now(ny_timezone).isoformat(),
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"error": str(e), "insights": []}, indent=2)

    def get_portfolio_summary(self) -> str:
        """
        Get high-level portfolio summary statistics.

        Returns:
            JSON string with portfolio overview
        """
        try:
            stats = self.db.get_portfolio_stats()

            return json.dumps(
                {
                    "active_holdings": stats["active_count"] or 0,
                    "total_value": round(stats["total_value"] or 0, 2),
                    "avg_return": round(stats["avg_return"] or 0, 2),
                    "best_performer": stats["best_performer"],
                    "worst_performer": stats["worst_performer"],
                    "win_rate": round(stats["win_rate"] or 0, 2),
                    "avg_holding_days": int(stats["avg_holding_days"] or 0),
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": str(e),
                    "active_holdings": 0,
                    "avg_return": 0,
                    "win_rate": 0,
                },
                indent=2,
            )
