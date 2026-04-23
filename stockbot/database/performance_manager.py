"""Performance tracking database manager extending StockDatabaseManager."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz

from stockbot.database.manager import StockDatabaseManager

ny_timezone = pytz.timezone("America/New_York")


class PerformanceTrackingManager(StockDatabaseManager):
    """Extends StockDatabaseManager with performance tracking capabilities."""

    # -------------------------------------------------------------------------
    # Portfolio Holdings
    # -------------------------------------------------------------------------

    def create_portfolio_holding(
        self,
        stock_find_id: int,
        ticker: str,
        entry_price: float,
        entry_date: str,
    ) -> int:
        """Create a new portfolio holding from a stock find."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_holdings (
                stock_find_id, ticker, entry_price, entry_date,
                current_price, last_price_update, holding_days,
                position_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stock_find_id,
                ticker.upper(),
                entry_price,
                entry_date,
                entry_price,  # Initial current_price = entry_price
                datetime.now(ny_timezone).isoformat(),
                0,  # Initial holding_days = 0
                "active",
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_holding_price(
        self, ticker: str, current_price: float
    ) -> Optional[Dict]:
        """Update current price and calculate returns for a holding."""
        cursor = self.conn.cursor()

        # Get holding details
        cursor.execute(
            """
            SELECT id, entry_price, entry_date, max_gain_pct, max_drawdown_pct
            FROM portfolio_holdings
            WHERE ticker = ? AND position_status = 'active'
            """,
            (ticker.upper(),),
        )
        row = cursor.fetchone()

        if not row:
            return None

        holding = dict(row)
        entry_price = holding["entry_price"]
        entry_date = datetime.fromisoformat(holding["entry_date"])

        # Calculate metrics
        total_return_pct = ((current_price - entry_price) / entry_price) * 100
        holding_days = (datetime.now(ny_timezone) - entry_date).days

        # Update max gain/drawdown
        max_gain = holding["max_gain_pct"] or 0
        max_drawdown = holding["max_drawdown_pct"] or 0

        if total_return_pct > max_gain:
            max_gain = total_return_pct
        if total_return_pct < max_drawdown:
            max_drawdown = total_return_pct

        # Update holding
        cursor.execute(
            """
            UPDATE portfolio_holdings
            SET current_price = ?,
                last_price_update = ?,
                holding_days = ?,
                total_return_pct = ?,
                max_gain_pct = ?,
                max_drawdown_pct = ?
            WHERE id = ?
            """,
            (
                current_price,
                datetime.now(ny_timezone).isoformat(),
                holding_days,
                total_return_pct,
                max_gain,
                max_drawdown,
                holding["id"],
            ),
        )
        self.conn.commit()

        return {
            "holding_id": holding["id"],
            "ticker": ticker,
            "updated": True,
            "return_pct": round(total_return_pct, 2),
        }

    def get_active_holdings(self) -> List[Dict]:
        """Get all active portfolio holdings."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM portfolio_holdings
            WHERE position_status = 'active'
            ORDER BY entry_date DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_holding_by_ticker(self, ticker: str) -> Optional[Dict]:
        """Get active holding by ticker."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM portfolio_holdings
            WHERE ticker = ? AND position_status = 'active'
            """,
            (ticker.upper(),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # -------------------------------------------------------------------------
    # Performance Tracking
    # -------------------------------------------------------------------------

    def save_performance_snapshot(
        self,
        holding_id: int,
        ticker: str,
        price: float,
        return_pct: float,
        days_held: int,
        volatility_30d: Optional[float] = None,
        catalyst_events: Optional[List[str]] = None,
        news_sentiment: Optional[str] = None,
    ) -> int:
        """Save a performance tracking snapshot."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO performance_tracking (
                holding_id, ticker, price_at_check, return_pct,
                days_held, check_date, volatility_30d,
                catalyst_events, news_sentiment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                holding_id,
                ticker.upper(),
                price,
                return_pct,
                days_held,
                datetime.now(ny_timezone).isoformat(),
                volatility_30d,
                json.dumps(catalyst_events) if catalyst_events else None,
                news_sentiment,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_performance_history(
        self, ticker: str, days: int = 90
    ) -> List[Dict]:
        """Get performance history for a ticker."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT * FROM performance_tracking
            WHERE ticker = ? AND check_date >= ?
            ORDER BY check_date DESC
            """,
            (ticker.upper(), cutoff_date),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("catalyst_events"):
                result["catalyst_events"] = json.loads(result["catalyst_events"])
            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # Catalyst Tracking
    # -------------------------------------------------------------------------

    def create_catalyst_tracking(
        self,
        stock_find_id: int,
        ticker: str,
        catalyst: str,
        catalyst_type: str,
        confidence_at_prediction: Optional[float] = None,
    ) -> int:
        """Create a catalyst tracking entry."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO catalyst_tracking (
                stock_find_id, ticker, predicted_catalyst,
                catalyst_type, confidence_at_prediction
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                stock_find_id,
                ticker.upper(),
                catalyst,
                catalyst_type,
                confidence_at_prediction,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_catalyst_realization(
        self,
        catalyst_id: int,
        status: str,
        impact_on_price: Optional[float] = None,
        validation_notes: Optional[str] = None,
    ):
        """Update catalyst realization status."""
        cursor = self.conn.cursor()

        update_fields = {
            "realization_status": status,
            "validated_at": datetime.now(ny_timezone).isoformat(),
        }

        if status == "realized":
            update_fields["realization_date"] = datetime.now(ny_timezone).isoformat()

        if impact_on_price is not None:
            update_fields["impact_on_price"] = impact_on_price

        if validation_notes:
            update_fields["validation_notes"] = validation_notes

        set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [catalyst_id]

        cursor.execute(
            f"""
            UPDATE catalyst_tracking
            SET {set_clause}
            WHERE id = ?
            """,
            values,
        )
        self.conn.commit()

    def get_pending_catalysts(self, limit: int = 50) -> List[Dict]:
        """Get pending catalyst validations."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM catalyst_tracking
            WHERE realization_status = 'pending'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_catalyst_stats(self, days: int = 90) -> Dict:
        """Get catalyst realization statistics."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT
                catalyst_type,
                COUNT(*) as total,
                SUM(CASE WHEN realization_status = 'realized' THEN 1 ELSE 0 END) as realized_count,
                AVG(CASE WHEN realization_status = 'realized' THEN impact_on_price END) as avg_impact
            FROM catalyst_tracking
            WHERE created_at >= ?
            GROUP BY catalyst_type
            """,
            (cutoff_date,),
        )

        stats = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            catalyst_type = row_dict["catalyst_type"]
            stats[catalyst_type] = {
                "total": row_dict["total"],
                "realized": row_dict["realized_count"],
                "realization_rate": (
                    row_dict["realized_count"] / row_dict["total"]
                    if row_dict["total"] > 0
                    else 0
                ),
                "avg_impact": row_dict["avg_impact"],
            }

        return stats

    # -------------------------------------------------------------------------
    # Learning Insights
    # -------------------------------------------------------------------------

    def save_learning_insight(
        self,
        insight_type: str,
        metric_name: str,
        metric_value: float,
        insight_summary: str,
        actionable_recommendation: str,
        insight_category: Optional[str] = None,
        sample_size: Optional[int] = None,
        time_period_days: Optional[int] = None,
        confidence_level: str = "medium",
    ) -> int:
        """Save a learning insight."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO learning_insights (
                insight_type, insight_category, metric_name, metric_value,
                sample_size, time_period_days, insight_summary,
                actionable_recommendation, confidence_level, generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                insight_type,
                insight_category,
                metric_name,
                metric_value,
                sample_size,
                time_period_days,
                insight_summary,
                actionable_recommendation,
                confidence_level,
                datetime.now(ny_timezone).isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_recent_learning_insights(
        self, days: int = 90, min_confidence: str = "medium"
    ) -> List[Dict]:
        """Get recent learning insights."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        confidence_order = {"high": 3, "medium": 2, "low": 1}
        min_conf_value = confidence_order.get(min_confidence, 2)

        cursor.execute(
            """
            SELECT * FROM learning_insights
            WHERE generated_at >= ?
            AND (
                CASE confidence_level
                    WHEN 'high' THEN 3
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 1
                    ELSE 0
                END
            ) >= ?
            ORDER BY generated_at DESC
            """,
            (cutoff_date, min_conf_value),
        )

        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Learning Analytics
    # -------------------------------------------------------------------------

    def get_confidence_calibration_stats(self, days: int = 90) -> Optional[Dict]:
        """Calculate confidence calibration statistics."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT
                AVG(CASE WHEN sf.confidence_score >= 8 THEN ph.total_return_pct END) as high_conf_avg_return,
                AVG(CASE WHEN sf.confidence_score >= 6 AND sf.confidence_score < 8 THEN ph.total_return_pct END) as med_conf_avg_return,
                AVG(CASE WHEN sf.confidence_score < 6 THEN ph.total_return_pct END) as low_conf_avg_return,
                COUNT(*) as total_analyzed
            FROM stock_finds sf
            JOIN portfolio_holdings ph ON sf.id = ph.stock_find_id
            WHERE ph.entry_date >= ? AND ph.holding_days >= 30
            """,
            (cutoff_date,),
        )

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_source_performance_stats(self, days: int = 90) -> List[Dict]:
        """Get performance statistics by discovery source."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT
                sf.discovery_source as source,
                COUNT(*) as pick_count,
                AVG(ph.total_return_pct) as avg_return,
                AVG(ph.holding_days) as avg_hold_time,
                SUM(CASE WHEN ph.total_return_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                MAX(ph.total_return_pct) as best_return,
                MIN(ph.total_return_pct) as worst_return
            FROM stock_finds sf
            JOIN portfolio_holdings ph ON sf.id = ph.stock_find_id
            WHERE ph.entry_date >= ? AND ph.holding_days >= 30
            GROUP BY sf.discovery_source
            ORDER BY avg_return DESC
            """,
            (cutoff_date,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_sector_performance_stats(self, days: int = 90) -> List[Dict]:
        """Get performance statistics by sector."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT
                sf.sector,
                COUNT(*) as pick_count,
                AVG(ph.total_return_pct) as avg_return,
                AVG(CASE WHEN sf.exchange = 'TSX' THEN ph.total_return_pct END) as tsx_avg_return,
                AVG(CASE WHEN sf.exchange != 'TSX' THEN ph.total_return_pct END) as us_avg_return
            FROM stock_finds sf
            JOIN portfolio_holdings ph ON sf.id = ph.stock_find_id
            WHERE ph.entry_date >= ? AND ph.holding_days >= 30 AND sf.sector IS NOT NULL
            GROUP BY sf.sector
            ORDER BY avg_return DESC
            """,
            (cutoff_date,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_portfolio_stats(self) -> Dict:
        """Get overall portfolio statistics."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) as active_count,
                SUM(current_price * shares_virtual) as total_value,
                AVG(total_return_pct) as avg_return,
                AVG(holding_days) as avg_holding_days,
                SUM(CASE WHEN total_return_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM portfolio_holdings
            WHERE position_status = 'active'
            """
        )

        stats = dict(cursor.fetchone())

        # Get best/worst performers
        cursor.execute(
            """
            SELECT ticker, total_return_pct as return
            FROM portfolio_holdings
            WHERE position_status = 'active'
            ORDER BY total_return_pct DESC
            LIMIT 1
            """
        )
        best = cursor.fetchone()
        stats["best_performer"] = dict(best) if best else {"ticker": "N/A", "return": 0}

        cursor.execute(
            """
            SELECT ticker, total_return_pct as return
            FROM portfolio_holdings
            WHERE position_status = 'active'
            ORDER BY total_return_pct ASC
            LIMIT 1
            """
        )
        worst = cursor.fetchone()
        stats["worst_performer"] = (
            dict(worst) if worst else {"ticker": "N/A", "return": 0}
        )

        return stats

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_stock_finds_by_run_id(self, run_id: int) -> List[Dict]:
        """Get all stock finds from a specific analysis run."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM stock_finds
            WHERE analysis_run_id = ?
            ORDER BY confidence_score DESC
            """,
            (run_id,),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            # Parse JSON fields
            if result.get("catalysts"):
                result["catalysts"] = json.loads(result["catalysts"])
            if result.get("risks"):
                result["risks"] = json.loads(result["risks"])
            results.append(result)

        return results

    def format_learning_insights_for_agent(self, insights: List[Dict]) -> str:
        """Format learning insights as JSON string for agent consumption."""
        if not insights:
            return json.dumps({"insights": [], "count": 0}, indent=2)

        formatted_insights = []
        for insight in insights:
            formatted_insights.append(
                {
                    "type": insight["insight_type"],
                    "category": insight.get("insight_category"),
                    "metric": insight["metric_name"],
                    "value": insight["metric_value"],
                    "sample_size": insight.get("sample_size"),
                    "summary": insight["insight_summary"],
                    "action": insight["actionable_recommendation"],
                    "confidence": insight["confidence_level"],
                }
            )

        return json.dumps(
            {"insights": formatted_insights, "count": len(formatted_insights)}, indent=2
        )
