"""SQLite database manager for stock analysis persistence."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz

ny_timezone = pytz.timezone("America/New_York")


class StockDatabaseManager:
    """Manages SQLite database for stock research and findings."""

    def __init__(self, db_path: str = "stock_analysis.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._initialize_database()

    def _initialize_database(self):
        """Create tables from schema file if they don't exist."""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema = f.read()
            self.conn.executescript(schema)
            self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    # -------------------------------------------------------------------------
    # Analysis Runs
    # -------------------------------------------------------------------------

    def create_analysis_run(
        self, run_type: str, preferences: Optional[Dict] = None
    ) -> int:
        """Start a new analysis run and return its ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO analysis_runs (run_type, started_at, preferences, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_type,
                datetime.now(ny_timezone).isoformat(),
                json.dumps(preferences) if preferences else None,
                "running",
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def complete_analysis_run(
        self,
        run_id: int,
        total_candidates: int = 0,
        final_selections: int = 0,
        status: str = "completed",
    ):
        """Mark analysis run as completed."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE analysis_runs
            SET completed_at = ?, status = ?, total_candidates = ?, final_selections = ?
            WHERE id = ?
            """,
            (
                datetime.now(ny_timezone).isoformat(),
                status,
                total_candidates,
                final_selections,
                run_id,
            ),
        )
        self.conn.commit()

    def get_recent_runs(self, run_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get recent analysis runs."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM analysis_runs"
        params: Tuple = ()

        if run_type:
            query += " WHERE run_type = ?"
            params = (run_type,)

        query += " ORDER BY started_at DESC LIMIT ?"
        params = params + (limit,)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Stock Finds
    # -------------------------------------------------------------------------

    def save_stock_find(self, find_data: Dict[str, Any]) -> int:
        """Save a stock finding to database."""
        cursor = self.conn.cursor()

        # Extract and prepare data
        ticker = find_data.get("ticker", "").upper()
        discovered_at = find_data.get("discovered_at") or datetime.now(ny_timezone).isoformat()

        cursor.execute(
            """
            INSERT INTO stock_finds (
                analysis_run_id, ticker, company_name, exchange, sector, industry,
                discovery_source, confidence_score, current_price, market_cap,
                pe_ratio, price_to_book, debt_to_equity, current_ratio, revenue_growth,
                analyst_rating, investment_thesis, catalysts, risks,
                price_target_bull, price_target_base, price_target_bear,
                discovered_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                find_data.get("analysis_run_id"),
                ticker,
                find_data.get("company_name"),
                find_data.get("exchange"),
                find_data.get("sector"),
                find_data.get("industry"),
                find_data.get("discovery_source", "screening"),
                find_data.get("confidence_score"),
                find_data.get("current_price"),
                find_data.get("market_cap"),
                find_data.get("pe_ratio"),
                find_data.get("price_to_book"),
                find_data.get("debt_to_equity"),
                find_data.get("current_ratio"),
                find_data.get("revenue_growth"),
                find_data.get("analyst_rating"),
                find_data.get("investment_thesis"),
                json.dumps(find_data.get("catalysts", [])),
                json.dumps(find_data.get("risks", [])),
                find_data.get("price_target_bull"),
                find_data.get("price_target_base"),
                find_data.get("price_target_bear"),
                discovered_at,
                "active",
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_stock_find(self, ticker: str, updates: Dict[str, Any]):
        """Update an existing stock find."""
        # Build dynamic UPDATE query
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ["catalysts", "risks"] and isinstance(value, (list, dict)):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            values.append(value)

        values.append(datetime.now(ny_timezone).isoformat())
        values.append(ticker)

        query = f"""
            UPDATE stock_finds
            SET {', '.join(set_clauses)}, last_updated = ?
            WHERE ticker = ?
        """

        cursor = self.conn.cursor()
        cursor.execute(query, values)
        self.conn.commit()

    def get_ticker_history(self, ticker: str) -> List[Dict]:
        """Get all historical finds for a ticker."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM stock_finds
            WHERE ticker = ?
            ORDER BY discovered_at DESC
            """,
            (ticker.upper(),),
        )
        return [dict(row) for row in cursor.fetchall()]

    def search_past_finds(
        self,
        query: Optional[str] = None,
        exchange: Optional[str] = None,
        sector: Optional[str] = None,
        min_confidence: Optional[float] = None,
        days_ago: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Search past stock finds with flexible filters."""
        cursor = self.conn.cursor()

        where_clauses = ["status = 'active'"]
        params: List[Any] = []

        if query:
            where_clauses.append(
                "(ticker LIKE ? OR company_name LIKE ? OR investment_thesis LIKE ?)"
            )
            search_term = f"%{query}%"
            params.extend([search_term, search_term, search_term])

        if exchange:
            where_clauses.append("exchange = ?")
            params.append(exchange.upper())

        if sector:
            where_clauses.append("sector LIKE ?")
            params.append(f"%{sector}%")

        if min_confidence:
            where_clauses.append("confidence_score >= ?")
            params.append(min_confidence)

        if days_ago:
            cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days_ago)).isoformat()
            where_clauses.append("discovered_at >= ?")
            params.append(cutoff_date)

        sql = f"""
            SELECT * FROM stock_finds
            WHERE {' AND '.join(where_clauses)}
            ORDER BY discovered_at DESC, confidence_score DESC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(sql, params)
        results = [dict(row) for row in cursor.fetchall()]

        # Parse JSON fields
        for result in results:
            if result.get("catalysts"):
                try:
                    result["catalysts"] = json.loads(result["catalysts"])
                except:
                    pass
            if result.get("risks"):
                try:
                    result["risks"] = json.loads(result["risks"])
                except:
                    pass

        return results

    def get_similar_stocks(
        self, ticker: str, sector: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Find similar stocks based on sector/industry."""
        cursor = self.conn.cursor()

        # First get the reference stock's info
        cursor.execute(
            "SELECT sector, industry FROM stock_finds WHERE ticker = ? ORDER BY discovered_at DESC LIMIT 1",
            (ticker.upper(),),
        )
        ref_stock = cursor.fetchone()

        if not ref_stock:
            return []

        ref_sector = sector or ref_stock["sector"]
        ref_industry = ref_stock["industry"]

        # Find similar stocks
        cursor.execute(
            """
            SELECT * FROM stock_finds
            WHERE ticker != ?
            AND status = 'active'
            AND (sector = ? OR industry = ?)
            ORDER BY
                CASE WHEN industry = ? THEN 2 WHEN sector = ? THEN 1 ELSE 0 END DESC,
                confidence_score DESC,
                discovered_at DESC
            LIMIT ?
            """,
            (ticker.upper(), ref_sector, ref_industry, ref_industry, ref_sector, limit),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_recent_discoveries(self, days: int = 30, min_confidence: float = 5.0) -> List[Dict]:
        """Get recent high-quality discoveries."""
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM stock_finds
            WHERE discovered_at >= ?
            AND confidence_score >= ?
            AND status = 'active'
            ORDER BY confidence_score DESC, discovered_at DESC
            """,
            (cutoff_date, min_confidence),
        )

        results = [dict(row) for row in cursor.fetchall()]

        # Parse JSON fields
        for result in results:
            if result.get("catalysts"):
                try:
                    result["catalysts"] = json.loads(result["catalysts"])
                except:
                    pass
            if result.get("risks"):
                try:
                    result["risks"] = json.loads(result["risks"])
                except:
                    pass

        return results

    # -------------------------------------------------------------------------
    # Reddit Mentions
    # -------------------------------------------------------------------------

    def save_reddit_mention(self, mention_data: Dict[str, Any]) -> int:
        """Save a Reddit mention."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO reddit_mentions (
                ticker, subreddit, title, selftext, sentiment,
                score, num_comments, permalink, top_comments, mentioned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mention_data.get("ticker", "").upper(),
                mention_data.get("subreddit"),
                mention_data.get("title"),
                mention_data.get("selftext"),
                mention_data.get("sentiment"),
                mention_data.get("score"),
                mention_data.get("num_comments"),
                mention_data.get("permalink"),
                json.dumps(mention_data.get("top_comments", [])),
                mention_data.get("mentioned_at") or datetime.now(ny_timezone).isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_reddit_mentions(self, ticker: str, days: int = 30) -> List[Dict]:
        """Get recent Reddit mentions for a ticker."""
        cutoff_date = (datetime.now(ny_timezone) - timedelta(days=days)).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM reddit_mentions
            WHERE ticker = ?
            AND mentioned_at >= ?
            ORDER BY mentioned_at DESC
            """,
            (ticker.upper(), cutoff_date),
        )

        results = [dict(row) for row in cursor.fetchall()]

        # Parse JSON fields
        for result in results:
            if result.get("top_comments"):
                try:
                    result["top_comments"] = json.loads(result["top_comments"])
                except:
                    pass

        return results

    # -------------------------------------------------------------------------
    # Financial Metrics
    # -------------------------------------------------------------------------

    def save_financial_metrics(self, ticker: str, metrics: Dict[str, Any], metric_date: str):
        """Save financial metrics for a ticker."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO financial_metrics (
                ticker, metric_date, revenue, net_income, operating_cash_flow,
                free_cash_flow, total_debt, shareholders_equity, shares_outstanding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.upper(),
                metric_date,
                metrics.get("revenue"),
                metrics.get("net_income"),
                metrics.get("operating_cash_flow"),
                metrics.get("free_cash_flow"),
                metrics.get("total_debt"),
                metrics.get("shareholders_equity"),
                metrics.get("shares_outstanding"),
            ),
        )
        self.conn.commit()

    def get_financial_metrics(self, ticker: str, limit: int = 12) -> List[Dict]:
        """Get historical financial metrics for a ticker."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM financial_metrics
            WHERE ticker = ?
            ORDER BY metric_date DESC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Research Notes
    # -------------------------------------------------------------------------

    def add_research_note(
        self, ticker: str, note_type: str, content: str, source: str = "agent"
    ) -> int:
        """Add a research note for a ticker."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO research_notes (ticker, note_type, content, source)
            VALUES (?, ?, ?, ?)
            """,
            (ticker.upper(), note_type, content, source),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_research_notes(
        self, ticker: str, note_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """Get research notes for a ticker."""
        cursor = self.conn.cursor()

        if note_type:
            cursor.execute(
                """
                SELECT * FROM research_notes
                WHERE ticker = ? AND note_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (ticker.upper(), note_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM research_notes
                WHERE ticker = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (ticker.upper(), limit),
            )

        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        stats = {}

        # Count records in each table
        for table in [
            "analysis_runs",
            "stock_finds",
            "reddit_mentions",
            "financial_metrics",
            "research_notes",
        ]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()["count"]

        # Active stocks
        cursor.execute("SELECT COUNT(*) as count FROM stock_finds WHERE status = 'active'")
        stats["active_stocks"] = cursor.fetchone()["count"]

        # Unique tickers
        cursor.execute("SELECT COUNT(DISTINCT ticker) as count FROM stock_finds")
        stats["unique_tickers"] = cursor.fetchone()["count"]

        return stats

    def format_for_agent(self, finds: List[Dict]) -> str:
        """Format stock finds as JSON string for agent consumption."""
        # Simplify data for agent
        simplified = []
        for find in finds:
            simplified.append({
                "ticker": find["ticker"],
                "company_name": find.get("company_name"),
                "exchange": find.get("exchange"),
                "sector": find.get("sector"),
                "confidence_score": find.get("confidence_score"),
                "current_price": find.get("current_price"),
                "market_cap": find.get("market_cap"),
                "investment_thesis": find.get("investment_thesis"),
                "catalysts": find.get("catalysts", []),
                "discovered_at": find.get("discovered_at"),
            })

        return json.dumps({"past_finds": simplified, "total": len(simplified)}, indent=2)
