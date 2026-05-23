"""Insider trades tool backed by Financial Datasets REST API.

Surfaces 90-day net open-market buying by insiders — one of the most-validated
value signals in academic finance. Returns a structured summary suitable for
both deterministic ranking and agent reasoning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests


@dataclass
class InsiderSummary:
    ticker: str
    window_days: int
    net_value_usd: float            # purchases − sales (open market only)
    buy_count: int
    sell_count: int
    distinct_buyers: int
    largest_buy_usd: float
    most_recent_buy_date: Optional[str]
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "window_days": self.window_days,
            "net_value_usd": self.net_value_usd,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "distinct_buyers": self.distinct_buyers,
            "largest_buy_usd": self.largest_buy_usd,
            "most_recent_buy_date": self.most_recent_buy_date,
            "transactions": self.transactions[:10],  # cap for prompt budget
            "error": self.error,
        }


_API = "https://api.financialdatasets.ai/insider-trades/"


def _is_open_market_buy(t: Dict[str, Any]) -> bool:
    txn_type = (t.get("transaction_type") or "").lower()
    shares = t.get("transaction_shares") or 0
    # Open market purchase, ignore option grants/exercises and gifts
    return "open market purchase" in txn_type and shares > 0


def _is_open_market_sale(t: Dict[str, Any]) -> bool:
    txn_type = (t.get("transaction_type") or "").lower()
    shares = t.get("transaction_shares") or 0
    return "open market sale" in txn_type and shares > 0


def fetch_insider_summary(ticker: str, days: int = 90, limit: int = 100) -> InsiderSummary:
    """Fetch insider transactions and reduce to a single ranked summary."""
    key = os.getenv("FINANCIAL_DATASETS_API_KEY")
    if not key:
        return InsiderSummary(
            ticker=ticker.upper(),
            window_days=days,
            net_value_usd=0.0,
            buy_count=0,
            sell_count=0,
            distinct_buyers=0,
            largest_buy_usd=0.0,
            most_recent_buy_date=None,
            error="FINANCIAL_DATASETS_API_KEY not set",
        )

    try:
        r = requests.get(
            _API,
            params={"ticker": ticker.upper(), "limit": limit},
            headers={"X-API-KEY": key},
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json() or {}
    except Exception as exc:  # noqa: BLE001
        return InsiderSummary(
            ticker=ticker.upper(),
            window_days=days,
            net_value_usd=0.0,
            buy_count=0,
            sell_count=0,
            distinct_buyers=0,
            largest_buy_usd=0.0,
            most_recent_buy_date=None,
            error=str(exc),
        )

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    transactions = payload.get("insider_trades") or []

    buy_total = 0.0
    sell_total = 0.0
    buy_count = 0
    sell_count = 0
    largest_buy = 0.0
    buyers: set[str] = set()
    most_recent_buy: Optional[str] = None
    kept: List[Dict[str, Any]] = []

    for txn in transactions:
        date_str = txn.get("transaction_date") or txn.get("filing_date")
        if not date_str:
            continue
        try:
            txn_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if txn_date.tzinfo is None:
                txn_date = txn_date.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if txn_date < cutoff:
            continue

        value = float(txn.get("transaction_value") or 0.0)

        if _is_open_market_buy(txn):
            buy_total += value
            buy_count += 1
            buyers.add(txn.get("name") or "unknown")
            if value > largest_buy:
                largest_buy = value
            if most_recent_buy is None or date_str > most_recent_buy:
                most_recent_buy = date_str
            kept.append(
                {
                    "date": date_str,
                    "name": txn.get("name"),
                    "title": txn.get("title"),
                    "type": "buy",
                    "value": value,
                    "shares": txn.get("transaction_shares"),
                }
            )
        elif _is_open_market_sale(txn):
            sell_total += value
            sell_count += 1
            kept.append(
                {
                    "date": date_str,
                    "name": txn.get("name"),
                    "title": txn.get("title"),
                    "type": "sell",
                    "value": value,
                    "shares": txn.get("transaction_shares"),
                }
            )

    return InsiderSummary(
        ticker=ticker.upper(),
        window_days=days,
        net_value_usd=round(buy_total - sell_total, 2),
        buy_count=buy_count,
        sell_count=sell_count,
        distinct_buyers=len(buyers),
        largest_buy_usd=round(largest_buy, 2),
        most_recent_buy_date=most_recent_buy,
        transactions=kept,
    )


def insider_signal_score(summary: InsiderSummary) -> float:
    """Reduce an InsiderSummary to a 0-10 score for funnel ranking.

    Heuristic: net buying matters more than gross activity. Multiple distinct
    buyers is a stronger signal than one large buyer. Capped at 10.
    """
    if summary.error or summary.net_value_usd <= 0:
        return 0.0
    # log-scale net buying in $ so a $5M buy doesn't dwarf a $100k buy by 50x
    import math
    log_score = math.log10(max(summary.net_value_usd, 1.0))  # ~3 at $1k, ~6 at $1M, ~9 at $1B
    base = max(0.0, log_score - 3.0)  # 0 at $1k, ~3 at $1M, ~6 at $1B
    distinct_bonus = min(summary.distinct_buyers * 0.5, 2.0)
    recency_bonus = 1.0 if summary.most_recent_buy_date else 0.0
    return min(10.0, base + distinct_bonus + recency_bonus)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    for t in ["AAPL", "NVDA", "F"]:
        s = fetch_insider_summary(t, days=180)
        print(f"\n{t}:")
        print(f"  net={s.net_value_usd:>15,.0f}  buys={s.buy_count}  sells={s.sell_count}  buyers={s.distinct_buyers}")
        print(f"  most recent buy: {s.most_recent_buy_date}")
        print(f"  signal score: {insider_signal_score(s):.2f}/10")
        if s.error:
            print(f"  ERROR: {s.error}")
