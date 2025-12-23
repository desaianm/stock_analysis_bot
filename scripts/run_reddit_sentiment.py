"""Run the BayStreet Reddit Scout agent in isolation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no branch
    sys.path.insert(0, str(PROJECT_ROOT))

from stockbot.flows.undervalued import (  # noqa: E402
    UndervaluedAnalysisFlow,
    ValueScreeningPreferences,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reddit sentiment agent for a ticker (TSX focus) or discover trending tickers.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker",
        help="Ticker symbol to scan (e.g., GELYF, TSLA, NVDA)",
    )
    group.add_argument(
        "--discover",
        action="store_true",
        help="Discover trending tickers on r/Baystreetbets / r/wallstreetbets.",
    )
    return parser.parse_args()


async def _run_agent(ticker: str) -> str:
    flow = UndervaluedAnalysisFlow(ValueScreeningPreferences())
    return await flow.run_reddit_sentiment_analysis(ticker)


async def _run_discovery() -> str:
    flow = UndervaluedAnalysisFlow(ValueScreeningPreferences())
    return flow.discover_reddit_tickers()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    try:
        if args.discover:
            report = asyncio.run(_run_discovery())
            print("\n=== Trending Reddit Tickers ===\n")
        else:
            report = asyncio.run(_run_agent(args.ticker))
            print(f"\n=== Reddit Sentiment Report for {args.ticker.upper()} ===\n")
    except Exception as exc:  # noqa: BLE001 - surface diagnostic info
        print(f"Reddit sentiment agent failed: {exc}")
        raise
    else:
        print(report)


if __name__ == "__main__":
    main()
