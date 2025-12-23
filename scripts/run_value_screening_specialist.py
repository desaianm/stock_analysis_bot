"""Run only the Value Stock Screening Specialist agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no branch - deterministic insert
    sys.path.insert(0, str(PROJECT_ROOT))

from stockbot.flows.undervalued import (
    UndervaluedAnalysisFlow,
    ValueScreeningPreferences,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Value Stock Screening Specialist agent in isolation.",
    )
    parser.add_argument("--max-price", type=float, default=100.0, help="Maximum share price")
    parser.add_argument("--min-price", type=float, default=5.0, help="Minimum share price")
    parser.add_argument("--min-volume", type=float, default=500_000, help="Minimum average volume")
    parser.add_argument("--max-pe", type=float, default=25.0, help="Maximum P/E ratio")
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=300_000_000,
        help="Minimum market capitalization",
    )
    parser.add_argument("--min-current-ratio", type=float, default=1.5, help="Minimum current ratio")
    parser.add_argument("--max-debt-equity", type=float, default=2.0, help="Maximum debt/equity ratio")
    parser.add_argument(
        "--price-vs-high",
        type=float,
        default=0.4,
        help="Maximum decline from 52-week high (fraction)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw agent output as JSON (falls back to plain text if serialization fails)",
    )
    return parser.parse_args()


def _build_preferences(args: argparse.Namespace) -> ValueScreeningPreferences:
    return ValueScreeningPreferences(
        max_price=args.max_price,
        min_price=args.min_price,
        min_volume=args.min_volume,
        max_pe=args.max_pe,
        min_market_cap=args.min_market_cap,
        min_current_ratio=args.min_current_ratio,
        max_debt_equity=args.max_debt_equity,
        price_vs_high=args.price_vs_high,
    )


async def _run_value_screening(
    preferences: ValueScreeningPreferences,
) -> Dict[str, Any]:
    flow = UndervaluedAnalysisFlow(preferences)
    result = await flow.run_value_screening()
    return {
        "preferences": preferences.model_dump(),
        "raw_output": result,
    }


def main() -> None:
    args = _parse_args()
    load_dotenv()

    preferences = _build_preferences(args)

    try:
        outcome = asyncio.run(_run_value_screening(preferences))
    except Exception as exc:  # noqa: BLE001 - surface stack for diagnostics
        print("Value Screening Specialist failed:\n")
        raise
    else:
        if args.json:
            try:
                print(json.dumps(outcome, indent=2))
            except (TypeError, ValueError):
                print(outcome)
        else:
            print("\n=== Value Screening Specialist Result ===")
            print("Preferences:")
            for key, value in outcome["preferences"].items():
                print(f"  - {key}: {value}")
            print("\nRaw output:\n")
            print(outcome["raw_output"])


if __name__ == "__main__":
    main()
