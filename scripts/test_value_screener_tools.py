"""Diagnostic harness for the Value Stock Screening Specialist tools.

Run this script to exercise each tool the Value Stock Screening Specialist agent
relies on and capture detailed error output when data providers are unavailable.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no branch - deterministic insert
    sys.path.insert(0, str(PROJECT_ROOT))

EXECUTOR = ThreadPoolExecutor(max_workers=4)

from stockbot.tools.data import (
    CompanyInfoTool,
    FinancialReportTool,
    RealTimeQuoteTool,
    StockPriceDataTool,
)


def _print_divider(label: str) -> None:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)


def _format_result(payload: Any, *, max_chars: int = 400) -> str:
    if isinstance(payload, (dict, list, tuple)):
        try:
            serialized = json.dumps(payload, indent=2)
        except (TypeError, ValueError):
            serialized = str(payload)
    else:
        serialized = str(payload)

    serialized = serialized.strip()
    if len(serialized) > max_chars:
        serialized = serialized[: max_chars - 3] + "..."
    return serialized or "<empty result>"


def _invoke_with_timeout(func: Callable[[str], Any], symbol: str, timeout: float) -> Any:
    future = EXECUTOR.submit(func, symbol)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout as exc:
        future.cancel()
        raise TimeoutError(f"Execution exceeded {timeout} seconds") from exc


def _run_tool(tool_name: str, runner: Callable[[str], Any], symbol: str, timeout: float) -> None:
    _print_divider(f"Testing {tool_name} with symbol {symbol}")
    try:
        result = _invoke_with_timeout(runner, symbol, timeout)
    except Exception:  # noqa: BLE001 - we want the full traceback for diagnostics
        print("Status: FAILED")
        print("Traceback:")
        print("-" * 80)
        print(traceback.format_exc())
    else:
        print("Status: SUCCESS")
        print("Result snippet:")
        print("-" * 80)
        print(_format_result(result))
        if isinstance(result, dict) and "error" in result:
            print("\nReported error:")
            print(textwrap.fill(str(result["error"]), width=80))
        if isinstance(result, dict) and "fallback" in result:
            print("\nFallback payload available.")


def _build_tool_runners() -> Iterable[Tuple[str, Callable[[str], Any]]]:
    price_tool = StockPriceDataTool()
    quote_tool = RealTimeQuoteTool()
    info_tool = CompanyInfoTool()
    report_tool = FinancialReportTool()

    return (
        ("StockPriceDataTool", lambda symbol: price_tool.run(symbol, "1y")),
        ("RealTimeQuoteTool", quote_tool.run),
        ("CompanyInfoTool", info_tool.run),
        ("FinancialReportTool", report_tool.run),
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Diagnose the Value Stock Screening Specialist tool failures",
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="NVDA",
        help="Ticker to exercise (default: NVDA)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Per-tool timeout in seconds (default: 8.0)",
    )
    args = parser.parse_args()

    print("Diagnostic harness for Value Stock Screening Specialist tools")
    print(f"Ticker under test: {args.symbol}")

    for tool_name, runner in _build_tool_runners():
        _run_tool(tool_name, runner, args.symbol, args.timeout)

    EXECUTOR.shutdown(wait=False)


if __name__ == "__main__":  # pragma: no cover - manual diagnostic script
    main()
