"""Standalone diagnostic for `CompanyInfoTool`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stockbot.tools.data import CompanyInfoTool
from stockbot.tools.tests.common import (
    execute_tool,
    load_environment,
    render_report,
    shutdown_executor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test company snapshot retrieval")
    parser.add_argument(
        "symbol",
        nargs="?",
        default="NVDA",
        help="Ticker symbol",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Execution timeout in seconds",
    )
    args = parser.parse_args()

    load_environment()
    tool = CompanyInfoTool()
    outcome = execute_tool(tool.run, args.symbol, timeout=args.timeout)
    print(render_report("CompanyInfoTool", outcome))
    shutdown_executor()


if __name__ == "__main__":
    main()
