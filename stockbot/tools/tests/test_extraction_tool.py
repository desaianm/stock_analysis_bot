"""Standalone diagnostic for `ExtractionTool`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stockbot.tools.data import ExtractionTool
from stockbot.tools.tests.common import (
    execute_tool,
    load_environment,
    render_report,
    shutdown_executor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the metric extraction helper")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="AAPL revenue net_income eps",
        help="Space-separated symbol and metrics",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=4.0,
        help="Execution timeout in seconds",
    )
    args = parser.parse_args()

    load_environment()
    tool = ExtractionTool()
    outcome = execute_tool(tool.run, args.prompt, timeout=args.timeout)
    print(render_report("ExtractionTool", outcome))
    shutdown_executor()


if __name__ == "__main__":
    main()
