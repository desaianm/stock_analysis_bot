"""Standalone diagnostic for `ChartingTool`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stockbot.tools.data import ChartingTool
from stockbot.tools.tests.common import (
    execute_tool,
    load_environment,
    render_report,
    shutdown_executor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test static chart generation")
    parser.add_argument(
        "--metric-name",
        default="Sample Metric",
        help="Metric label for the chart",
    )
    parser.add_argument(
        "--data",
        type=float,
        nargs="+",
        default=[100.0, 120.5, 98.2, 140.0, 132.7],
        help="Numeric data points to plot",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Execution timeout in seconds",
    )
    args = parser.parse_args()

    load_environment()
    Path("plots").mkdir(parents=True, exist_ok=True)
    tool = ChartingTool()
    outcome = execute_tool(tool.run, args.metric_name, args.data, timeout=args.timeout)
    print(render_report("ChartingTool", outcome))
    shutdown_executor()


if __name__ == "__main__":
    main()
