"""Standalone diagnostic for `ChatAnalysisTool`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stockbot.tools.data import ChatAnalysisTool
from stockbot.tools.tests.common import (
    execute_tool,
    load_environment,
    render_report,
    shutdown_executor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gemini-based chart analysis")
    parser.add_argument(
        "--prompt",
        default="Summarize the key insights from available plots.",
        help="Instruction passed to the analysis tool",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Execution timeout in seconds",
    )
    args = parser.parse_args()

    load_environment()
    tool = ChatAnalysisTool()
    outcome = execute_tool(tool.run, args.prompt, timeout=args.timeout)
    print(render_report("ChatAnalysisTool", outcome))
    shutdown_executor()


if __name__ == "__main__":
    main()
