"""Shared helpers for standalone tool diagnostics."""

from __future__ import annotations

import json
import sys
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no branch - deterministic insert
    sys.path.insert(0, str(PROJECT_ROOT))

_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def load_environment() -> None:
    """Load environment variables required by tooling."""

    load_dotenv()


def format_payload(payload: Any, *, max_chars: int = 600) -> str:
    """Return a human-readable preview of arbitrary payloads."""

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
    return serialized or "<empty>"


def execute_tool(
    runner: Callable[..., Any],
    *args: Any,
    timeout: float = 8.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a tool callable, capturing success, errors, and timeouts."""

    future = _EXECUTOR.submit(runner, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
    except FuturesTimeout as exc:
        future.cancel()
        return {
            "status": "timeout",
            "error": f"Execution exceeded {timeout} seconds",
            "traceback": None,
        }
    except Exception:  # noqa: BLE001 - diagnostic reporting
        return {
            "status": "error",
            "error": "Tool raised an exception",
            "traceback": traceback.format_exc(),
        }
    else:
        return {
            "status": "success",
            "result": result,
        }


def render_report(tool_name: str, outcome: Dict[str, Any]) -> str:
    """Create a formatted string summarizing a tool execution outcome."""

    lines = ["=" * 80, f"Tool diagnostic: {tool_name}", "=" * 80]
    status = outcome.get("status", "unknown")
    lines.append(f"Status: {status.upper()}")

    if status == "success":
        lines.append("Result preview:")
        lines.append(textwrap.indent(format_payload(outcome.get("result")), prefix="  "))
        payload = outcome.get("result")
        if isinstance(payload, dict):
            if payload.get("error"):
                lines.append("\nReported error:")
                lines.append(textwrap.fill(str(payload["error"]), width=78))
            if payload.get("fallback"):
                lines.append("\nFallback payload present (see raw output).")
    else:
        if outcome.get("error"):
            lines.append("Error message:")
            lines.append(textwrap.fill(str(outcome["error"]), width=78))
        if outcome.get("traceback"):
            lines.append("\nTraceback:")
            lines.append(outcome["traceback"].rstrip())

    return "\n".join(lines)


def shutdown_executor() -> None:
    """Idempotently shut down the shared executor."""

    _EXECUTOR.shutdown(wait=False)
