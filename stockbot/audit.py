"""JSON state files for in-flight analysis runs.

Each analysis flow writes its current phase to ``state/{run_type}_state.json``
so a crash mid-run leaves a recoverable breadcrumb. The state file is removed
on successful completion. Reading the file from a fresh process tells you
where the previous run was when it failed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytz

ny_timezone = pytz.timezone("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"


def _state_path(run_type: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{run_type}_state.json"


def write_state(run_type: str, **fields: Any) -> Path:
    """Atomically write the in-flight state for a run type."""
    path = _state_path(run_type)
    payload = {
        "run_type": run_type,
        "updated_at": datetime.now(ny_timezone).isoformat(),
        **fields,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def read_state(run_type: str) -> Optional[Dict[str, Any]]:
    """Return the last written state for a run type, or None."""
    path = _state_path(run_type)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def clear_state(run_type: str) -> None:
    """Remove the state file for a completed run."""
    path = _state_path(run_type)
    if path.exists():
        path.unlink()
