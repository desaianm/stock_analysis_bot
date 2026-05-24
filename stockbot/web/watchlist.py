"""JSON-backed watchlist store. Lives outside SQLite per project DB constraints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytz

ny_timezone = pytz.timezone("America/New_York")

WATCHLIST_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "watchlist.json"


@dataclass
class WatchEntry:
    ticker: str
    added_at: str
    interest_level: int = 3        # 1-5 stars
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _load_raw() -> Dict:
    if not WATCHLIST_PATH.exists():
        return {"items": []}
    try:
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"items": []}


def _save_raw(payload: Dict) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def list_watchlist() -> List[WatchEntry]:
    payload = _load_raw()
    out: List[WatchEntry] = []
    for item in payload.get("items", []):
        try:
            out.append(WatchEntry(**item))
        except TypeError:
            continue
    return out


def get_entry(ticker: str) -> Optional[WatchEntry]:
    ticker = ticker.upper()
    for entry in list_watchlist():
        if entry.ticker == ticker:
            return entry
    return None


def add_to_watchlist(ticker: str, interest_level: int = 3, notes: str = "") -> WatchEntry:
    ticker = ticker.upper()
    payload = _load_raw()
    items = payload.get("items", [])
    for existing in items:
        if existing.get("ticker", "").upper() == ticker:
            existing["interest_level"] = interest_level
            existing["notes"] = notes
            _save_raw(payload)
            return WatchEntry(**existing)
    entry = WatchEntry(
        ticker=ticker,
        added_at=datetime.now(ny_timezone).isoformat(),
        interest_level=interest_level,
        notes=notes,
    )
    items.append(entry.to_dict())
    payload["items"] = items
    _save_raw(payload)
    return entry


def remove_from_watchlist(ticker: str) -> bool:
    ticker = ticker.upper()
    payload = _load_raw()
    items = payload.get("items", [])
    new_items = [i for i in items if i.get("ticker", "").upper() != ticker]
    if len(new_items) == len(items):
        return False
    payload["items"] = new_items
    _save_raw(payload)
    return True


def update_interest(ticker: str, interest_level: int) -> Optional[WatchEntry]:
    ticker = ticker.upper()
    payload = _load_raw()
    for item in payload.get("items", []):
        if item.get("ticker", "").upper() == ticker:
            item["interest_level"] = max(1, min(5, int(interest_level)))
            _save_raw(payload)
            return WatchEntry(**item)
    return None
