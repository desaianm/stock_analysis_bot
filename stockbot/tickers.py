"""Canonical market-data ticker handling."""

from __future__ import annotations

import re


_SYMBOL_RE = re.compile(r"^[A-Z0-9](?:[A-Z0-9.-]{0,13}[A-Z0-9])?$")
_EXCHANGE_RE = re.compile(r"^(?P<exchange>[A-Z]+)\s*:\s*(?P<symbol>.+)$")


def normalize_ticker(value: str) -> str:
    """Return a canonical yfinance symbol or raise ``ValueError``.

    Exchange prefixes are accepted for common agent output. TSX symbols gain
    Yahoo's ``.TO`` suffix; US exchange prefixes are removed. Existing valid
    market-data suffixes are preserved.
    """
    if not isinstance(value, str):
        raise ValueError("ticker must be a string")
    ticker = value.strip().upper()
    match = _EXCHANGE_RE.fullmatch(ticker)
    exchange = None
    if match:
        exchange = match.group("exchange")
        ticker = match.group("symbol").strip()
        if exchange not in {"TSX", "TSE", "NYSE", "NASDAQ", "AMEX"}:
            raise ValueError(f"unsupported exchange prefix: {exchange}")
    if not _SYMBOL_RE.fullmatch(ticker) or ".." in ticker:
        raise ValueError(f"invalid ticker: {value!r}")
    if exchange in {"TSX", "TSE"} and "." not in ticker:
        ticker = f"{ticker}.TO"
    elif ticker in {"BRK.B", "BF.B"}:
        ticker = ticker.replace(".", "-")
    return ticker
