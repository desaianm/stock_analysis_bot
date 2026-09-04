"""Financial validation for supply-chain bottlenecks found by web research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from stockbot.screening.numeric_screen import ScreeningGates, StockSnapshot


MIN_SCARCITY_SCORE = 4.0


@dataclass
class ScarcitySignal:
    symbol: str
    score: float
    reasons: List[str] = field(default_factory=list)
    research: Dict[str, Any] = field(default_factory=dict)


def _at_least(value: Optional[float], threshold: float) -> bool:
    return value is not None and value >= threshold


def score_scarcity_candidate(
    snapshot: StockSnapshot,
    gates: ScreeningGates,
    research: Optional[Dict[str, Any]],
) -> Optional[ScarcitySignal]:
    """Validate a researched bottleneck beneficiary against live fundamentals.

    Web research supplies the changing industry thesis. This function is
    intentionally industry-agnostic and only decides whether the named public
    company has enough liquidity and financial traction to enter the funnel.
    """
    if not research or snapshot.error:
        return None
    if snapshot.price is None or snapshot.price < gates.min_price:
        return None
    if snapshot.volume is None or snapshot.volume < gates.min_volume:
        return None
    if snapshot.market_cap is None or snapshot.market_cap < gates.min_market_cap:
        return None

    # A researched company may exceed the classic price/P-E ceiling, but it
    # still needs positive FCF or strong top-line growth to avoid pure stories.
    if not (
        snapshot.free_cash_flow is not None and snapshot.free_cash_flow > 0
    ) and not _at_least(snapshot.revenue_growth, 0.20):
        return None

    confidence = float(research.get("confidence_score") or 0.0)
    if confidence < 6.0:
        return None
    score = min(confidence, 10.0) * 0.30
    reasons = [
        f"researched bottleneck: {research.get('theme', 'unknown')}",
        f"supply-chain role: {research.get('role', 'unknown')}",
    ]

    revenue_growth = snapshot.revenue_growth or 0.0
    if revenue_growth >= 0.40:
        score += 3.0
    elif revenue_growth >= 0.20:
        score += 2.25
    elif revenue_growth >= 0.10:
        score += 1.5
    elif revenue_growth > 0:
        score += 0.5
    if revenue_growth > 0:
        reasons.append(f"revenue growth {revenue_growth:.1%}")

    earnings_growth = snapshot.earnings_growth or 0.0
    if earnings_growth >= 0.30:
        score += 1.5
    elif earnings_growth >= 0.10:
        score += 1.0
    elif earnings_growth > 0:
        score += 0.5
    if earnings_growth > 0:
        reasons.append(f"earnings growth {earnings_growth:.1%}")

    if _at_least(snapshot.gross_margins, 0.40):
        score += 1.0
        reasons.append(f"gross margin {snapshot.gross_margins:.1%}")
    elif _at_least(snapshot.gross_margins, 0.20):
        score += 0.5

    if snapshot.free_cash_flow is not None and snapshot.free_cash_flow > 0:
        score += 1.0
        reasons.append("positive free cash flow")

    analyst_upside = snapshot.analyst_upside or 0.0
    if analyst_upside >= 0.25:
        score += 1.0
    elif analyst_upside >= 0.10:
        score += 0.5
    if analyst_upside >= 0.10:
        reasons.append(f"analyst upside {analyst_upside:.1%}")

    drawdown = (
        1 - snapshot.price_vs_52w_high
        if snapshot.price_vs_52w_high is not None
        else 0.0
    )
    if 0.10 <= drawdown <= 0.50:
        score += 0.5
        reasons.append(f"{drawdown:.1%} below 52-week high")

    score = round(score, 2)
    if score < MIN_SCARCITY_SCORE:
        return None
    return ScarcitySignal(
        symbol=snapshot.symbol,
        score=score,
        reasons=reasons,
        research=research,
    )
