"""Forward-looking scarcity/capacity lane for the undervalued funnel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from stockbot.screening.numeric_screen import ScreeningGates, StockSnapshot
from stockbot.tools.institutional import InstitutionalPortfolio


SCARCITY_INDUSTRY_TERMS = (
    "communication equipment",
    "computer hardware",
    "data center",
    "electrical equipment",
    "electronic components",
    "infrastructure operations",
    "memory",
    "networking",
    "power generation",
    "semiconductor",
    "solar",
    "specialty industrial machinery",
    "storage",
    "utilities - renewable",
)
MIN_SCARCITY_SCORE = 4.0


@dataclass
class ScarcitySignal:
    symbol: str
    score: float
    reasons: List[str] = field(default_factory=list)
    institutional_direct_long_value: float = 0.0
    institutional_positions: List[dict] = field(default_factory=list)


def _at_least(value: Optional[float], threshold: float) -> bool:
    return value is not None and value >= threshold


def score_scarcity_candidate(
    snapshot: StockSnapshot,
    gates: ScreeningGates,
    portfolio: Optional[InstitutionalPortfolio] = None,
) -> Optional[ScarcitySignal]:
    """Score capacity-constrained growth without relaxing basic liquidity safety."""
    if snapshot.error:
        return None
    if snapshot.price is None or snapshot.price < gates.min_price:
        return None
    if snapshot.volume is None or snapshot.volume < gates.min_volume:
        return None
    if snapshot.market_cap is None or snapshot.market_cap < gates.min_market_cap:
        return None

    positions = portfolio.positions_for(snapshot.company_name) if portfolio else []
    direct_long_value = (
        portfolio.direct_long_value(snapshot.company_name) if portfolio else 0.0
    )
    industry = (snapshot.industry or "").lower()
    thematic_match = any(term in industry for term in SCARCITY_INDUSTRY_TERMS)
    if not thematic_match and direct_long_value <= 0:
        return None

    # This lane can admit a temporarily expensive stock, but it still requires
    # cash generation or clearly strong top-line growth.
    if not (
        snapshot.free_cash_flow is not None
        and snapshot.free_cash_flow > 0
    ) and not _at_least(snapshot.revenue_growth, 0.20):
        return None

    score = 1.5
    reasons = [f"capacity-sensitive industry: {snapshot.industry or 'unknown'}"]

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

    if direct_long_value > 0:
        score += 1.5
        reasons.append(
            "direct long in latest Situational Awareness 13F "
            f"(${direct_long_value / 1_000_000:.1f}M disclosed value)"
        )

    score = round(score, 2)
    if score < MIN_SCARCITY_SCORE:
        return None
    return ScarcitySignal(
        symbol=snapshot.symbol,
        score=score,
        reasons=reasons,
        institutional_direct_long_value=direct_long_value,
        institutional_positions=positions,
    )
