"""Reverse-DCF: solve for the implied growth rate that justifies today's price.

Given current free cash flow, current enterprise value, a discount rate (WACC),
and a terminal growth rate, this answers: *what constant year-1-to-N FCF growth
rate would make the stock fairly valued today?*

If the implied growth far exceeds the company's historical growth, the stock is
expensive. If it's well below, there's a margin of safety.

Report the implied growth across three discount rates (8% / 10% / 12%) so the
reader can see sensitivity without us hand-waving a single WACC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReverseDCFResult:
    ticker: str
    current_fcf_usd: float
    enterprise_value_usd: float
    forecast_years: int
    terminal_growth: float
    by_discount_rate: Dict[str, float]   # e.g. {"8%": 0.054, "10%": 0.072, "12%": 0.090}
    notes: List[str]
    error: Optional[str] = None

    def implied_growth_at(self, discount_rate: float) -> Optional[float]:
        return self.by_discount_rate.get(f"{int(discount_rate * 100)}%")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "current_fcf_usd": self.current_fcf_usd,
            "enterprise_value_usd": self.enterprise_value_usd,
            "forecast_years": self.forecast_years,
            "terminal_growth": self.terminal_growth,
            "implied_growth_by_discount_rate": self.by_discount_rate,
            "notes": self.notes,
            "error": self.error,
        }


def _npv_two_stage(
    current_fcf: float, growth: float, discount: float, years: int, terminal_g: float
) -> float:
    """NPV of FCF growing at `growth` for `years`, then terminal at `terminal_g`."""
    npv = 0.0
    fcf = current_fcf
    for year in range(1, years + 1):
        fcf = fcf * (1 + growth)
        npv += fcf / ((1 + discount) ** year)
    # Terminal value: Gordon growth on year-N+1 FCF
    terminal_fcf = fcf * (1 + terminal_g)
    if discount <= terminal_g:
        return float("inf")  # degenerate
    terminal_value = terminal_fcf / (discount - terminal_g)
    npv += terminal_value / ((1 + discount) ** years)
    return npv


def solve_implied_growth(
    current_fcf: float,
    target_npv: float,
    discount: float,
    years: int = 10,
    terminal_g: float = 0.025,
) -> Optional[float]:
    """Binary-search the growth rate g such that NPV(g) == target_npv.

    Returns None if the search bounds don't bracket the solution.
    """
    if current_fcf <= 0 or target_npv <= 0:
        return None
    if discount <= terminal_g:
        return None

    low, high = -0.30, 0.50  # -30% to +50% annual growth
    f_low = _npv_two_stage(current_fcf, low, discount, years, terminal_g) - target_npv
    f_high = _npv_two_stage(current_fcf, high, discount, years, terminal_g) - target_npv

    # If both have the same sign, growth needed is outside the bracket
    if f_low * f_high > 0:
        # Try expanding upper bound
        for g in [0.75, 1.0, 1.5, 2.0]:
            f_g = _npv_two_stage(current_fcf, g, discount, years, terminal_g) - target_npv
            if f_low * f_g <= 0:
                high = g
                f_high = f_g
                break
        else:
            return None

    # Bisection
    for _ in range(60):
        mid = (low + high) / 2
        f_mid = _npv_two_stage(current_fcf, mid, discount, years, terminal_g) - target_npv
        if abs(f_mid) < 1.0:
            return mid
        if f_low * f_mid <= 0:
            high = mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2


def reverse_dcf(
    ticker: str,
    current_fcf: float,
    enterprise_value: float,
    years: int = 10,
    terminal_growth: float = 0.025,
    discount_rates: tuple[float, ...] = (0.08, 0.10, 0.12),
) -> ReverseDCFResult:
    """Compute implied growth across multiple discount-rate assumptions."""
    notes: List[str] = []

    if current_fcf is None or current_fcf <= 0:
        return ReverseDCFResult(
            ticker=ticker.upper(),
            current_fcf_usd=current_fcf or 0.0,
            enterprise_value_usd=enterprise_value or 0.0,
            forecast_years=years,
            terminal_growth=terminal_growth,
            by_discount_rate={},
            notes=[],
            error="Free cash flow is non-positive; reverse-DCF not meaningful",
        )

    if enterprise_value is None or enterprise_value <= 0:
        return ReverseDCFResult(
            ticker=ticker.upper(),
            current_fcf_usd=current_fcf,
            enterprise_value_usd=enterprise_value or 0.0,
            forecast_years=years,
            terminal_growth=terminal_growth,
            by_discount_rate={},
            notes=[],
            error="Enterprise value missing",
        )

    by_dr: Dict[str, float] = {}
    for dr in discount_rates:
        g = solve_implied_growth(current_fcf, enterprise_value, dr, years, terminal_growth)
        if g is None:
            notes.append(f"discount {int(dr * 100)}%: implied growth outside ±200% range")
            continue
        by_dr[f"{int(dr * 100)}%"] = round(g, 4)

    if not by_dr:
        return ReverseDCFResult(
            ticker=ticker.upper(),
            current_fcf_usd=current_fcf,
            enterprise_value_usd=enterprise_value,
            forecast_years=years,
            terminal_growth=terminal_growth,
            by_discount_rate={},
            notes=notes,
            error="No discount-rate assumption produced a meaningful growth solution",
        )

    return ReverseDCFResult(
        ticker=ticker.upper(),
        current_fcf_usd=current_fcf,
        enterprise_value_usd=enterprise_value,
        forecast_years=years,
        terminal_growth=terminal_growth,
        by_discount_rate=by_dr,
        notes=notes,
    )


def margin_of_safety_score(implied_growth: float, historical_growth: float) -> float:
    """Reduce growth gap to a 0-10 score.

    Big positive gap (price requires growth far above what the company has been
    delivering) = expensive. Big negative gap (priced for growth below history) =
    margin of safety.
    """
    if implied_growth is None or historical_growth is None:
        return 5.0  # neutral
    gap = implied_growth - historical_growth
    # gap of -10% → score 10 (very cheap), gap of +10% → score 0 (very expensive)
    score = 5.0 - (gap * 50.0)
    return max(0.0, min(10.0, score))


if __name__ == "__main__":
    # Sanity: a company with $1B FCF and $20B EV needs ~10% growth at 10% discount.
    r = reverse_dcf("DEMO", current_fcf=1_000_000_000, enterprise_value=20_000_000_000)
    print(f"DEMO: EV=$20B, FCF=$1B")
    print(f"  implied growth: {r.by_discount_rate}")

    # Apple-ish ballpark check: FCF ~$110B, EV ~$3.5T → much lower implied growth
    r2 = reverse_dcf("AAPL-ish", current_fcf=110_000_000_000, enterprise_value=3_500_000_000_000)
    print(f"\nAAPL-ish: EV=$3.5T, FCF=$110B")
    print(f"  implied growth: {r2.by_discount_rate}")

    # MOS score example
    print(f"\nMOS example: implied 5%, historical 12% (margin of safety):")
    print(f"  score: {margin_of_safety_score(0.05, 0.12):.1f}/10")
    print(f"MOS example: implied 25%, historical 4% (expensive):")
    print(f"  score: {margin_of_safety_score(0.25, 0.04):.1f}/10")
