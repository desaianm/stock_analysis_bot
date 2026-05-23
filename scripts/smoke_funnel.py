"""Smoke: run the funnel-first /undervalued flow against a small universe (150 names)."""

import asyncio

from dotenv import load_dotenv

from stockbot.flows.undervalued import UndervaluedAnalysisFlow, ValueScreeningPreferences
from stockbot.screening.universe import load_universe

load_dotenv()


async def main():
    # Override the universe to a 150-ticker subset for cheap smoke
    universe = load_universe()[:150]
    print(f"Smoke universe: {len(universe)} tickers")

    prefs = ValueScreeningPreferences(
        max_price=500.0,
        min_price=3.0,
        min_volume=200_000,
        max_pe=30.0,
        min_market_cap=100_000_000,
        min_current_ratio=1.0,
        max_debt_equity=3.0,
        price_vs_high=0.5,
    )
    flow = UndervaluedAnalysisFlow(prefs)

    # Monkey-patch the funnel universe to our subset
    import stockbot.flows.undervalued as flow_mod
    original_run = flow_mod.QuantFunnel.run

    def _run_with_subset(self):
        if self.universe is None:
            self.universe = universe
        return original_run(self)

    flow_mod.QuantFunnel.run = _run_with_subset
    try:
        report = await flow.execute_undervalued_analysis()
    finally:
        flow_mod.QuantFunnel.run = original_run

    print("\n=== REPORT TAIL ===")
    print(report[-3000:])


if __name__ == "__main__":
    asyncio.run(main())
