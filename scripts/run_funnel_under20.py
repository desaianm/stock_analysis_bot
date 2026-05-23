"""Run /undervalued funnel with max_price=$20 — cheap stock screen.

Uses the full S&P 500 + S&P 600 + TSX Composite universe (~1,325 tickers).
Filters to stocks priced under $20 with reasonable quality gates. Saves results
to the DB so they appear in the watchlist UI's library.
"""

import asyncio

from dotenv import load_dotenv

from stockbot.flows.undervalued import UndervaluedAnalysisFlow, ValueScreeningPreferences

load_dotenv()


async def main():
    # Cheap-stock preset
    prefs = ValueScreeningPreferences(
        max_price=20.0,        # under $20
        min_price=2.0,         # above penny-stock territory
        min_volume=200_000,    # ensure liquidity
        max_pe=30.0,           # avoid the absurd
        min_market_cap=100_000_000,   # small/mid cap floor
        min_current_ratio=1.0,        # solvent
        max_debt_equity=3.0,
        price_vs_high=0.6,
    )
    print("=== UNDERVALUED FUNNEL · UNDER-$20 PRESET ===")
    print(f"Max price: ${prefs.max_price}")
    print(f"Min market cap: ${prefs.min_market_cap:,.0f}")
    print(f"Universe: full (S&P 500 + S&P 600 + TSX Composite)")
    print()

    flow = UndervaluedAnalysisFlow(prefs)
    report = await flow.execute_undervalued_analysis()
    print("\n=== DONE ===")
    print(report[-2000:])


if __name__ == "__main__":
    asyncio.run(main())
