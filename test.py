"""Integration smoke test for stock analysis bot.

Defaults to data-tool smoke checks only (no model calls). Set
RUN_FLOW_TESTS=1 to additionally exercise the full Agno flows (uses API quota).
"""

import asyncio
import os

from dotenv import load_dotenv

from stockbot.flows.recommendations import InvestmentPreferences, Top20StocksFlow
from stockbot.flows.single_stock import SingleStockAnalysisFlow
from stockbot.flows.undervalued import (
    UndervaluedAnalysisFlow,
    ValueScreeningPreferences,
)
from stockbot.tools.data import (
    AnalystRecommendationsTool,
    CompanyInfoTool,
    OptionsChainTool,
    RealTimeQuoteTool,
    StockNewsTool,
    StockPriceDataTool,
)

load_dotenv()

stock_price_data = StockPriceDataTool()
real_time_quote = RealTimeQuoteTool()
options_chain = OptionsChainTool()
analyst_recommendations = AnalystRecommendationsTool()
stock_news = StockNewsTool()
company_info = CompanyInfoTool()


async def run_data_tool_smoke_test(ticker: str) -> None:
    """Exercise core data tools for a benchmark ticker."""
    print("== Data Tools Smoke Test ==")
    print("Stock price (sample)", stock_price_data.run(ticker, "1y")[:5])
    print("Real-time quote", real_time_quote.run(ticker))
    expiry = os.getenv("TEST_OPTION_EXPIRY", "2026-12-18")
    print("Options chain sample", options_chain.run(ticker, expiry)[2])
    print("Analyst recommendations (rows)", len(analyst_recommendations.run(ticker)))
    news = stock_news.run(ticker)
    print(f"Recent news count: {len(news)}; first headline: {news[0].get('title') if news else None}")
    snapshot = company_info.run(ticker)
    print("Company snapshot:", snapshot)


def _default_investment_preferences() -> InvestmentPreferences:
    return InvestmentPreferences(
        strategy="balanced",
        risk_tolerance="moderate",
        time_horizon="5",
        min_market_cap=1.0,
        max_position_size=0.10,
    )


def _default_value_preferences() -> ValueScreeningPreferences:
    return ValueScreeningPreferences(
        max_price=100.0,
        min_price=5.0,
        min_volume=500000,
        max_pe=25.0,
        min_market_cap=300_000_000,
        min_current_ratio=1.5,
        max_debt_equity=2.0,
        price_vs_high=0.4,
    )


async def run_single_stock_flow(ticker: str) -> None:
    print("== Single Stock Analysis Flow ==")
    flow = SingleStockAnalysisFlow(ticker)
    await flow.execute_analysis()


async def run_top20_flow() -> None:
    print("== Top-20 Recommendation Flow ==")
    flow = Top20StocksFlow(_default_investment_preferences())
    await flow.execute_portfolio_construction()


async def run_undervalued_flow() -> None:
    print("== Undervalued Screening Flow ==")
    flow = UndervaluedAnalysisFlow(_default_value_preferences())
    await flow.execute_undervalued_analysis()


async def _run_flow(label: str, coro) -> None:
    try:
        await coro
        print(f"{label} succeeded")
    except Exception as exc:  # noqa: BLE001 - integration logging
        print(f"{label} failed: {exc}")


async def main() -> None:
    ticker = os.getenv("TEST_TICKER", "NVDA")
    await run_data_tool_smoke_test(ticker)

    if os.getenv("RUN_FLOW_TESTS", "0") == "1":
        await _run_flow("Single Stock Flow", run_single_stock_flow(ticker))
        await _run_flow("Top-20 Flow", run_top20_flow())
        await _run_flow("Undervalued Flow", run_undervalued_flow())
    else:
        print("\nSkipping full flow execution (set RUN_FLOW_TESTS=1 to enable).")


if __name__ == "__main__":
    asyncio.run(main())
