import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from stockbot.flows import (
    EnhancedStockAnalysisFlow,
    Top20StocksFlow,
    InvestmentPreferences,
    UndervaluedAnalysisFlow,
    ValueScreeningPreferences,
)
from stockbot.tools import (
    StockPriceDataTool,
    RealTimeQuoteTool,
    OptionsChainTool,
    AnalystRecommendationsTool,
    StockNewsTool,
    CompanyInfoTool,
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
    expiry = os.getenv("TEST_OPTION_EXPIRY", "2024-12-20")
    print("Options chain sample", options_chain.run(ticker, expiry))
    print("Analyst recommendations", analyst_recommendations.run(ticker))
    news_payload = stock_news.run(ticker)
    print("Recent news result type", type(news_payload))
    company_snapshot = company_info.run(ticker)
    if isinstance(company_snapshot, dict):
        print("Company overview keys", list(company_snapshot.keys())[:5])


async def run_single_stock_flow(ticker: str) -> None:
    print("== Enhanced Stock Analysis Flow ==")
    flow = EnhancedStockAnalysisFlow()
    await flow.kickoff_async(inputs={"ticker": ticker})
    print("Single-stock flow completed")


def _default_investment_preferences() -> InvestmentPreferences:
    return InvestmentPreferences(
        strategy="balanced",
        risk_tolerance="moderate",
        time_horizon="5",
        min_market_cap=1.0,
        max_position_size=0.10,
        preferred_sectors=[],
        excluded_sectors=[],
        esg_focus=False,
        dividend_focus=False,
        international_exposure=False,
    )


def _default_value_preferences() -> ValueScreeningPreferences:
    return ValueScreeningPreferences(
        max_price=100.0,
        min_price=5.0,
        min_volume=500000,
        max_pe=25.0,
        min_market_cap=300000000,
        min_current_ratio=1.5,
        max_debt_equity=2.0,
        price_vs_high=0.4,
    )


async def run_top20_flow() -> None:
    print("== Top-20 Recommendation Flow ==")
    flow = Top20StocksFlow(_default_investment_preferences())
    await flow.execute_portfolio_construction()
    print("Top-20 flow completed")


async def run_undervalued_flow() -> None:
    print("== Undervalued Screening Flow ==")
    flow = UndervaluedAnalysisFlow(_default_value_preferences())
    await flow.execute_undervalued_analysis()
    print("Undervalued flow completed")


async def _run_flow(label: str, coro) -> None:
    try:
        await coro
    except Exception as exc:  # pragma: no cover - integration logging
        print(f"{label} failed: {exc}")
    else:
        print(f"{label} succeeded")


async def main() -> None:
    ticker = os.getenv("TEST_TICKER", "NVDA")
    await run_data_tool_smoke_test(ticker)

    if os.getenv("RUN_CREW_FLOW_TESTS", "0") == "1":
        await _run_flow("Enhanced Stock Analysis Flow", run_single_stock_flow(ticker))
        await _run_flow("Top-20 Recommendation Flow", run_top20_flow())
        await _run_flow("Undervalued Screening Flow", run_undervalued_flow())
    else:
        print("Skipping CrewAI flow execution (set RUN_CREW_FLOW_TESTS=1 to enable).")


if __name__ == "__main__":
    asyncio.run(main())
