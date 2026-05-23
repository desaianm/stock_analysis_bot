"""Single stock analysis flow powered by Agno agents and lightweight tools."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytz
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from pydantic import BaseModel

from stockbot.tools.data import (
    ChartingTool,
    CompanyInfoTool,
    FinancialReportTool,
    OptionsChainTool,
    RealTimeQuoteTool,
    StockNewsTool,
    StockPriceDataTool,
    TavilySearchTool,
    AnalystRecommendationsTool,
)

ny_timezone = pytz.timezone("America/New_York")


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "single_stock" / f"{prompt_name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


class StockAnalysisData(BaseModel):
    ticker: str
    company_name: str = ""
    current_price: float = 0.0
    historical_data: Dict[str, Any] = {}
    sentiment_data: Dict[str, Any] = {}
    financial_metrics: Dict[str, Any] = {}
    technical_analysis: Dict[str, Any] = {}
    analysis_complete: bool = False


class SingleStockAnalysisFlow:
    """Coordinates single stock analysis using Agno agents."""

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.output_dir = Path("outputs/stock_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = "gpt-5.4-mini"

        # Instantiate reusable tool objects
        self._stock_price_tool = StockPriceDataTool()
        self._company_info_tool = CompanyInfoTool()
        self._financial_report_tool = FinancialReportTool()
        self._real_time_quote_tool = RealTimeQuoteTool()
        self._stock_news_tool = StockNewsTool()
        self._charting_tool = ChartingTool()
        self._options_chain_tool = OptionsChainTool()
        self._tavily_tool = TavilySearchTool()
        self._analyst_tool = AnalystRecommendationsTool()
        self._chart_paths: list[str] = []

        shared_tools = [
            self.get_stock_price_history,
            self.get_company_profile,
            self.get_financial_facts,
            self.get_real_time_quote,
            self.get_recent_news,
            self.search_company_info,
            self.get_options_chain_snapshot,
            self.get_analyst_recommendations,
            self.create_metric_chart,
        ]

        shared_model = OpenAIChat(
            id=self.model_id,
            temperature=1,
            max_completion_tokens=10000,
        )

        # Research Agent
        research_instructions = load_prompt("financial_data_researcher_instructions").strip().split('\n')
        self.researcher = Agent(
            name="Financial Data Researcher",
            model=shared_model,
            instructions=research_instructions,
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        # Technical Analyst
        technical_instructions = load_prompt("technical_analyst_instructions").strip().split('\n')
        self.technical_analyst = Agent(
            name="Technical Analysis Specialist",
            model=shared_model,
            instructions=technical_instructions,
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        # Fundamental Analyst
        fundamental_instructions = load_prompt("fundamental_analyst_instructions").strip().split('\n')
        self.fundamental_analyst = Agent(
            name="Fundamental Analyst",
            model=shared_model,
            instructions=fundamental_instructions,
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        # Sentiment Analyst
        sentiment_instructions = load_prompt("market_sentiment_analyst_instructions").strip().split('\n')
        self.sentiment_analyst = Agent(
            name="Market Sentiment Analyst",
            model=shared_model,
            instructions=sentiment_instructions,
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        # Report Writer
        report_instructions = load_prompt("investment_report_writer_instructions").strip().split('\n')
        self.report_writer = Agent(
            name="Investment Report Writer",
            model=shared_model,
            instructions=report_instructions,
            tools=[],  # Report writer doesn't need tools
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

    # -------------------------------------------------------------------------
    # Tool methods (JSON-returning)
    # -------------------------------------------------------------------------
    def _format_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, default=str)

    def _format_error(self, source: str, **metadata: Any) -> str:
        error_message = metadata.pop("error", "unknown error")
        payload = {"source": source, "error": error_message, "metadata": metadata}
        return self._format_json(payload)

    def get_stock_price_history(self, symbol: str, period: str = "5y") -> str:
        """Return historical closing prices for a symbol and period."""
        try:
            prices = self._stock_price_tool.run(symbol, period)
        except Exception as exc:
            return self._format_error(
                "get_stock_price_history", symbol=symbol, period=period, error=str(exc)
            )
        payload = {
            "symbol": symbol.upper(),
            "period": period,
            "close_prices": prices,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_company_profile(self, symbol: str) -> str:
        """Provide key company identifiers and valuation metrics."""
        try:
            info = self._company_info_tool.run(symbol)
        except Exception as exc:
            return self._format_error("get_company_profile", symbol=symbol, error=str(exc))

        if isinstance(info, list) and len(info) >= 6:
            payload = {
                "symbol": symbol.upper(),
                "long_name": info[0],
                "sector": info[1],
                "industry": info[2],
                "market_cap": info[3],
                "forward_pe": info[4],
                "dividend_yield": info[5],
                "retrieved_at": datetime.now(ny_timezone).isoformat(),
            }
        else:
            payload = {"symbol": symbol.upper(), "raw_data": info}

        return self._format_json(payload)

    def get_financial_facts(self, symbol: str) -> str:
        """Return QuickFS-style financial statement data for a symbol."""
        try:
            data = self._financial_report_tool.run(symbol)
        except Exception as exc:
            return self._format_error("get_financial_facts", symbol=symbol, error=str(exc))

        metrics_to_track = [
            "revenue",
            "net_income",
            "operating_cash_flow",
            "free_cash_flow",
            "total_debt",
            "shareholders_equity",
        ]
        metric_summaries: Dict[str, Dict[str, Any]] = {}

        for metric in metrics_to_track:
            metric_values = data.get(metric) if isinstance(data, dict) else None
            if isinstance(metric_values, list) and metric_values:
                last_values = metric_values[-5:]
                chart_path = None
                try:
                    chart_result = self._charting_tool.run(
                        f"{symbol.upper()} {metric.replace('_', ' ').title()}",
                        last_values,
                    )
                    chart_path = getattr(chart_result, "file_path", None)
                    if chart_path:
                        self._register_chart_path(chart_path)
                except Exception:
                    chart_path = None

                metric_summaries[metric] = {
                    "last_values": last_values,
                    "chart_path": chart_path,
                }

        payload = {
            "symbol": symbol.upper(),
            "condensed_metrics": metric_summaries,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_real_time_quote(self, symbol: str) -> str:
        """Fetch intraday price, volume, and market cap."""
        try:
            price, volume, market_cap = self._real_time_quote_tool.run(symbol)
        except Exception as exc:
            return self._format_error("get_real_time_quote", symbol=symbol, error=str(exc))

        payload = {
            "symbol": symbol.upper(),
            "price": price,
            "volume": volume,
            "market_cap": market_cap,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_recent_news(self, symbol: str) -> str:
        """Return the five most recent Yahoo Finance headlines for the symbol."""
        try:
            headlines = self._stock_news_tool.run(symbol)
        except Exception as exc:
            return self._format_error("get_recent_news", symbol=symbol, error=str(exc))

        payload = {
            "symbol": symbol.upper(),
            "headlines": headlines,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def search_company_info(self, query: str) -> str:
        """Use Tavily to search for company information and market developments."""
        try:
            results = self._tavily_tool.run(query)
        except Exception as exc:
            return self._format_error("search_company_info", query=query, error=str(exc))

        payload = {
            "query": query,
            "results": results,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_options_chain_snapshot(self, symbol: str, expiration_date: str) -> str:
        """Return the nearest available options chain for an expiration date."""
        try:
            calls, puts, actual_expiration = self._options_chain_tool.run(
                symbol, expiration_date
            )
        except Exception as exc:
            return self._format_error(
                "get_options_chain_snapshot",
                symbol=symbol,
                expiration_date=expiration_date,
                error=str(exc),
            )

        payload = {
            "symbol": symbol.upper(),
            "requested_expiration": expiration_date,
            "actual_expiration": actual_expiration,
            "calls": calls,
            "puts": puts,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Return analyst recommendations and price targets."""
        try:
            recommendations = self._analyst_tool.run(symbol)
        except Exception as exc:
            return self._format_error("get_analyst_recommendations", symbol=symbol, error=str(exc))

        payload = {
            "symbol": symbol.upper(),
            "recommendations": recommendations,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def create_metric_chart(self, metric_name: str, data: List[float]) -> str:
        """Persist a PNG chart for the given metric data."""
        try:
            chart_output = self._charting_tool.run(metric_name, data)
        except Exception as exc:
            return self._format_error(
                "create_metric_chart", metric_name=metric_name, error=str(exc)
            )

        payload = {
            "metric_name": metric_name,
            "file_path": chart_output.file_path,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        if chart_output.file_path:
            self._register_chart_path(chart_output.file_path)
        return self._format_json(payload)

    def _register_chart_path(self, path: str) -> None:
        """Track generated chart images for downstream agents."""
        if path and path not in self._chart_paths:
            self._chart_paths.append(path)

    def _build_image_payload(self, limit: int = 10) -> List[Image]:
        """Convert stored chart paths into Agno Image attachments."""
        images: List[Image] = []
        for path in self._chart_paths[-limit:]:
            try:
                if os.path.exists(path):
                    images.append(Image(path=path))
            except Exception:
                continue
        return images

    # -------------------------------------------------------------------------
    # Analysis workflow
    # -------------------------------------------------------------------------
    async def execute_analysis(self) -> str:
        """Run the full stock analysis pipeline."""
        print(f"\n=== Analyzing {self.ticker} ===")

        # Step 1: Research and gather data
        print("\n1. Gathering financial data...")
        research_prompt = load_prompt("gather_basic_data_task").format(ticker=self.ticker)
        research_result = await self.researcher.arun(research_prompt, images=self._build_image_payload())
        research_content = self._extract_content(research_result)

        # Step 2: Technical analysis
        print("\n2. Performing technical analysis...")
        technical_prompt = load_prompt("technical_analysis_task").format(
            ticker=self.ticker,
            historical_data=research_content[:2000]  # Truncate for context
        )
        technical_result = await self.technical_analyst.arun(
            technical_prompt, images=self._build_image_payload()
        )
        technical_content = self._extract_content(technical_result)

        # Step 3: Fundamental analysis
        print("\n3. Analyzing fundamentals...")
        fundamental_prompt = load_prompt("fundamental_analysis_task").format(ticker=self.ticker)
        fundamental_result = await self.fundamental_analyst.arun(
            fundamental_prompt, images=self._build_image_payload()
        )
        fundamental_content = self._extract_content(fundamental_result)

        # Step 4: Sentiment analysis
        print("\n4. Analyzing market sentiment...")
        sentiment_prompt = load_prompt("sentiment_analysis_task").format(ticker=self.ticker)
        sentiment_result = await self.sentiment_analyst.arun(
            sentiment_prompt, images=self._build_image_payload()
        )
        sentiment_content = self._extract_content(sentiment_result)

        # Step 5: Compile final report
        print("\n5. Compiling final report...")
        report_prompt = load_prompt("compile_report_task").format(
            ticker=self.ticker,
            technical_analysis=technical_content,
            fundamental_analysis=fundamental_content,
            sentiment_analysis=sentiment_content,
        )
        final_report = await self.report_writer.arun(report_prompt, images=self._build_image_payload())
        final_content = self._extract_content(final_report)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d")
        report_file = f"stock_analysis_{self.ticker}_{timestamp}.md"

        sanitized_content = self._sanitize_output(final_content)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# Stock Analysis Report: {self.ticker}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(sanitized_content)

        print(f"\n✓ Analysis complete! Report saved to: {report_file}")
        return final_content

    def _extract_content(self, response: Any) -> str:
        """Normalize Agno run responses to plain strings."""
        if response is None:
            return ""
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(response, str):
            return response
        return str(content or response)

    def _sanitize_output(self, content: str) -> str:
        """Remove markdown fences or whitespace the agent may prepend."""
        if not content:
            return ""
        cleaned = content.strip()
        if cleaned.startswith("```markdown"):
            cleaned = cleaned[len("```markdown") :].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```") :].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned


async def main():
    """Execute single stock analysis."""
    ticker = input("Enter stock ticker: ").strip().upper() or "AAPL"
    flow = SingleStockAnalysisFlow(ticker)
    await flow.execute_analysis()


if __name__ == "__main__":
    asyncio.run(main())
