"""Top 20 stock portfolio construction flow powered by Agno agents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.utils.log import configure_agno_logging
from pydantic import BaseModel, Field

from stockbot.database.manager import StockDatabaseManager
from stockbot.tools.data import (
    AnalystRecommendationsTool,
    ChartingTool,
    CompanyInfoTool,
    FinancialReportTool,
    OptionsChainTool,
    RealTimeQuoteTool,
    StockNewsTool,
    StockPriceDataTool,
    TavilySearchTool,
    WebSearchTool,
)

ny_timezone = pytz.timezone("America/New_York")


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts/top20 directory."""
    prompt_path = (
        Path(__file__).parent.parent.parent
        / "prompts"
        / "top20"
        / f"{prompt_name}.txt"
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


class InvestmentPreferences(BaseModel):
    strategy: str = Field(..., description="growth, value, or balanced")
    risk_tolerance: str = Field(..., description="conservative, moderate, or aggressive")
    time_horizon: str = Field(..., description="years")
    min_market_cap: float = Field(..., description="billions USD")
    max_position_size: float = Field(..., description="fraction (e.g. 0.10)")
    preferred_sectors: List[str] = Field(default_factory=list)
    excluded_sectors: List[str] = Field(default_factory=list)
    esg_focus: bool = False
    dividend_focus: bool = False
    international_exposure: bool = False


class Top20StocksFlow:
    """Coordinates top-20 portfolio construction using Agno agents."""

    SCREENING_TOKEN_LIMIT = 60_000

    def __init__(self, preferences: InvestmentPreferences):
        self.preferences = preferences
        self.output_dir = Path("outputs/top20")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.reasoning_model_id = "gpt-5.4-mini"
        self.summary_model_id = "gpt-5.4-nano"

        self.db = StockDatabaseManager()
        self.current_run_id: Optional[int] = None
        self.log_file_path: Optional[Path] = None
        self.logger: Optional[logging.Logger] = None

        # Tool wrappers (same pattern as undervalued.py)
        self._stock_price_tool = StockPriceDataTool()
        self._company_info_tool = CompanyInfoTool()
        self._financial_report_tool = FinancialReportTool()
        self._real_time_quote_tool = RealTimeQuoteTool()
        self._stock_news_tool = StockNewsTool()
        self._charting_tool = ChartingTool()
        self._options_chain_tool = OptionsChainTool()
        self._analyst_tool = AnalystRecommendationsTool()
        self._tavily_tool = TavilySearchTool()
        self._web_search_tool = WebSearchTool()
        self._chart_paths: list[str] = []

        shared_tools = [
            self.get_stock_price_history,
            self.get_company_profile,
            self.get_financial_facts,
            self.get_real_time_quote,
            self.get_recent_news,
            self.get_analyst_recommendations,
            self.search_market_events,
            self.search_global_research,
            self.create_metric_chart,
        ]

        shared_model = OpenAIChat(
            id=self.reasoning_model_id,
            temperature=1,
            max_completion_tokens=10000,
        )

        self.screening_agent = Agent(
            name="Quantitative Screening Specialist",
            model=shared_model,
            instructions=load_prompt("screening_instructions").strip().split("\n"),
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        self.tournament_agent = Agent(
            name="Tournament Analysis Director",
            model=shared_model,
            instructions=load_prompt("tournament_instructions").strip().split("\n"),
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        self.portfolio_agent = Agent(
            name="Portfolio Construction Expert",
            model=shared_model,
            instructions=load_prompt("portfolio_instructions").strip().split("\n"),
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    def _configure_agno_debug_logging(self) -> Path:
        """Configure Agno to write debug logs to a timestamped file."""
        timestamp = datetime.now(ny_timezone).strftime("%H%M%S_%Y%m%d")
        log_file_path = self.logs_dir / f"top20_{timestamp}.log"
        logger_name = f"agno-top20-{timestamp}"

        custom_logger = logging.getLogger(logger_name)
        custom_logger.handlers.clear()
        custom_logger.setLevel(logging.DEBUG)
        custom_logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        custom_logger.addHandler(file_handler)
        custom_logger.addHandler(stream_handler)

        configure_agno_logging(
            custom_default_logger=custom_logger,
            custom_agent_logger=custom_logger,
            custom_team_logger=custom_logger,
            custom_workflow_logger=custom_logger,
        )
        self.logger = custom_logger
        self.log_file_path = log_file_path
        return log_file_path

    # ---------------------------------------------------------------------
    # Tool wrappers (JSON-returning, matches undervalued.py pattern)
    # ---------------------------------------------------------------------
    def _format_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, default=str)

    def _format_error(self, source: str, **metadata: Any) -> str:
        err = metadata.pop("error", "unknown error")
        return self._format_json({"source": source, "error": err, "metadata": metadata})

    def get_stock_price_history(self, symbol: str, period: str = "1y") -> str:
        """Return historical closing prices for a symbol and period."""
        try:
            prices = self._stock_price_tool.run(symbol, period)
        except Exception as exc:
            return self._format_error(
                "get_stock_price_history", symbol=symbol, period=period, error=str(exc)
            )
        recent = prices[-60:] if isinstance(prices, list) else []
        payload = {
            "symbol": symbol.upper(),
            "period": period,
            "close_prices": recent,
            "total_points": len(prices) if isinstance(prices, list) else 0,
            "period_high": max(prices) if isinstance(prices, list) and prices else None,
            "period_low": min(prices) if isinstance(prices, list) and prices else None,
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
        """Return condensed financial statement data."""
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
        condensed: Dict[str, List[float]] = {}
        for metric in metrics_to_track:
            values = data.get(metric) if isinstance(data, dict) else None
            if isinstance(values, list) and values:
                condensed[metric] = values[-3:]

        payload = {
            "symbol": symbol.upper(),
            "source": data.get("source") if isinstance(data, dict) else None,
            "condensed_metrics": condensed,
            "screening_metrics": {
                "current_price": data.get("current_price") if isinstance(data, dict) else None,
                "market_cap": data.get("market_cap") if isinstance(data, dict) else None,
                "price_to_earnings": data.get("price_to_earnings") if isinstance(data, dict) else None,
                "forward_pe": data.get("forward_pe") if isinstance(data, dict) else None,
                "price_to_book": data.get("price_to_book") if isinstance(data, dict) else None,
                "current_ratio": data.get("current_ratio") if isinstance(data, dict) else None,
                "debt_to_equity": data.get("debt_to_equity") if isinstance(data, dict) else None,
                "volume": data.get("volume") if isinstance(data, dict) else None,
            },
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_real_time_quote(self, symbol: str) -> str:
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

    def get_analyst_recommendations(self, symbol: str) -> str:
        try:
            recs = self._analyst_tool.run(symbol)
        except Exception as exc:
            return self._format_error("get_analyst_recommendations", symbol=symbol, error=str(exc))
        payload = {
            "symbol": symbol.upper(),
            "recommendations": recs[:10] if isinstance(recs, list) else recs,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def search_market_events(self, query: str) -> str:
        try:
            results = self._web_search_tool.run(query)
        except Exception as exc:
            return self._format_error("search_market_events", query=query, error=str(exc))
        payload = {
            "query": query,
            "results": results[:8] if isinstance(results, list) else results,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def search_global_research(self, query: str) -> str:
        try:
            results = self._tavily_tool.run(query)
        except Exception as exc:
            return self._format_error("search_global_research", query=query, error=str(exc))
        payload = {
            "query": query,
            "results": results[:8] if isinstance(results, list) else results,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def create_metric_chart(self, metric_name: str, data: List[float]) -> str:
        try:
            chart = self._charting_tool.run(metric_name, data)
        except Exception as exc:
            return self._format_error("create_metric_chart", metric_name=metric_name, error=str(exc))
        payload = {
            "metric_name": metric_name,
            "file_path": chart.file_path,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        if chart.file_path:
            self._register_chart_path(chart.file_path)
        return self._format_json(payload)

    def _register_chart_path(self, path: str) -> None:
        if path and path not in self._chart_paths:
            self._chart_paths.append(path)

    def _build_image_payload(self, limit: int = 8) -> List[Image]:
        images: List[Image] = []
        for path in self._chart_paths[-limit:]:
            try:
                if os.path.exists(path):
                    images.append(Image(path=path))
            except Exception:
                continue
        return images

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _extract_content(self, response: Any) -> str:
        if response is None:
            return ""
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(response, str):
            return response
        return str(content or response)

    def _sanitize_output(self, content: str) -> str:
        if not content:
            return ""
        cleaned = content.strip()
        if cleaned.startswith("```markdown"):
            cleaned = cleaned[len("```markdown"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _truncate_for_context(self, text: str, char_limit: int) -> str:
        if not text or len(text) <= char_limit:
            return text
        return text[:char_limit] + f"\n\n[Truncated to {char_limit:,} chars]"

    async def save_phase_output(self, phase: str, content: str, description: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{phase}_{timestamp}.md"
        cleaned = self._sanitize_output(content)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {description}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(cleaned)

    # ---------------------------------------------------------------------
    # Prompt builders
    # ---------------------------------------------------------------------
    def _build_screening_prompt(self) -> str:
        prefs = self.preferences
        return load_prompt("screening_prompt").format(
            timestamp=datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S"),
            strategy=prefs.strategy,
            risk_tolerance=prefs.risk_tolerance,
            time_horizon=prefs.time_horizon,
            min_market_cap=f"{prefs.min_market_cap:.1f}",
            max_position_size_pct=f"{prefs.max_position_size * 100:.1f}",
            preferred_sectors=", ".join(prefs.preferred_sectors) or "any",
            excluded_sectors=", ".join(prefs.excluded_sectors) or "none",
            esg_focus="yes" if prefs.esg_focus else "no",
            dividend_focus="yes" if prefs.dividend_focus else "no",
            international_exposure="yes" if prefs.international_exposure else "no",
        )

    def _build_tournament_prompt(self, candidates: str) -> str:
        return load_prompt("tournament_prompt").format(
            timestamp=datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S"),
            candidates=self._truncate_for_context(candidates, self.SCREENING_TOKEN_LIMIT * 4),
        )

    def _build_portfolio_prompt(self, ranked: str) -> str:
        prefs = self.preferences
        return load_prompt("portfolio_prompt").format(
            timestamp=datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S"),
            ranked_top_20=self._truncate_for_context(ranked, self.SCREENING_TOKEN_LIMIT * 4),
            max_position_size_pct=f"{prefs.max_position_size * 100:.1f}",
            risk_tolerance=prefs.risk_tolerance,
        )

    def _compose_final_report(
        self, screening: str, tournament: str, portfolio: str
    ) -> str:
        timestamp = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        return f"""# Top 20 Portfolio Recommendation Report
Generated: {timestamp}

## Strategy
- Strategy: {self.preferences.strategy}
- Risk tolerance: {self.preferences.risk_tolerance}
- Time horizon: {self.preferences.time_horizon} years
- Max position size: {self.preferences.max_position_size * 100:.1f}%

## Phase 1: Screening
{screening}

## Phase 2: Tournament Ranking
{tournament}

## Phase 3: Portfolio Construction
{portfolio}
"""

    # ---------------------------------------------------------------------
    # Main workflow
    # ---------------------------------------------------------------------
    async def execute_portfolio_construction(self) -> str:
        """Run screening → tournament → portfolio construction."""
        log_path = self._configure_agno_debug_logging()
        print(f"Agno debug logs: {log_path}")

        self.current_run_id = self.db.create_analysis_run(
            run_type="top20",
            preferences=self.preferences.model_dump(),
        )
        print(f"\nStarted top20 run #{self.current_run_id}")

        try:
            print("\nPhase 1: Screening...")
            screening_raw = await self.screening_agent.arun(
                self._build_screening_prompt(),
                images=self._build_image_payload(),
            )
            screening_text = self._sanitize_output(self._extract_content(screening_raw))
            await self.save_phase_output(
                "screening", screening_text, "Initial candidate universe"
            )

            print("\nPhase 2: Tournament ranking...")
            tournament_raw = await self.tournament_agent.arun(
                self._build_tournament_prompt(screening_text),
                images=self._build_image_payload(),
            )
            tournament_text = self._sanitize_output(self._extract_content(tournament_raw))
            await self.save_phase_output(
                "tournament", tournament_text, "Head-to-head ranking"
            )

            print("\nPhase 3: Portfolio construction...")
            portfolio_raw = await self.portfolio_agent.arun(
                self._build_portfolio_prompt(tournament_text),
                images=self._build_image_payload(),
            )
            portfolio_text = self._sanitize_output(self._extract_content(portfolio_raw))
            await self.save_phase_output(
                "portfolio", portfolio_text, "Final allocated portfolio"
            )

            final_report = self._compose_final_report(
                screening_text, tournament_text, portfolio_text
            )
            await self.save_phase_output(
                "final_report", final_report, "Top 20 portfolio recommendation"
            )

            self.db.complete_analysis_run(
                run_id=self.current_run_id,
                total_candidates=20,
                final_selections=20,
                status="completed",
            )
            print(f"\nTop20 run #{self.current_run_id} completed.")
            return final_report

        except Exception:
            if self.current_run_id:
                self.db.complete_analysis_run(
                    run_id=self.current_run_id, status="failed"
                )
            raise


async def main():
    """Local entry point for testing the top20 flow."""
    preferences = InvestmentPreferences(
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
    flow = Top20StocksFlow(preferences)
    await flow.execute_portfolio_construction()


if __name__ == "__main__":
    asyncio.run(main())
