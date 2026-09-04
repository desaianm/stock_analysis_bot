"""Undervalued stock analysis flow powered by Agno agents and lightweight tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pytz
import requests
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from agno.utils.log import configure_agno_logging
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

from stockbot.audit import clear_state, write_state
from stockbot.database.manager import StockDatabaseManager
from stockbot.database.performance_manager import PerformanceTrackingManager
from stockbot.screening.funnel import FunnelCandidate, QuantFunnel
from stockbot.screening.numeric_screen import ScreeningGates
from stockbot.screening.universe import Ticker, load_universe
from stockbot.tickers import normalize_ticker
from stockbot.tools.data import (
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
from stockbot.tools.reddit import RedditClient

ny_timezone = pytz.timezone("America/New_York")


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = (
        Path(__file__).parent.parent.parent
        / "prompts"
        / "undervalued"
        / f"{prompt_name}.txt"
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


class UndervaluedMetrics(BaseModel):
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    price_to_book: float
    price_to_earnings: float
    forward_pe: float
    peg_ratio: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    insider_buying: float
    institutional_ownership: float
    short_interest: float
    rsi_14: float
    revenue_growth: float
    earnings_growth: float
    cash_flow_growth: float
    analyst_targets: List[float]
    potential_catalysts: List[str]


class ValueScreeningPreferences(BaseModel):
    max_price: float = Field(default=100.0, description="Maximum stock price")
    min_price: float = Field(default=5.0, description="Minimum stock price")
    min_volume: float = Field(
        default=500000, description="Minimum daily trading volume"
    )
    max_pe: float = Field(default=25.0, description="Maximum P/E ratio")
    min_market_cap: float = Field(
        default=300000000, description="Minimum market cap ($300M)"
    )
    min_current_ratio: float = Field(default=1.5, description="Minimum current ratio")
    max_debt_equity: float = Field(default=2.0, description="Maximum debt/equity ratio")
    price_vs_high: float = Field(
        default=0.4, description="Maximum decline from 52-week high (40%)"
    )


class DeepDiveReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_at: datetime
    candidates_reviewed: int = Field(ge=0)
    candidates_accepted: int = Field(ge=0)


class DeepDiveStock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    verdict: Literal["accept", "reject"]
    confidence_score: float = Field(ge=0, le=10)
    thesis: str = Field(min_length=1)
    key_risks: List[str] = Field(min_length=1)
    primary_catalyst: str = Field(min_length=1)
    entry_strategy: str = Field(min_length=1)
    stop_loss_pct: float = Field(ge=0, le=1)
    position_size_pct: float = Field(ge=0, le=1)
    rejection_reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_rejection_reason(self) -> "DeepDiveStock":
        if self.verdict == "accept" and self.position_size_pct <= 0:
            raise ValueError("position_size_pct must be greater than 0 for accepted stocks")
        if self.verdict == "reject" and self.position_size_pct != 0:
            raise ValueError("position_size_pct must equal 0 for rejected stocks")
        if self.verdict == "reject" and not self.rejection_reason:
            raise ValueError("rejection_reason is required for rejected stocks")
        if self.verdict == "accept" and self.rejection_reason is not None:
            raise ValueError("rejection_reason must be omitted for accepted stocks")
        return self


class DeepDiveOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortlist_review: DeepDiveReview
    stocks: List[DeepDiveStock]


class BottleneckTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    constrained_input: str = Field(min_length=1)
    demand_driver: str = Field(min_length=1)
    change_signal: str = Field(min_length=1)
    supply_response_lag: str = Field(min_length=1)
    time_horizon_months: int = Field(ge=1, le=36)
    evidence: List[str] = Field(min_length=2)
    source_urls: List[str] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_sources(self) -> "BottleneckTheme":
        if any(not url.startswith(("https://", "http://")) for url in self.source_urls):
            raise ValueError("source_urls must contain HTTP(S) URLs")
        if len(set(self.source_urls)) < 2:
            raise ValueError("source_urls must contain two independent URLs")
        return self


class BottleneckCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    theme: str = Field(min_length=1)
    role: str = Field(min_length=1)
    confidence_score: float = Field(ge=0, le=10)
    earnings_transmission: str = Field(min_length=1)
    evidence: List[str] = Field(min_length=2)
    source_urls: List[str] = Field(min_length=2)
    already_repriced_risk: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "BottleneckCandidate":
        if any(not url.startswith(("https://", "http://")) for url in self.source_urls):
            raise ValueError("source_urls must contain HTTP(S) URLs")
        if len(set(self.source_urls)) < 2:
            raise ValueError("source_urls must contain two independent URLs")
        return self


class BottleneckResearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    researched_at: datetime
    market_regime_summary: str = Field(min_length=1)
    themes: List[BottleneckTheme] = Field(min_length=1, max_length=8)
    candidates: List[BottleneckCandidate] = Field(max_length=15)

    @model_validator(mode="after")
    def validate_candidate_themes(self) -> "BottleneckResearchOutput":
        theme_names = [theme.name for theme in self.themes]
        themes = set(theme_names)
        if len(themes) != len(theme_names):
            raise ValueError("theme names must be unique")
        unknown = sorted({candidate.theme for candidate in self.candidates} - themes)
        if unknown:
            raise ValueError(f"candidates reference unknown themes: {unknown}")
        return self


class UndervaluedAnalysisFlow:
    """Coordinates the undervalued stock analysis using an Agno agent."""

    MODEL_INPUT_TOKEN_LIMIT = 300_000
    MAX_SCREENING_PROMPT_TOKENS = 180_000
    MAX_REDDIT_SUMMARY_TOKENS = 20_000
    MAX_LEARNING_INSIGHTS_TOKENS = 12_000
    SCREENING_TOKEN_LIMIT = 80_000

    def __init__(self, preferences: ValueScreeningPreferences):
        self.preferences = preferences
        self.output_dir = Path("outputs/undervalued_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path: Optional[Path] = None
        self.logger: Optional[logging.Logger] = None
        self.reasoning_model_id = "gpt-5.4-mini"
        self.summary_model_id = "gpt-5.4-nano"
        self.extraction_model_id = "gpt-4.1-nano"
        self.model_id = self.reasoning_model_id

        # Initialize database
        self.db = StockDatabaseManager()
        self.perf_db = PerformanceTrackingManager()  # For learning insights
        self.current_run_id: Optional[int] = None

        # Instantiate reusable tool objects
        self._stock_price_tool = StockPriceDataTool()
        self._company_info_tool = CompanyInfoTool()
        self._financial_report_tool = FinancialReportTool()
        self._real_time_quote_tool = RealTimeQuoteTool()
        self._stock_news_tool = StockNewsTool()
        self._charting_tool = ChartingTool()
        self._options_chain_tool = OptionsChainTool()
        self._tavily_tool = TavilySearchTool()
        self.web_search_tool = WebSearchTool()
        self._token_encoder = self._load_token_encoder()
        self._chart_paths: list[str] = []
        self._reddit_client = RedditClient.from_env()
        self._symbol_resolution_cache: Dict[str, str] = {}
        self._candidate_metrics_cache: Dict[str, Dict[str, Any]] = {}

        shared_tools = [
            self.get_stock_price_history,
            self.get_company_profile,
            self.get_financial_facts,
            self.get_real_time_quote,
            self.get_recent_news,
            self.search_market_events,
            self.search_global_research,
            self.get_options_chain_snapshot,
            self.create_metric_chart,
            self.summarize_context_tool,
            # Database tools for historical context
            self.search_past_finds,
            self.get_ticker_history,
            self.get_similar_stocks,
            self.get_recent_discoveries,
        ]
        shared_model = OpenAIChat(
            id=self.reasoning_model_id,
            temperature=1,
            max_completion_tokens=10000,
        )

        screening_instructions = (
            load_prompt("screening_agent_instructions").strip().split("\n")
        )
        self.screening_agent = Agent(
            name="Value Screening Specialist",
            model=shared_model,
            instructions=screening_instructions,
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        turnaround_instructions = (
            load_prompt("turnaround_agent_instructions").strip().split("\n")
        )
        self.turnaround_agent = Agent(
            name="Turnaround Potential Analyst",
            model=shared_model,
            instructions=turnaround_instructions,
            tools=shared_tools,
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        reddit_instructions = (
            load_prompt("reddit_sentiment_agent_instructions").strip().split("\n")
        )
        self.reddit_sentiment_agent = Agent(
            name="BayStreet Reddit Scout",
            model=OpenAIChat(
                id=self.summary_model_id, temperature=1, max_completion_tokens=5000
            ),
            instructions=reddit_instructions,
            tools=[self.reddit_sentiment_scan],
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        bottleneck_instructions = (
            load_prompt("bottleneck_research_instructions").strip().split("\n")
        )
        self.bottleneck_research_agent = Agent(
            name="Cross-Sector Bottleneck Researcher",
            model=OpenAIChat(
                id=self.reasoning_model_id,
                temperature=1,
                max_completion_tokens=8000,
            ),
            instructions=bottleneck_instructions,
            tools=[self.search_global_research, self.search_market_events],
            markdown=False,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

        # Funnel deep-dive agent: takes the pre-screened shortlist and emits
        # strict-JSON thesis per candidate. Replaces the old Reddit-driven
        # screening + turnaround agents as the discovery driver.
        deep_dive_instructions = (
            load_prompt("funnel_deep_dive_instructions").strip().split("\n")
        )
        self.deep_dive_agent = Agent(
            name="Funnel Deep-Dive Analyst",
            model=OpenAIChat(
                id=self.reasoning_model_id,
                temperature=1,
                max_completion_tokens=8000,
            ),
            instructions=deep_dive_instructions,
            tools=[
                # Slim toolset — agent should trust funnel metrics, only verify catalysts
                self.get_recent_news,
                self.search_market_events,
                self.get_ticker_history,
                self.get_similar_stocks,
            ],
            markdown=False,  # JSON output, no markdown formatting
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

    def _configure_agno_debug_logging(self) -> Path:
        """Configure Agno to write debug logs to a timestamped file and stdout."""
        timestamp = datetime.now(ny_timezone).strftime("%H%M%S_%Y%m%d")
        log_file_path = self.logs_dir / f"{timestamp}.log"
        logger_name = f"agno-undervalued-{timestamp}"

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
        custom_logger.info(
            "Agno debug logging configured for undervalued analysis: %s",
            log_file_path,
        )
        self.log_file_path = log_file_path
        return log_file_path

    def _load_token_encoder(self):
        """Load the tokenizer for the configured OpenAI model."""
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model_id)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._token_encoder is None:
            return len(text) // 4
        return len(self._token_encoder.encode(text))

    # -------------------------------------------------------------------------
    # Lightweight wrappers around the legacy tool classes
    # -------------------------------------------------------------------------
    def _format_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, default=str)

    def _format_error(self, source: str, **metadata: Any) -> str:
        error_message = metadata.pop("error", "unknown error")
        payload = {"source": source, "error": error_message, "metadata": metadata}
        return self._format_json(payload)

    def _resolve_yf_symbol(self, symbol: str) -> str:
        """Return the yfinance-resolvable ticker for a symbol.

        Canadian tickers are often found on Reddit without the Yahoo suffix.
        Score candidates against actual Yahoo quote/history data and use a few
        known ambiguous aliases observed in live runs.
        """
        import yfinance as yf

        raw_symbol = (symbol or "").strip().upper()
        if not raw_symbol:
            return raw_symbol

        if raw_symbol in self._symbol_resolution_cache:
            return self._symbol_resolution_cache[raw_symbol]

        aliases = {
            "ARC": "ARX.TO",
            "ARX": "ARX.TO",
            "QIMC": "QIMC.CN",
            "TNZ": "TNZ.TO",
            "HHE": "HHE.CN",
            "CVVY": "CVVY.TO",
        }
        if raw_symbol in aliases:
            self._symbol_resolution_cache[raw_symbol] = aliases[raw_symbol]
            return aliases[raw_symbol]

        if "." in raw_symbol or ":" in raw_symbol:
            self._symbol_resolution_cache[raw_symbol] = raw_symbol
            return raw_symbol

        candidates = [
            f"{raw_symbol}.TO",
            f"{raw_symbol}.V",
            f"{raw_symbol}.CN",
            f"{raw_symbol}.NE",
            raw_symbol,
        ]
        best_candidate = raw_symbol
        best_score = -1.0
        for candidate in candidates:
            try:
                ticker = yf.Ticker(candidate)
                hist = ticker.history(period="1mo")
                if not hist.empty:
                    info = ticker.info or {}
                    market_cap = info.get("marketCap") or 0
                    volume = info.get("volume") or info.get("averageVolume") or 0
                    score = len(hist) + min(float(volume or 0) / 1_000_000, 10)
                    if market_cap:
                        score += 5
                    if candidate.endswith((".TO", ".V", ".CN", ".NE")):
                        score += 2
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
            except Exception:
                continue

        if best_candidate != raw_symbol:
            logging.getLogger(__name__).info(
                "yfinance ticker resolved: %s -> %s", raw_symbol, best_candidate
            )
        self._symbol_resolution_cache[raw_symbol] = best_candidate
        return best_candidate

    def _compact_search_results(
        self, results: Any, max_items: int = 5, max_snippet_chars: int = 280
    ) -> Any:
        """Reduce search payloads before they are fed back into the model."""
        raw_items = None
        if hasattr(results, "results"):
            raw_items = getattr(results, "results")
        elif isinstance(results, dict):
            raw_items = (
                results.get("results") or results.get("data") or results.get("items")
            )
        elif isinstance(results, list):
            raw_items = results

        if raw_items is None:
            return self._truncate_for_context(str(results), 2_000)

        compact_items: List[Dict[str, Any]] = []
        for item in list(raw_items)[:max_items]:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name")
                url = item.get("url") or item.get("link")
                snippet = (
                    item.get("text")
                    or item.get("content")
                    or item.get("raw_content")
                    or item.get("snippet")
                )
                published = item.get("published_date") or item.get("published")
            else:
                title = getattr(item, "title", None)
                url = getattr(item, "url", None)
                snippet = (
                    getattr(item, "text", None)
                    or getattr(item, "content", None)
                    or getattr(item, "raw_content", None)
                    or getattr(item, "snippet", None)
                )
                published = getattr(item, "published_date", None) or getattr(
                    item, "published", None
                )

            compact_items.append(
                {
                    "title": title,
                    "url": url,
                    "published": published,
                    "snippet": (snippet or "")[:max_snippet_chars],
                }
            )

        return compact_items

    def get_stock_price_history(self, symbol: str, period: str = "1y") -> str:
        """Return historical closing prices for a symbol and period."""
        symbol = self._resolve_yf_symbol(symbol)
        try:
            prices = self._stock_price_tool.run(symbol, period)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error(
                "get_stock_price_history", symbol=symbol, period=period, error=str(exc)
            )
        recent_prices = prices[-60:] if isinstance(prices, list) else []
        payload = {
            "symbol": symbol.upper(),
            "period": period,
            "close_prices": recent_prices,
            "total_points": len(prices) if isinstance(prices, list) else 0,
            "first_close": prices[0] if isinstance(prices, list) and prices else None,
            "last_close": prices[-1] if isinstance(prices, list) and prices else None,
            "period_high": max(prices) if isinstance(prices, list) and prices else None,
            "period_low": min(prices) if isinstance(prices, list) and prices else None,
            "note": "close_prices contains only the most recent 60 closes to keep tool output compact.",
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_company_profile(self, symbol: str) -> str:
        """Provide key company identifiers and valuation metrics."""
        symbol = self._resolve_yf_symbol(symbol)
        try:
            info = self._company_info_tool.run(symbol)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error(
                "get_company_profile", symbol=symbol, error=str(exc)
            )

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
        """Return financial statement data and hard-screen ratios for a symbol."""
        symbol = self._resolve_yf_symbol(symbol)
        try:
            data = self._financial_report_tool.run(symbol)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error(
                "get_financial_facts", symbol=symbol, error=str(exc)
            )

        metrics_to_track = [
            "revenue",
            "net_income",
            "operating_cash_flow",
            "free_cash_flow",
            "total_debt",
            "shareholders_equity",
            "current_assets",
            "current_liabilities",
        ]
        metric_summaries: Dict[str, Dict[str, Any]] = {}

        for metric in metrics_to_track:
            metric_values = data.get(metric) if isinstance(data, dict) else None
            if isinstance(metric_values, list) and metric_values:
                last_values = metric_values[-3:]
                chart_path = None
                try:
                    chart_result = self._charting_tool.run(
                        f"{symbol.upper()} {metric.replace('_', ' ').title()}",
                        last_values,
                    )
                    chart_path = getattr(chart_result, "file_path", None) or getattr(
                        chart_result, "get", lambda *_: None
                    )("file_path")
                    if chart_path:
                        self._register_chart_path(chart_path)
                except Exception as chart_exc:  # pragma: no cover - best effort
                    chart_path = f"chart_unavailable: {chart_exc}"

                metric_summaries[metric] = {
                    "last_values": last_values,
                    "chart_path": chart_path,
                }

        payload = {
            "symbol": symbol.upper(),
            "source": data.get("source") if isinstance(data, dict) else None,
            "condensed_metrics": metric_summaries,
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
            "note": "Raw historical tables trimmed to last 3 periods per metric to control token usage.",
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        self._candidate_metrics_cache[symbol.upper()] = payload["screening_metrics"]
        return self._format_json(payload)

    def get_real_time_quote(self, symbol: str) -> str:
        """Fetch intraday price, volume, and market cap."""
        symbol = self._resolve_yf_symbol(symbol)
        try:
            price, volume, market_cap = self._real_time_quote_tool.run(symbol)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error(
                "get_real_time_quote", symbol=symbol, error=str(exc)
            )

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
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error("get_recent_news", symbol=symbol, error=str(exc))

        payload = {
            "symbol": symbol.upper(),
            "headlines": headlines,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def search_market_events(self, query: str) -> str:
        """Use EXA Web Search to research market or company developments."""
        try:
            results = self.web_search_tool.run(query)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error(
                "search_market_events", query=query, error=str(exc)
            )

        payload = {
            "query": query,
            "results": self._compact_search_results(results),
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def search_global_research(self, query: str) -> str:
        """Use Tavily to capture broader macro or thematic research."""
        try:
            results = self._tavily_tool.run(query)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error(
                "search_global_research", query=query, error=str(exc)
            )

        payload = {
            "query": query,
            "results": self._compact_search_results(results),
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def reddit_sentiment_scan(self, query: str, max_posts: int = 30) -> str:
        """Collect Reddit posts for the LLM to analyze (no heuristic scoring)."""
        subs = ["Baystreetbets", "wallstreetbets"]
        aggregated_posts: List[Dict[str, Any]] = []
        normalized_query = query.upper()

        for subreddit in subs:
            path = f"/r/{subreddit}/search"
            params = {
                "q": f"{normalized_query} TSX"
                if subreddit == "Baystreetbets"
                else normalized_query,
                "restrict_sr": "1",
                "sort": "new",
                "limit": str(max_posts // len(subs)),
                "t": "week",
            }
            try:
                response = self._reddit_client.get(path, params=params, timeout=15)
                response.raise_for_status()
                items = response.json().get("data", {}).get("children", [])
            except Exception as exc:  # pragma: no cover - relies on Reddit uptime
                return self._format_error(
                    "reddit_sentiment_scan",
                    query=query,
                    error=f"{subreddit} fetch failed: {exc}",
                )

            for child in items:
                post = child.get("data", {})
                aggregated_posts.append(
                    {
                        "subreddit": subreddit,
                        "title": post.get("title"),
                        "selftext": (post.get("selftext") or "")[:1000],
                        "score": post.get("score"),
                        "num_comments": post.get("num_comments"),
                        "created_utc": post.get("created_utc"),
                        "author": post.get("author"),
                        "permalink": f"https://www.reddit.com{post.get('permalink', '')}",
                    }
                )

        payload = {
            "query": normalized_query,
            "primary_focus": "Baystreetbets",
            "posts": aggregated_posts,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def discover_reddit_tickers(self, max_posts: int = 60) -> str:
        """Aggregate recent Reddit posts, fetch top comments, and summarize in batches."""
        subs = ["Baystreetbets", "wallstreetbets"]
        collected_posts: List[Dict[str, Any]] = []

        for subreddit in subs:
            path = f"/r/{subreddit}/new"
            params = {"limit": str(max_posts // len(subs))}
            try:
                response = self._reddit_client.get(path, params=params, timeout=15)
                response.raise_for_status()
                items = response.json().get("data", {}).get("children", [])
            except Exception as exc:
                return self._format_error(
                    "discover_reddit_tickers",
                    error=f"{subreddit} fetch failed: {exc}",
                )

            for child in items:
                post = child.get("data", {})
                permalink = f"https://www.reddit.com{post.get('permalink', '')}"
                top_comments = self._fetch_top_comments(permalink)
                collected_posts.append(
                    {
                        "subreddit": subreddit,
                        "title": post.get("title"),
                        "selftext": post.get("selftext"),
                        "score": post.get("score"),
                        "num_comments": post.get("num_comments"),
                        "created_utc": post.get("created_utc"),
                        "author": post.get("author"),
                        "permalink": permalink,
                        "top_comments": top_comments,
                    }
                )

        payload = {
            "posts": collected_posts,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }

        batch_summaries: List[Dict[str, Any]] = []
        if collected_posts:
            try:
                summary_model = OpenAIChat(
                    id=self.summary_model_id, temperature=1, max_completion_tokens=4000
                )
                summarizer = Agent(model=summary_model, markdown=True)
                for idx in range(0, len(collected_posts), 50):
                    batch = collected_posts[idx : idx + 50]
                    batch_payload = json.dumps(batch, indent=2)
                    summary = summarizer.run(
                        f"Summarize Reddit posts batch {idx // 50 + 1}. Highlight emerging tickers, "
                        "catalysts, sentiment, and risks.\n\n"
                        "IMPORTANT: For each ticker mentioned, include the Reddit post URL (permalink) "
                        "so it can be cited in the final report. Format: [Ticker mentioned in r/subreddit](permalink)\n\n"
                        f"{batch_payload}"
                    )
                    batch_summaries.append(
                        {
                            "batch": idx // 50 + 1,
                            "range": f"posts {idx + 1}-{min(idx + 50, len(collected_posts))}",
                            "summary": getattr(summary, "content", str(summary)),
                        }
                    )
            except Exception:
                batch_summaries.append(
                    {
                        "batch": 0,
                        "range": "all posts",
                        "summary": "LLM summary unavailable.",
                    }
                )

        payload["batch_summaries"] = batch_summaries
        return json.dumps(payload, indent=2)

    def summarize_context_tool(self, text: str, max_tokens: int = 400000) -> str:
        """Allow agents to compress long context blocks proactively."""
        token_count = self._count_tokens(text)
        summary = self._truncate_for_context(text, max_tokens)
        payload = {
            "original_tokens": token_count,
            "max_tokens": max_tokens,
            "summary": summary,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        if token_count > max_tokens:
            payload["note"] = (
                "Input exceeded the requested budget; summary contains the highest-priority sections."
            )
        return self._format_json(payload)

    async def run_reddit_sentiment_analysis(self, ticker: str) -> str:
        """Execute the BayStreet-focused Reddit sentiment agent."""
        template = load_prompt("reddit_sentiment_prompt")
        prompt = template.format(ticker=ticker.upper())
        result = await self.reddit_sentiment_agent.arun(prompt)
        return self._extract_content(result)

    def get_options_chain_snapshot(self, symbol: str, expiration_date: str) -> str:
        """Return the nearest available options chain for an expiration date."""
        try:
            calls, puts, actual_expiration = self._options_chain_tool.run(
                symbol, expiration_date
            )
        except Exception as exc:  # pragma: no cover - depends on API
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
            "calls": [
                {
                    "strike": row.get("strike"),
                    "lastPrice": row.get("lastPrice"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "volume": row.get("volume"),
                    "openInterest": row.get("openInterest"),
                    "impliedVolatility": row.get("impliedVolatility"),
                }
                for row in (calls or [])[:8]
            ],
            "puts": [
                {
                    "strike": row.get("strike"),
                    "lastPrice": row.get("lastPrice"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "volume": row.get("volume"),
                    "openInterest": row.get("openInterest"),
                    "impliedVolatility": row.get("impliedVolatility"),
                }
                for row in (puts or [])[:8]
            ],
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def create_metric_chart(self, metric_name: str, data: List[float]) -> str:
        """Persist a PNG chart for the given metric data."""
        try:
            chart_output = self._charting_tool.run(metric_name, data)
        except Exception as exc:  # pragma: no cover - depends on API
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

    # -------------------------------------------------------------------------
    # Database tool functions
    # -------------------------------------------------------------------------

    def search_past_finds(
        self,
        query: str = "",
        exchange: str = "",
        sector: str = "",
        min_confidence: float = 5.0,
        days_ago: int = 180,
    ) -> str:
        """Search historical stock discoveries from past analyses."""
        try:
            finds = self.db.search_past_finds(
                query=query if query else None,
                exchange=exchange if exchange else None,
                sector=sector if sector else None,
                min_confidence=min_confidence,
                days_ago=days_ago,
                limit=20,
            )

            return self.db.format_for_agent(finds)

        except Exception as exc:
            return self._format_error(
                "search_past_finds",
                query=query,
                exchange=exchange,
                sector=sector,
                error=str(exc),
            )

    def get_ticker_history(self, ticker: str) -> str:
        """Get complete analysis history for a specific ticker."""
        try:
            history = self.db.get_ticker_history(ticker)
            reddit_mentions = self.db.get_reddit_mentions(ticker, days=90)

            payload = {
                "ticker": ticker.upper(),
                "total_analyses": len(history),
                "past_finds": [
                    {
                        "discovered_at": h.get("discovered_at"),
                        "confidence_score": h.get("confidence_score"),
                        "current_price": h.get("current_price"),
                        "investment_thesis": h.get("investment_thesis"),
                        "exchange": h.get("exchange"),
                    }
                    for h in history[:5]  # Last 5 analyses
                ],
                "reddit_mentions_count": len(reddit_mentions),
                "recent_reddit_sentiment": [
                    {
                        "subreddit": r.get("subreddit"),
                        "title": r.get("title"),
                        "sentiment": r.get("sentiment"),
                        "score": r.get("score"),
                        "mentioned_at": r.get("mentioned_at"),
                    }
                    for r in reddit_mentions[:3]
                ],
                "retrieved_at": datetime.now(ny_timezone).isoformat(),
            }

            return self._format_json(payload)

        except Exception as exc:
            return self._format_error(
                "get_ticker_history", ticker=ticker, error=str(exc)
            )

    def get_similar_stocks(self, ticker: str, sector: str = "") -> str:
        """Find similar stocks based on sector/industry from past research."""
        try:
            similar = self.db.get_similar_stocks(
                ticker, sector=sector if sector else None, limit=10
            )

            payload = {
                "reference_ticker": ticker.upper(),
                "similar_stocks": [
                    {
                        "ticker": s.get("ticker"),
                        "company_name": s.get("company_name"),
                        "sector": s.get("sector"),
                        "industry": s.get("industry"),
                        "confidence_score": s.get("confidence_score"),
                        "exchange": s.get("exchange"),
                    }
                    for s in similar
                ],
                "total": len(similar),
                "retrieved_at": datetime.now(ny_timezone).isoformat(),
            }

            return self._format_json(payload)

        except Exception as exc:
            return self._format_error(
                "get_similar_stocks", ticker=ticker, sector=sector, error=str(exc)
            )

    def get_recent_discoveries(
        self, days: int = 30, min_confidence: float = 6.0
    ) -> str:
        """Get recent high-quality stock discoveries."""
        try:
            discoveries = self.db.get_recent_discoveries(
                days=days, min_confidence=min_confidence
            )

            payload = {
                "period_days": days,
                "min_confidence": min_confidence,
                "discoveries": [
                    {
                        "ticker": d.get("ticker"),
                        "company_name": d.get("company_name"),
                        "exchange": d.get("exchange"),
                        "sector": d.get("sector"),
                        "confidence_score": d.get("confidence_score"),
                        "discovery_source": d.get("discovery_source"),
                        "investment_thesis": d.get("investment_thesis", "")[:200],
                        "discovered_at": d.get("discovered_at"),
                    }
                    for d in discoveries
                ],
                "total": len(discoveries),
                "retrieved_at": datetime.now(ny_timezone).isoformat(),
            }

            return self._format_json(payload)

        except Exception as exc:
            return self._format_error(
                "get_recent_discoveries",
                days=days,
                min_confidence=min_confidence,
                error=str(exc),
            )

    def _split_resolved_symbol(self, symbol: str) -> tuple[str, str, str]:
        """Return base ticker, exchange suffix, and DB exchange from a Yahoo symbol."""
        resolved = self._resolve_yf_symbol(symbol)
        if "." in resolved:
            ticker, suffix = resolved.split(".", 1)
        else:
            ticker, suffix = resolved, ""
        exchange = "TSX" if suffix in {"TO", "V", "CN", "NE"} else "US"
        return ticker.upper(), suffix.upper(), exchange

    def _normalize_report_ticker(self, ticker_info: Dict[str, Any]) -> Dict[str, str]:
        """Canonicalize a ticker extracted from model output before validation."""
        raw_ticker = str(ticker_info.get("ticker") or "").strip().upper()
        raw_suffix = str(ticker_info.get("exchange_suffix") or "").strip().upper()
        if raw_suffix and "." not in raw_ticker:
            raw_symbol = f"{raw_ticker}.{raw_suffix}"
        else:
            raw_symbol = raw_ticker
        ticker, suffix, exchange = self._split_resolved_symbol(raw_symbol)
        return {
            "ticker": ticker,
            "exchange_suffix": suffix,
            "exchange": exchange,
            "resolved_symbol": f"{ticker}.{suffix}" if suffix else ticker,
            "company_name": str(ticker_info.get("company_name") or "").strip(),
        }

    def _extract_candidate_tickers_from_report(self, report: str) -> list[dict]:
        """Extract likely stock tickers from headings/bold entries only."""
        candidates: list[dict] = []
        seen: set[str] = set()
        ignore = {
            "CEO",
            "CFO",
            "CAD",
            "USD",
            "EPS",
            "EBITDA",
            "P/E",
            "VHI",
            "RSI",
            "ETF",
            "ET",
        }
        patterns = [
            re.compile(
                r"^\s*(?:#{2,4}\s*)?(?:\d+\.\s*)?\*\*([^(\n]{2,80})"
                r"\((?:([A-Z]{2,5}):)?([A-Z]{1,6}(?:\.[A-Z]{1,3})?)\)\*\*",
                re.MULTILINE,
            ),
            re.compile(
                r"^\s*(?:#{2,4}\s*)?(?:\d+\.\s*)?([^(\n]{2,80})"
                r"\((?:([A-Z]{2,5}):)?([A-Z]{1,6}(?:\.[A-Z]{1,3})?)\)",
                re.MULTILINE,
            ),
        ]

        for pattern in patterns:
            for match in pattern.finditer(report):
                company_name = match.group(1).strip(" -*#")
                exchange = match.group(2) or ""
                ticker_full = match.group(3).strip().upper()
                if ticker_full in ignore:
                    continue
                if "." in ticker_full:
                    ticker, suffix = ticker_full.split(".", 1)
                else:
                    ticker = ticker_full
                    suffix = "TO" if exchange == "TSX" else ""
                key = f"{ticker}.{suffix}" if suffix else ticker
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "ticker": ticker,
                        "exchange_suffix": suffix,
                        "company_name": company_name,
                    }
                )

        return candidates

    def _get_live_screening_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch quote/profile/financial metrics used for hard screening."""
        resolved = self._resolve_yf_symbol(symbol)
        cache_key = resolved.upper()
        if cache_key in self._candidate_metrics_cache:
            return self._candidate_metrics_cache[cache_key]

        metrics: Dict[str, Any] = {"resolved_symbol": resolved}
        try:
            price, volume, market_cap = self._real_time_quote_tool.run(resolved)
            metrics.update(
                {
                    "current_price": price,
                    "volume": volume,
                    "market_cap": market_cap,
                }
            )
        except Exception as exc:
            metrics["quote_error"] = str(exc)

        try:
            profile = self._company_info_tool.run(resolved)
            if isinstance(profile, list) and len(profile) >= 6:
                metrics.update(
                    {
                        "company_name": profile[0],
                        "sector": profile[1],
                        "industry": profile[2],
                        "market_cap": metrics.get("market_cap") or profile[3],
                        "forward_pe": profile[4],
                    }
                )
        except Exception as exc:
            metrics["profile_error"] = str(exc)

        try:
            facts = self._financial_report_tool.run(resolved)
            if isinstance(facts, dict):
                metrics.update(
                    {
                        "market_cap": metrics.get("market_cap")
                        or facts.get("market_cap"),
                        "current_price": metrics.get("current_price")
                        or facts.get("current_price"),
                        "volume": metrics.get("volume") or facts.get("volume"),
                        "price_to_earnings": facts.get("price_to_earnings"),
                        "forward_pe": metrics.get("forward_pe")
                        or facts.get("forward_pe"),
                        "price_to_book": facts.get("price_to_book"),
                        "current_ratio": facts.get("current_ratio"),
                        "debt_to_equity": facts.get("debt_to_equity"),
                    }
                )
        except Exception as exc:
            metrics["financial_error"] = str(exc)

        self._candidate_metrics_cache[cache_key] = metrics
        return metrics

    def _screen_candidate_metrics(
        self, symbol: str, metrics: Dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Apply non-negotiable numeric filters before persistence/tracking."""
        prefs = self.preferences
        failures: list[str] = []
        price = metrics.get("current_price")
        market_cap = metrics.get("market_cap")
        volume = metrics.get("volume")
        pe_ratio = metrics.get("price_to_earnings") or metrics.get("forward_pe")
        current_ratio = metrics.get("current_ratio")
        debt_to_equity = metrics.get("debt_to_equity")

        if price is None:
            failures.append("missing current price")
        elif price < prefs.min_price or price > prefs.max_price:
            failures.append(
                f"price {price:.2f} outside {prefs.min_price:.2f}-{prefs.max_price:.2f}"
            )

        if market_cap is None:
            failures.append("missing market cap")
        elif market_cap < prefs.min_market_cap:
            failures.append(
                f"market cap {market_cap:,.0f} below {prefs.min_market_cap:,.0f}"
            )

        if volume is None:
            failures.append("missing volume")
        elif volume < prefs.min_volume:
            failures.append(f"volume {volume:,.0f} below {prefs.min_volume:,.0f}")

        if pe_ratio is None:
            failures.append("missing P/E")
        elif pe_ratio > prefs.max_pe:
            failures.append(f"P/E {pe_ratio:.2f} above {prefs.max_pe:.2f}")

        if current_ratio is None:
            failures.append("missing current ratio")
        elif current_ratio < prefs.min_current_ratio:
            failures.append(
                f"current ratio {current_ratio:.2f} below {prefs.min_current_ratio:.2f}"
            )

        if debt_to_equity is None:
            failures.append("missing debt/equity")
        elif debt_to_equity > prefs.max_debt_equity:
            failures.append(
                f"debt/equity {debt_to_equity:.2f} above {prefs.max_debt_equity:.2f}"
            )

        return not failures, failures

    def _hard_screen_report_candidates(self, report: str) -> tuple[str, list[dict]]:
        """Append deterministic pass/fail results for tickers named in a report."""
        extracted = self._extract_candidate_tickers_from_report(report)
        if not extracted:
            return report, []

        accepted: list[dict] = []
        review_lines = [
            "## Code-Level Hard Screen Review",
            "The following pass/fail status was computed from live market-data tools.",
        ]

        for ticker_info in extracted:
            normalized = self._normalize_report_ticker(ticker_info)
            resolved_symbol = normalized["resolved_symbol"]
            metrics = self._get_live_screening_metrics(resolved_symbol)
            passed, failures = self._screen_candidate_metrics(resolved_symbol, metrics)
            status = "PASS" if passed else "REJECT"
            review_lines.append(
                "- "
                f"{status}: {resolved_symbol} | "
                f"price={metrics.get('current_price')} | "
                f"market_cap={metrics.get('market_cap')} | "
                f"volume={metrics.get('volume')} | "
                f"pe={metrics.get('price_to_earnings') or metrics.get('forward_pe')} | "
                f"current_ratio={metrics.get('current_ratio')} | "
                f"debt_to_equity={metrics.get('debt_to_equity')} | "
                f"reason={'; '.join(failures) if failures else 'meets hard filters'}"
            )
            if passed:
                normalized["metrics"] = metrics
                accepted.append(normalized)

        return f"{report.rstrip()}\n\n" + "\n".join(review_lines), accepted

    async def _extract_tickers_with_llm(self, report: str) -> list[dict]:
        """Use LLM to extract ticker symbols and metadata from report.

        Returns list of dicts with ticker, company_name, and section_content.
        """
        import json
        import os

        from openai import AsyncOpenAI

        extraction_prompt = f"""Extract all stock ticker symbols from this report.
For each ticker, provide:
1. ticker: The stock symbol (e.g., "MDA", "CVO")
2. exchange_suffix: The exchange suffix if present (e.g., "TO", "V", "")
3. company_name: The full company name

Return ONLY a JSON array, no other text:
[{{"ticker": "MDA", "exchange_suffix": "TO", "company_name": "MDA Space Ltd."}}, ...]

Report:
{report[:4000]}
"""

        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model=self.extraction_model_id,
                temperature=1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts ticker symbols from stock reports.",
                    },
                    {"role": "user", "content": extraction_prompt},
                ],
            )

            content = response.choices[0].message.content.strip()

            # Clean response - remove markdown code fences if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            tickers = json.loads(content)
            return tickers if isinstance(tickers, list) else []

        except Exception as e:
            print(f"  LLM ticker extraction failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    async def _save_stock_finds_from_report(self, report: str) -> int:
        """Parse final report and save stock finds to database.

        Returns number of stocks saved.
        """
        import re
        from datetime import datetime

        if not report or not self.current_run_id:
            return 0

        saved_count = 0

        print("  Extracting candidate tickers from report...")
        ticker_list = self._extract_candidate_tickers_from_report(report)

        if not ticker_list:
            print("  Regex extraction found no final-report tickers; trying LLM fallback...")
            ticker_list = await self._extract_tickers_with_llm(report)

        if not ticker_list:
            print("  No ticker patterns found in report")
            return 0

        # Process each ticker
        for ticker_info in ticker_list:
            normalized = self._normalize_report_ticker(ticker_info)
            ticker = normalized["ticker"]
            exchange_suffix = normalized["exchange_suffix"]
            exchange = normalized["exchange"]
            resolved_symbol = normalized["resolved_symbol"]
            company_name = normalized.get("company_name", "")

            metrics = self._get_live_screening_metrics(resolved_symbol)
            passed, failures = self._screen_candidate_metrics(resolved_symbol, metrics)
            if not passed:
                print(
                    f"  Skipping {resolved_symbol}: hard screen failed ({'; '.join(failures)})"
                )
                continue
            if not company_name:
                company_name = metrics.get("company_name") or resolved_symbol

            # Find content section for this ticker in the report
            ticker_full = resolved_symbol
            search_pattern = re.escape(ticker_full)
            ticker_match = re.search(search_pattern, report)

            if ticker_match:
                start_pos = ticker_match.end()
                # Find next ticker or end of report
                next_ticker_pos = len(report)
                for next_info in ticker_list[ticker_list.index(ticker_info) + 1 :]:
                    next_normalized = self._normalize_report_ticker(next_info)
                    next_full = next_normalized["resolved_symbol"]
                    next_match = re.search(re.escape(next_full), report[start_pos:])
                    if next_match:
                        next_ticker_pos = start_pos + next_match.start()
                        break
                content = report[start_pos:next_ticker_pos]
            else:
                content = report  # Use whole report if specific section not found

            try:
                # Extract confidence score - look for patterns like "Confidence: 8.5" or "Score: 8/10"
                confidence_score = None
                confidence_patterns = [
                    r"confidence[:\s]+(\d+(?:\.\d+)?)",
                    r"score[:\s]+(\d+(?:\.\d+)?)",
                    r"rating[:\s]+(\d+(?:\.\d+)?)",
                ]
                for pattern in confidence_patterns:
                    conf_match = re.search(pattern, content, re.IGNORECASE)
                    if conf_match:
                        confidence_score = float(conf_match.group(1))
                        break
                if confidence_score is None:
                    confidence_score = 5.0

                # Extract price - matches "$6.78" or "Price: 6.78"
                price_match = re.search(
                    r"(?:price|trading)[:\s]+\$?([\d,]+\.?\d*)", content, re.IGNORECASE
                )
                current_price = (
                    float(price_match.group(1).replace(",", ""))
                    if price_match
                    else None
                )
                current_price = current_price or metrics.get("current_price")

                # Extract market cap - matches "$650.9M" or "Mkt Cap: $650.9M"
                mcap_match = re.search(
                    r"(?:mkt\s+cap|market\s+cap)[:\s]+\$?([\d,]+\.?\d*)\s*([BMK])?",
                    content,
                    re.IGNORECASE,
                )
                market_cap = None
                if mcap_match:
                    value = float(mcap_match.group(1).replace(",", ""))
                    multiplier = mcap_match.group(2)
                    if multiplier == "B":
                        market_cap = value * 1_000_000_000
                    elif multiplier == "M":
                        market_cap = value * 1_000_000
                    elif multiplier == "K":
                        market_cap = value * 1_000
                    else:
                        market_cap = value
                market_cap = market_cap or metrics.get("market_cap")

                # Extract P/E ratio
                pe_match = re.search(
                    r"P/E[:\s]+(?:\(fwd\)[:\s]+)?([\d,]+\.?\d*)", content, re.IGNORECASE
                )
                pe_ratio = (
                    float(pe_match.group(1).replace(",", "")) if pe_match else None
                )
                pe_ratio = pe_ratio or metrics.get("price_to_earnings") or metrics.get(
                    "forward_pe"
                )

                # Extract sector/industry from content or company name
                sector_match = re.search(
                    r"sector[:\s]+([^\n]+)", content, re.IGNORECASE
                )
                sector = (
                    sector_match.group(1).strip()
                    if sector_match
                    else metrics.get("sector")
                )

                industry_match = re.search(
                    r"industry[:\s]+([^\n]+)", content, re.IGNORECASE
                )
                industry = (
                    industry_match.group(1).strip()
                    if industry_match
                    else metrics.get("industry")
                )

                # Extract investment thesis - look for Reddit catalyst or key points
                thesis_parts = []

                # Look for Reddit catalyst
                catalyst_match = re.search(
                    r"\*\*Reddit Catalyst:\*\*\s*([^\n]+)", content
                )
                if catalyst_match:
                    thesis_parts.append(catalyst_match.group(1).strip())

                # Get first few non-header lines as thesis
                for line in content.split("\n")[:15]:
                    line = line.strip()
                    if (
                        line
                        and not line.startswith("#")
                        and not line.startswith("**")
                        and not line.startswith("-")
                        and len(line) > 20
                    ):
                        thesis_parts.append(line)
                        if len(thesis_parts) >= 2:
                            break

                investment_thesis = (
                    " ".join(thesis_parts)[:500] if thesis_parts else company_name
                )

                # Create stock find record
                find_data = {
                    "analysis_run_id": self.current_run_id,
                    "ticker": ticker,
                    "company_name": company_name,
                    "exchange": exchange,
                    "sector": sector,
                    "industry": industry,
                    "discovery_source": "screening",
                    "confidence_score": confidence_score,
                    "current_price": current_price,
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "debt_to_equity": metrics.get("debt_to_equity"),
                    "current_ratio": metrics.get("current_ratio"),
                    "price_to_book": metrics.get("price_to_book"),
                    "investment_thesis": investment_thesis,
                    "discovered_at": datetime.now(ny_timezone).isoformat(),
                }

                # Save to database
                self.db.save_stock_find(find_data)
                saved_count += 1
                print(
                    f"  Saved {ticker} ({company_name}) - Price: ${current_price}, Confidence: {confidence_score}"
                )

            except Exception as e:
                print(f"  Warning: Could not save {ticker}: {e}")
                import traceback

                traceback.print_exc()
                continue

        return saved_count

    async def _trigger_portfolio_tracking(self, run_id: int):
        """Initialize portfolio tracking for stocks from this analysis run."""
        try:
            from stockbot.flows.performance_tracker import PerformanceTrackerFlow

            tracker = PerformanceTrackerFlow()
            await tracker.initialize_holdings_from_run(run_id)
            print(f"  Portfolio tracking initialized for run #{run_id}")

        except Exception as e:
            print(f"  Error initializing portfolio tracking: {e}")

    # -------------------------------------------------------------------------
    # Analysis workflow
    # -------------------------------------------------------------------------
    async def run_value_screening(self) -> str:
        """Execute only the initial screening prompt."""
        reddit_summary = self._get_reddit_discovery_summary()
        self._log_prompt_size("Reddit discovery summary", reddit_summary)
        screening_prompt = self._build_screening_prompt(reddit_summary)
        self._log_prompt_size("Screening prompt", screening_prompt)
        screening_result = await self.screening_agent.arun(
            screening_prompt,
            images=self._build_image_payload(),
        )
        return self._extract_content(screening_result)

    def _run_startup_ritual(self) -> None:
        """Log past-performance context before any agent runs.

        Surfaces what the learning system would otherwise inject silently into the
        screening prompt. Helps you read the run log and understand which insights
        were available to the agent at the time.
        """
        print("\n=== Startup Ritual ===")
        try:
            section, _adjustments = self._build_learning_insights_section()
        except Exception as exc:
            print(f"  Learning insights unavailable: {exc}")
            return
        if not section.strip():
            print("  No prior performance data — running cold.")
            return
        for line in section.strip().splitlines():
            print(f"  {line}")
        print("======================\n")

    async def execute_legacy_reddit_analysis(self) -> str:
        """Legacy Reddit-first flow. Kept for comparison; not called by main.py.

        See execute_undervalued_analysis below for the funnel-first replacement.
        """
        log_file_path = self._configure_agno_debug_logging()
        print(f"Agno debug logs: {log_file_path}")

        # Startup ritual: log past-performance context before agents run
        self._run_startup_ritual()

        # Create analysis run in database
        self.current_run_id = self.db.create_analysis_run(
            run_type="undervalued",
            preferences=self.preferences.__dict__,
        )
        print(f"\nStarted analysis run #{self.current_run_id}")
        write_state(
            "undervalued",
            run_id=self.current_run_id,
            phase="started",
            preferences=self.preferences.model_dump(),
        )

        try:
            print("\nExecuting undervalued stock screening with Agno...")
            screening_raw = await self.run_value_screening()
            sanitized_screening = self._sanitize_agent_output(screening_raw)
            sanitized_screening, accepted_screening_candidates = (
                self._hard_screen_report_candidates(sanitized_screening)
            )
            if accepted_screening_candidates:
                print(
                    "  Hard screen accepted "
                    f"{len(accepted_screening_candidates)} report candidates"
                )
            screening_context = self._truncate_for_context(
                sanitized_screening, self.SCREENING_TOKEN_LIMIT
            )
            await self.save_phase_output(
                "initial_screening",
                sanitized_screening,
                "Initial value stock screening results",
            )
            write_state(
                "undervalued",
                run_id=self.current_run_id,
                phase="screening_complete",
                screening_candidates=len(accepted_screening_candidates),
            )

            print("\nAnalyzing turnaround potential...")
            turnaround_prompt = self._build_turnaround_prompt(screening_context)
            self._log_prompt_size("Turnaround input summary", screening_context)
            self._log_prompt_size("Turnaround prompt", turnaround_prompt)
            turnaround_result = await self.turnaround_agent.arun(
                turnaround_prompt,
                images=self._build_image_payload(),
            )
            turnaround_content = self._extract_content(turnaround_result)
            await self.save_phase_output(
                "turnaround_analysis",
                turnaround_content,
                "Detailed turnaround potential analysis",
            )

            final_report = self.create_final_report(
                sanitized_screening, turnaround_content
            )
            await self.save_phase_output(
                "final_report",
                final_report,
                "Final undervalued stocks analysis",
            )

            # Parse and save stock finds to database with LLM extraction
            saved_count = await self._save_stock_finds_from_report(final_report)

            # Save research reports to database
            self.db.add_research_note(
                ticker="ANALYSIS_RUN",
                note_type="screening_report",
                content=sanitized_screening[:5000],  # Truncate to fit in database
                source="screening_agent",
            )
            self.db.add_research_note(
                ticker="ANALYSIS_RUN",
                note_type="turnaround_report",
                content=turnaround_content[:5000],
                source="turnaround_agent",
            )
            self.db.add_research_note(
                ticker="ANALYSIS_RUN",
                note_type="final_report",
                content=final_report[:5000],
                source="undervalued_flow",
            )
            print(f"  Saved research reports to database")

            # Complete analysis run
            self.db.complete_analysis_run(
                run_id=self.current_run_id,
                total_candidates=saved_count,
                final_selections=saved_count,
                status="completed",
            )
            print(
                f"\nAnalysis run #{self.current_run_id} completed. Saved {saved_count} stock finds to database."
            )

            # Initialize portfolio tracking for all saved stocks
            await self._trigger_portfolio_tracking(self.current_run_id)

            clear_state("undervalued")
            return final_report

        except Exception as e:
            # Mark run as failed
            if self.current_run_id:
                self.db.complete_analysis_run(
                    run_id=self.current_run_id,
                    status="failed",
                )
            write_state(
                "undervalued",
                run_id=self.current_run_id,
                phase="failed",
                error=str(e),
            )
            raise

    # -------------------------------------------------------------------------
    # Funnel-first analysis (replaces legacy Reddit-driven flow)
    # -------------------------------------------------------------------------
    def _preferences_to_gates(self) -> ScreeningGates:
        """Map the ValueScreeningPreferences (user-tunable) onto funnel gates."""
        p = self.preferences
        return ScreeningGates(
            max_price=p.max_price,
            min_price=p.min_price,
            min_volume=p.min_volume,
            max_pe=p.max_pe,
            min_market_cap=p.min_market_cap,
            max_debt_equity=p.max_debt_equity,
            min_current_ratio=p.min_current_ratio,
            max_decline_from_high=p.price_vs_high,
            require_positive_fcf=True,
        )

    def _get_reddit_overlay(self, shortlist: List[FunnelCandidate]) -> str:
        """Lightweight Reddit catalyst overlay for the shortlist tickers only.

        This intentionally does NOT drive discovery. It surfaces whether any of
        the funnel's picks are currently being discussed on Reddit — useful as
        sentiment context for the thesis writer, nothing more.
        """
        mentions: List[str] = []
        for c in shortlist[:5]:  # cap to cheapest 5 to limit API calls
            try:
                raw = self.reddit_sentiment_scan(c.symbol, max_posts=10)
                payload = json.loads(raw)
                posts = payload.get("posts") or []
                if posts:
                    titles = [p.get("title") for p in posts[:3] if p.get("title")]
                    if titles:
                        mentions.append(f"- {c.symbol}: {'; '.join(titles[:2])}")
            except Exception:
                continue
        if not mentions:
            return "No funnel-shortlisted tickers had recent Reddit mentions."
        return "\n".join(mentions)

    def _build_funnel_deep_dive_prompt(
        self,
        candidates: List[FunnelCandidate],
        stats: Dict[str, Any],
        reddit_overlay: str,
        bottleneck_context: str,
    ) -> str:
        now = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        summaries = [c.to_prompt_summary() for c in candidates]
        template = load_prompt("funnel_deep_dive_prompt")
        return template.format(
            timestamp=now,
            candidate_count=len(candidates),
            funnel_stats=json.dumps(stats, indent=2, default=str),
            candidates_json=json.dumps(summaries, indent=2, default=str),
            reddit_overlay=reddit_overlay,
            bottleneck_context=bottleneck_context,
        )

    @staticmethod
    def _parse_bottleneck_research_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse and validate the web research agent's strict JSON response."""
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        validated = BottleneckResearchOutput.model_validate(payload).model_dump(
            mode="json"
        )
        candidates_by_ticker: Dict[str, Dict[str, Any]] = {}
        for candidate in validated["candidates"]:
            ticker = normalize_ticker(candidate["ticker"])
            candidate["ticker"] = ticker
            previous = candidates_by_ticker.get(ticker)
            if previous is None or candidate["confidence_score"] > previous[
                "confidence_score"
            ]:
                candidates_by_ticker[ticker] = candidate
        validated["candidates"] = list(candidates_by_ticker.values())
        return validated

    async def _research_bottlenecks(self) -> Optional[Dict[str, Any]]:
        """Run the broad web-discovery pass; failure leaves the value lane intact."""
        template = load_prompt("bottleneck_research_prompt")
        prompt = template.format(
            timestamp=datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        )
        self._log_prompt_size("Bottleneck research prompt", prompt)
        try:
            response = await self.bottleneck_research_agent.arun(prompt)
            raw_text = self._extract_content(response)
            research = self._parse_bottleneck_research_json(raw_text)
            if research is None:
                await self.save_phase_output(
                    "bottleneck_research_raw",
                    raw_text,
                    "Raw bottleneck research (JSON parse failed)",
                )
                return None
            await self.save_phase_output(
                "bottleneck_research",
                json.dumps(research, indent=2, default=str),
                "Cross-sector bottleneck research",
            )
            return research
        except Exception as exc:
            print(f"  Bottleneck research unavailable: {exc}")
            return None

    @staticmethod
    def _augment_universe_with_research(
        universe: List[Ticker], research: Optional[Dict[str, Any]]
    ) -> List[Ticker]:
        """Ensure researched companies are scanned even outside the base index."""
        augmented = list(universe)
        seen = {normalize_ticker(ticker.symbol) for ticker in augmented}
        for candidate in (research or {}).get("candidates", []):
            ticker = normalize_ticker(candidate["ticker"])
            if ticker in seen:
                continue
            augmented.append(
                Ticker(
                    symbol=ticker,
                    exchange="TSX" if ticker.endswith(".TO") else "US",
                    source="bottleneck_research",
                )
            )
            seen.add(ticker)
        return augmented

    @staticmethod
    def _format_bottleneck_context(
        research: Optional[Dict[str, Any]], candidates: List[FunnelCandidate]
    ) -> str:
        if not research:
            return "Cross-sector bottleneck research was unavailable for this run."
        shortlist_symbols = {candidate.symbol for candidate in candidates}
        return json.dumps(
            {
                "market_regime_summary": research["market_regime_summary"],
                "themes": research["themes"],
                "shortlisted_research_candidates": [
                    candidate
                    for candidate in research["candidates"]
                    if candidate["ticker"] in shortlist_symbols
                ],
            },
            indent=2,
            default=str,
        )

    def _parse_deep_dive_json(
        self, text: str, expected_tickers: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Parse the strict-JSON output from the deep-dive agent. Tolerate fences."""
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None
        expected = [normalize_ticker(ticker) for ticker in expected_tickers]
        if len(expected) != len(set(expected)):
            raise ValueError("expected shortlist contains duplicate tickers")

        # Models occasionally emit a second, malformed placeholder for a ticker
        # they already reviewed. Remove such rows before validating the complete
        # response; otherwise one invalid duplicate prevents the valid verdict
        # from reaching the deduplication below. A malformed sole verdict is
        # deliberately left in place so normal schema validation still rejects it.
        if isinstance(payload, dict) and isinstance(payload.get("stocks"), list):
            grouped_stocks: Dict[str, List[Any]] = {}
            stock_order: List[str] = []
            passthrough_stocks: List[Any] = []
            for raw_stock in payload["stocks"]:
                if not isinstance(raw_stock, dict):
                    passthrough_stocks.append(raw_stock)
                    continue
                try:
                    ticker = normalize_ticker(raw_stock.get("ticker"))
                except ValueError:
                    passthrough_stocks.append(raw_stock)
                    continue
                if ticker not in grouped_stocks:
                    grouped_stocks[ticker] = []
                    stock_order.append(ticker)
                grouped_stocks[ticker].append(raw_stock)

            cleaned_stocks: List[Any] = []
            for ticker in stock_order:
                rows = grouped_stocks[ticker]
                if len(rows) == 1:
                    cleaned_stocks.append(rows[0])
                    continue

                valid_rows: List[Dict[str, Any]] = []
                for row in rows:
                    try:
                        valid_row = DeepDiveStock.model_validate(row).model_dump(
                            mode="json"
                        )
                    except ValueError:
                        continue
                    valid_row["ticker"] = ticker
                    valid_rows.append(valid_row)

                if not valid_rows:
                    cleaned_stocks.extend(rows)
                    continue
                if any(row != valid_rows[0] for row in valid_rows[1:]):
                    raise ValueError(
                        f"deep-dive output contains conflicting duplicate ticker: {ticker}"
                    )
                cleaned_stocks.append(valid_rows[0])

            payload = {**payload, "stocks": cleaned_stocks + passthrough_stocks}

        validated = DeepDiveOutput.model_validate(payload).model_dump(mode="json")

        unique_stocks: List[Dict[str, Any]] = []
        stocks_by_ticker: Dict[str, Dict[str, Any]] = {}
        for stock in validated["stocks"]:
            ticker = normalize_ticker(stock["ticker"])
            stock["ticker"] = ticker
            previous = stocks_by_ticker.get(ticker)
            if previous is None:
                stocks_by_ticker[ticker] = stock
                unique_stocks.append(stock)
            elif stock != previous:
                raise ValueError(
                    f"deep-dive output contains conflicting duplicate ticker: {ticker}"
                )

        validated["stocks"] = unique_stocks
        actual = [stock["ticker"] for stock in unique_stocks]
        unexpected = sorted(set(actual) - set(expected))
        if unexpected:
            raise ValueError(f"deep-dive output contains unexpected tickers: {unexpected}")
        omitted = sorted(set(expected) - set(actual))
        if omitted:
            raise ValueError(f"deep-dive output omitted expected tickers: {omitted}")
        validated["shortlist_review"]["candidates_reviewed"] = len(
            validated["stocks"]
        )
        validated["shortlist_review"]["candidates_accepted"] = sum(
            stock["verdict"] == "accept" for stock in validated["stocks"]
        )
        return validated

    @staticmethod
    def _require_deep_dive(
        deep_dive: Optional[Dict[str, Any]], shortlist: List[FunnelCandidate]
    ) -> None:
        if shortlist and not deep_dive:
            raise RuntimeError(
                "deep-dive analysis returned empty or unparseable output for a nonempty shortlist"
            )

    def _persist_failed_run(self, error: BaseException, phase: str = "failed") -> None:
        if self.current_run_id:
            self.db.complete_analysis_run(
                run_id=self.current_run_id, status="failed"
            )
        write_state(
            "undervalued",
            run_id=self.current_run_id,
            phase=phase,
            error=str(error),
        )

    async def _save_funnel_stocks(
        self,
        deep_dive: Dict[str, Any],
        funnel_map: Dict[str, FunnelCandidate],
    ) -> int:
        """Persist accepted picks from the JSON deep-dive output."""
        if not self.current_run_id or not deep_dive:
            return 0
        saved = 0
        for stock in deep_dive.get("stocks", []):
            ticker = (stock.get("ticker") or "").strip().upper()
            if not ticker or stock.get("verdict") != "accept":
                continue
            candidate = funnel_map.get(ticker)
            if not candidate:
                continue
            snap = candidate.snapshot
            exchange = "TSX" if snap.exchange == "TSX" else "US"
            find_data = {
                "analysis_run_id": self.current_run_id,
                "ticker": ticker,
                "company_name": stock.get("company_name") or ticker,
                "exchange": exchange,
                "sector": snap.sector,
                "industry": snap.industry,
                "discovery_source": "+".join(
                    getattr(candidate, "discovery_lanes", [])
                )
                or "quant_funnel",
                "confidence_score": float(stock.get("confidence_score") or 5.0),
                "current_price": snap.price,
                "market_cap": snap.market_cap,
                "pe_ratio": (
                    snap.trailing_pe
                    if snap.trailing_pe is not None
                    else snap.forward_pe
                ),
                "debt_to_equity": snap.debt_to_equity,
                "current_ratio": snap.current_ratio,
                "price_to_book": snap.price_to_book,
                "investment_thesis": (stock.get("thesis") or "")[:500],
                "discovered_at": datetime.now(ny_timezone).isoformat(),
            }
            try:
                self.db.save_stock_find(find_data)
                saved += 1
                print(
                    f"  Saved {ticker} ({stock.get('company_name')}) "
                    f"conf={stock.get('confidence_score')} thesis={stock.get('thesis', '')[:80]}..."
                )
            except Exception as exc:
                print(f"  Failed to save {ticker}: {exc}")
        return saved

    def _format_funnel_report(
        self,
        stats: Dict[str, Any],
        candidates: List[FunnelCandidate],
        deep_dive: Optional[Dict[str, Any]],
        reddit_overlay: str,
        bottleneck_context: str,
    ) -> str:
        """Compose the final markdown report from funnel stats + deep-dive verdicts."""
        timestamp = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        prefs = self.preferences
        lines: List[str] = [
            "# Undervalued Stocks Analysis Report (Quant Funnel)",
            f"Generated: {timestamp}",
            "",
            "## Funnel Statistics",
        ]
        for k, v in stats.items():
            lines.append(f"- {k}: {v}")

        lines += [
            "",
            "## Screening Parameters",
            f"- Price range: ${prefs.min_price:.2f} - ${prefs.max_price:.2f}",
            f"- Min volume: {prefs.min_volume:,.0f}",
            f"- Max P/E: {prefs.max_pe}",
            f"- Min market cap: ${prefs.min_market_cap:,.0f}",
            f"- Min current ratio: {prefs.min_current_ratio}",
            f"- Max debt/equity: {prefs.max_debt_equity}",
            "",
            "## Shortlist (pre-deep-dive)",
        ]
        for c in candidates:
            s = c.snapshot
            pe = s.trailing_pe or s.forward_pe
            implied = c.dcf.by_discount_rate if c.dcf and not c.dcf.error else {}
            insider_net = c.insider.net_value_usd if c.insider and not c.insider.error else None
            lanes = ", ".join(c.discovery_lanes) or "unclassified"
            scarcity = c.scarcity.score if c.scarcity else 0.0
            lines.append(
                f"- **{s.symbol}** ({s.sector or 'Unknown'}) · price ${s.price:.2f} · "
                f"mcap ${(s.market_cap or 0)/1e9:.1f}B · P/E {pe:.1f} · "
                f"sector value {c.ranking.composite_value_score} · "
                f"implied growth @10% {implied.get('10%', 'n/a')} · "
                f"insider net 180d {('$%s' % f'{insider_net:,.0f}') if insider_net is not None else 'n/a'} · "
                f"lanes {lanes} · scarcity {scarcity:.2f} · "
                f"funnel score {c.composite_funnel_score}"
            )

        lines += [
            "",
            "## Cross-Sector Bottleneck Research",
            bottleneck_context,
            "",
            "## Reddit Catalyst Overlay (supporting evidence only)",
            reddit_overlay,
        ]

        if deep_dive and deep_dive.get("stocks"):
            lines += ["", "## Deep-Dive Verdicts"]
            for stock in deep_dive["stocks"]:
                verdict = stock.get("verdict", "?")
                marker = "✓" if verdict == "accept" else "✗"
                lines += [
                    "",
                    f"### {marker} {stock.get('ticker')} — {stock.get('company_name', '')} ({verdict.upper()})",
                    f"- **Confidence:** {stock.get('confidence_score')}",
                    f"- **Thesis:** {stock.get('thesis', '')}",
                    f"- **Catalyst:** {stock.get('primary_catalyst', '')}",
                    f"- **Risks:** {'; '.join(stock.get('key_risks', []))}",
                    f"- **Entry:** {stock.get('entry_strategy', '')}",
                    f"- **Stop loss:** {stock.get('stop_loss_pct', 'n/a')}",
                    f"- **Size:** {stock.get('position_size_pct', 'n/a')}",
                ]
                if verdict == "reject":
                    lines.append(f"- **Rejection reason:** {stock.get('rejection_reason', '')}")
        else:
            lines += ["", "## Deep-Dive Verdicts", "_Deep-dive output unavailable or unparseable._"]

        return "\n".join(lines)

    async def execute_undervalued_analysis(
        self, universe: Optional[List[Ticker]] = None
    ) -> str:
        """Funnel-first flow: deterministic stages 1-5 then LLM thesis-writing on shortlist."""
        log_file_path = self._configure_agno_debug_logging()
        print(f"Agno debug logs: {log_file_path}")

        self._run_startup_ritual()

        self.current_run_id = self.db.create_analysis_run(
            run_type="undervalued",
            preferences=self.preferences.__dict__,
        )
        print(f"\nStarted funnel run #{self.current_run_id}")
        write_state(
            "undervalued",
            run_id=self.current_run_id,
            phase="started",
            preferences=self.preferences.model_dump(),
            mode="quant_funnel",
        )

        try:
            print("\n=== Stage 1A: Cross-Sector Bottleneck Research (web) ===")
            bottleneck_research = await self._research_bottlenecks()
            research_candidates = {
                candidate["ticker"]: candidate
                for candidate in (bottleneck_research or {}).get("candidates", [])
            }
            base_universe = list(universe) if universe is not None else load_universe()
            augmented_universe = self._augment_universe_with_research(
                base_universe, bottleneck_research
            )
            print(
                f"  Research found {len(research_candidates)} public candidates; "
                f"scanning {len(augmented_universe) - len(base_universe)} outside "
                "the base universe."
            )

            # Stages 1-5: deterministic quant funnel
            print("\n=== Stages 1-5: Quant Funnel (deterministic) ===")
            funnel = QuantFunnel(
                gates=self._preferences_to_gates(),
                top_n_for_dcf=30,
                top_n_for_insider=20,
                top_n_final=10,
                workers=5,
                universe=augmented_universe,
                bottleneck_research=research_candidates,
            )
            result = funnel.run()
            shortlist: List[FunnelCandidate] = result["candidates"]
            stats: Dict[str, Any] = result["stats"]
            stats["stage_1_base_universe_size"] = len(base_universe)
            stats["stage_1_research_candidates"] = len(research_candidates)
            stats["stage_1_research_tickers_added"] = (
                len(augmented_universe) - len(base_universe)
            )
            stats["bottleneck_research_status"] = (
                "loaded" if bottleneck_research else "unavailable"
            )
            funnel_map = {c.symbol: c for c in shortlist}

            print(f"\nFunnel produced {len(shortlist)} candidates.")
            write_state(
                "undervalued",
                run_id=self.current_run_id,
                phase="funnel_complete",
                shortlist_size=len(shortlist),
                stats=stats,
            )

            # Lightweight Reddit overlay on the shortlist only
            print("\nFetching Reddit catalyst overlay for shortlist...")
            reddit_overlay = self._get_reddit_overlay(shortlist)
            bottleneck_context = self._format_bottleneck_context(
                bottleneck_research, shortlist
            )

            # Stage 6: Agent deep-dive with structured JSON output
            print("\n=== Stage 6: Agent Deep-Dive (LLM narrative) ===")
            deep_dive_dict: Optional[Dict[str, Any]] = None
            if shortlist:
                prompt = self._build_funnel_deep_dive_prompt(
                    shortlist, stats, reddit_overlay, bottleneck_context
                )
                self._log_prompt_size("Deep-dive prompt", prompt)
                response = await self.deep_dive_agent.arun(prompt)
                raw_text = self._extract_content(response)
                deep_dive_dict = self._parse_deep_dive_json(
                    raw_text, [candidate.symbol for candidate in shortlist]
                )
                if deep_dive_dict is None:
                    await self.save_phase_output(
                        "deep_dive_raw",
                        raw_text,
                        "Raw deep-dive output (JSON parse failed)",
                    )
                    self._require_deep_dive(deep_dive_dict, shortlist)
                else:
                    await self.save_phase_output(
                        "deep_dive",
                        json.dumps(deep_dive_dict, indent=2, default=str),
                        "Structured deep-dive verdicts",
                    )
            else:
                print("  Empty shortlist — skipping agent deep-dive.")

            # Persist accepted picks
            saved_count = 0
            if deep_dive_dict:
                saved_count = await self._save_funnel_stocks(deep_dive_dict, funnel_map)

            # Compose + save final report
            final_report = self._format_funnel_report(
                stats,
                shortlist,
                deep_dive_dict,
                reddit_overlay,
                bottleneck_context,
            )
            await self.save_phase_output(
                "final_report",
                final_report,
                "Funnel-first undervalued analysis",
            )

            self.db.add_research_note(
                ticker="ANALYSIS_RUN",
                note_type="funnel_report",
                content=final_report[:5000],
                source="quant_funnel",
            )

            self.db.complete_analysis_run(
                run_id=self.current_run_id,
                total_candidates=len(shortlist),
                final_selections=saved_count,
                status="completed",
            )
            print(
                f"\nFunnel run #{self.current_run_id} complete. "
                f"Shortlist={len(shortlist)} Saved={saved_count}"
            )

            await self._trigger_portfolio_tracking(self.current_run_id)
            clear_state("undervalued")
            return final_report

        except asyncio.CancelledError as exc:
            self._persist_failed_run(exc, phase="cancelled")
            raise
        except Exception as exc:
            self._persist_failed_run(exc)
            raise

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

    def _sanitize_agent_output(self, content: str) -> str:
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

    def _build_image_payload(self, limit: int = 8) -> List[Image]:
        """Convert stored chart paths into Agno Image attachments."""
        images: List[Image] = []
        for path in self._chart_paths[-limit:]:
            try:
                if os.path.exists(path):
                    images.append(Image(path=path))
            except Exception:
                continue
        return images

    def _register_chart_path(self, path: str) -> None:
        """Track generated chart images for downstream agents."""
        if path and path not in self._chart_paths:
            self._chart_paths.append(path)

    def _fetch_top_comments(self, permalink: str, limit: int = 3) -> List[str]:
        """Retrieve top Reddit comments for a given post."""
        comments: List[str] = []
        if not permalink:
            return comments
        try:
            response = self._reddit_client.get(permalink, timeout=15)
            response.raise_for_status()
            threads = response.json()
            if len(threads) > 1:
                comment_items = threads[1].get("data", {}).get("children", [])
                for child in comment_items:
                    if child.get("kind") != "t1":
                        continue
                    body = child.get("data", {}).get("body")
                    if body:
                        comments.append(body[:500])
                    if len(comments) >= limit:
                        break
        except Exception:
            return comments
        return comments

    def _get_reddit_discovery_summary(self) -> str:
        """Fetch discovery data and return the LLM summary."""
        try:
            discovery_json = self.discover_reddit_tickers()
            payload = json.loads(discovery_json)
            summary = payload.get("batch_summaries")
            if summary:
                combined = "\n\n".join(
                    f"Batch {item.get('batch')}: {item.get('summary')}"
                    for item in summary
                    if item.get("summary")
                )
                if combined:
                    return self._truncate_for_context(
                        combined, self.MAX_REDDIT_SUMMARY_TOKENS
                    )
            llm_summary = payload.get("llm_summary")
            if llm_summary:
                return self._truncate_for_context(
                    llm_summary, self.MAX_REDDIT_SUMMARY_TOKENS
                )
            return self._truncate_for_context(
                discovery_json, self.MAX_REDDIT_SUMMARY_TOKENS
            )
        except Exception:
            return ""

    def _truncate_for_context(self, text: str, limit_tokens: int) -> str:
        """Ensure downstream prompts stay within token limits."""
        if not text:
            return ""

        if self._token_encoder is None:
            approx_chars = limit_tokens * 4
            if len(text) <= approx_chars:
                return text
            notice = (
                f"\n\n[Truncated screening summary to ~{approx_chars:,} characters "
                "because tokenizer data was unavailable. Refocus on the sections above.]"
            )
            return text[:approx_chars] + notice

        tokens = self._token_encoder.encode(text)
        token_length = len(tokens)
        if token_length <= limit_tokens:
            return text

        truncated = self._token_encoder.decode(tokens[:limit_tokens])
        notice = (
            f"\n\n[Compressed screening summary from {token_length:,} tokens "
            f"to {limit_tokens:,} tokens to stay within the model's context window.]"
        )
        return truncated + notice

    def _log_prompt_size(self, label: str, text: str) -> None:
        """Emit token counts for large prompt sections into the run log."""
        token_count = self._count_tokens(text)
        message = f"{label} token count: {token_count:,}"
        if self.logger is not None:
            self.logger.debug(message)
        else:
            print(message)

    def _build_learning_insights_section(self) -> tuple[str, str]:
        """Build learning insights section and confidence adjustments from past performance.

        Returns:
            tuple[str, str]: (learning_insights_section, confidence_adjustments)
        """
        try:
            # Get recent learning insights from performance database
            insights_list = self.perf_db.get_recent_learning_insights(
                days=90, min_confidence="medium"
            )

            if not insights_list:
                return (
                    "",
                    "(add +1.5 for TSX stocks with Reddit validation, +0.5 for strong hiring)",
                )

            # Build the learning insights section
            section_parts = ["\n## SELF-IMPROVEMENT: Learning from Past Performance\n"]
            section_parts.append(
                "**Apply these data-driven insights to improve stock selection:**\n"
            )

            # Get detailed stats
            conf_stats = self.perf_db.get_confidence_calibration_stats(days=90)
            source_stats = self.perf_db.get_source_performance_stats(days=90)
            sector_stats = self.perf_db.get_sector_performance_stats(days=90)
            catalyst_stats = (
                self.perf_db.get_catalyst_stats(days=90)
                if hasattr(self.perf_db, "get_catalyst_stats")
                else {}
            )

            # Confidence Calibration
            if conf_stats and conf_stats.get("total_analyzed", 0) > 0:
                high_ret = conf_stats.get("high_conf_avg_return") or 0
                med_ret = conf_stats.get("med_conf_avg_return") or 0
                section_parts.append(
                    f"\n### Confidence Calibration (based on {conf_stats['total_analyzed']} past picks)"
                )
                section_parts.append(
                    f"- High confidence (>8.0) picks: **{high_ret:+.1f}%** avg return"
                )
                section_parts.append(
                    f"- Medium confidence (5-8) picks: **{med_ret:+.1f}%** avg return"
                )
                if high_ret > med_ret + 5:
                    section_parts.append(
                        "- **Insight**: High confidence scoring is well-calibrated. Trust strong convictions."
                    )
                elif med_ret > high_ret:
                    section_parts.append(
                        "- **Insight**: Medium confidence picks are outperforming. Be more conservative with high scores."
                    )

            # Source Reliability
            if source_stats:
                section_parts.append("\n### Source Performance")
                for source in source_stats[:3]:  # Top 3 sources
                    section_parts.append(
                        f"- **{source['discovery_source']}**: {source['avg_return']:+.1f}% avg return, "
                        f"{source['win_rate']:.0f}% win rate ({source['pick_count']} picks)"
                    )

            # Sector Performance
            if sector_stats:
                section_parts.append("\n### Top Performing Sectors (Last 90 Days)")
                for sector in sector_stats[:3]:  # Top 3 sectors
                    section_parts.append(
                        f"- **{sector['sector']}**: {sector['avg_return']:+.1f}% avg return ({sector['pick_count']} picks)"
                    )
                if sector_stats:
                    section_parts.append(
                        f"- **Insight**: Prioritize {sector_stats[0]['sector']} sector opportunities."
                    )

            # Catalyst Accuracy
            if catalyst_stats and catalyst_stats.get("total_catalysts", 0) > 0:
                section_parts.append("\n### Catalyst Realization Rates")
                for cat_type, stats in catalyst_stats.get("by_type", {}).items():
                    if stats.get("total", 0) > 0:
                        rate = stats.get("realization_rate", 0) * 100
                        impact = stats.get("avg_impact", 0)
                        section_parts.append(
                            f"- **{cat_type}**: {rate:.0f}% realization, {impact:+.1f}% avg price impact"
                        )

            # Actionable Recommendations from stored insights
            recommendations = [
                i for i in insights_list if i.get("actionable_recommendation")
            ]
            if recommendations:
                section_parts.append("\n### Active Recommendations")
                for rec in recommendations[:3]:
                    section_parts.append(f"- {rec['actionable_recommendation']}")

            section_parts.append(
                "\n**Apply these learnings when scoring candidates.**\n"
            )

            # Build dynamic confidence adjustments
            conf_adjustments = self._build_confidence_adjustments(
                source_stats, sector_stats
            )

            return ("\n".join(section_parts), conf_adjustments)

        except Exception as e:
            print(f"  Warning: Could not load learning insights: {e}")
            return (
                "",
                "(add +1.5 for TSX stocks with Reddit validation, +0.5 for strong hiring)",
            )

    def _build_confidence_adjustments(
        self, source_stats: list, sector_stats: list
    ) -> str:
        """Build dynamic confidence adjustment rules based on historical performance."""
        adjustments = ["(scoring adjustments based on past performance:"]

        # Base adjustments
        adjustments.append("+1.5 for TSX stocks with Reddit validation")
        adjustments.append("+0.5 for strong hiring activity")

        # Source-based adjustments
        if source_stats:
            best_source = source_stats[0] if source_stats else None
            if best_source and best_source.get("win_rate", 0) > 70:
                adjustments.append(
                    f"+0.5 for {best_source['discovery_source']} discoveries (70%+ win rate)"
                )

        # Sector-based adjustments
        if sector_stats:
            best_sector = sector_stats[0] if sector_stats else None
            worst_sector = sector_stats[-1] if len(sector_stats) > 2 else None
            if best_sector and best_sector.get("avg_return", 0) > 15:
                adjustments.append(
                    f"+0.5 for {best_sector['sector']} sector (top performer)"
                )
            if worst_sector and worst_sector.get("avg_return", 0) < -5:
                adjustments.append(
                    f"-0.5 for {worst_sector['sector']} sector (underperforming)"
                )

        adjustments.append(")")
        return " ".join(adjustments)

    def _build_screening_prompt(self, reddit_summary: str | None = None) -> str:
        """Generate the detailed screening instructions for the agent."""
        now = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        prefs = self.preferences

        reddit_section = ""
        if reddit_summary:
            reddit_summary = self._truncate_for_context(
                reddit_summary, self.MAX_REDDIT_SUMMARY_TOKENS
            )
            reddit_section = f"\n\n## PRIORITY INTELLIGENCE: Recent Reddit Community Insights\n{reddit_summary}\n\n**CRITICAL**: Prioritize TSX-listed stocks and Canadian equities mentioned in r/Baystreetbets. These community-identified opportunities should be thoroughly investigated first, as they represent emerging value plays with potential retail momentum. Cross-reference Reddit catalysts with fundamental data to validate investment thesis.\n"

        # Get learning insights from past performance
        learning_insights_section, confidence_adjustments = (
            self._build_learning_insights_section()
        )
        if learning_insights_section:
            learning_insights_section = self._truncate_for_context(
                learning_insights_section, self.MAX_LEARNING_INSIGHTS_TOKENS
            )

        template = load_prompt("screening_prompt")
        prompt = template.format(
            timestamp=now,
            reddit_section=reddit_section,
            learning_insights_section=learning_insights_section,
            confidence_adjustments=confidence_adjustments,
            max_price=f"{prefs.max_price:.2f}",
            min_price=f"{prefs.min_price:.2f}",
            min_volume=f"{prefs.min_volume:,.0f}",
            price_vs_high_pct=f"{prefs.price_vs_high * 100:.0f}",
            max_pe=f"{prefs.max_pe:.1f}",
            min_market_cap=f"{prefs.min_market_cap:,.0f}",
            min_current_ratio=f"{prefs.min_current_ratio:.2f}",
            max_debt_equity=f"{prefs.max_debt_equity:.2f}",
        )
        if self._count_tokens(prompt) <= self.MAX_SCREENING_PROMPT_TOKENS:
            return prompt

        prompt = template.format(
            timestamp=now,
            reddit_section="",
            learning_insights_section=learning_insights_section,
            confidence_adjustments=confidence_adjustments,
            max_price=f"{prefs.max_price:.2f}",
            min_price=f"{prefs.min_price:.2f}",
            min_volume=f"{prefs.min_volume:,.0f}",
            price_vs_high_pct=f"{prefs.price_vs_high * 100:.0f}",
            max_pe=f"{prefs.max_pe:.1f}",
            min_market_cap=f"{prefs.min_market_cap:,.0f}",
            min_current_ratio=f"{prefs.min_current_ratio:.2f}",
            max_debt_equity=f"{prefs.max_debt_equity:.2f}",
        )
        if self._count_tokens(prompt) <= self.MAX_SCREENING_PROMPT_TOKENS:
            return prompt

        return template.format(
            timestamp=now,
            reddit_section="",
            learning_insights_section="",
            confidence_adjustments=confidence_adjustments,
            max_price=f"{prefs.max_price:.2f}",
            min_price=f"{prefs.min_price:.2f}",
            min_volume=f"{prefs.min_volume:,.0f}",
            price_vs_high_pct=f"{prefs.price_vs_high * 100:.0f}",
            max_pe=f"{prefs.max_pe:.1f}",
            min_market_cap=f"{prefs.min_market_cap:,.0f}",
            min_current_ratio=f"{prefs.min_current_ratio:.2f}",
            max_debt_equity=f"{prefs.max_debt_equity:.2f}",
        )

    def _build_turnaround_prompt(self, screening_summary: str) -> str:
        """Build the second-phase prompt referencing screening output."""
        now = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")

        # Get learning insights (reuse same method, but only use the section part)
        learning_insights_section, _ = self._build_learning_insights_section()
        if learning_insights_section:
            learning_insights_section = self._truncate_for_context(
                learning_insights_section, self.MAX_LEARNING_INSIGHTS_TOKENS
            )
        screening_summary = self._truncate_for_context(
            screening_summary, self.SCREENING_TOKEN_LIMIT
        )

        template = load_prompt("turnaround_prompt")
        return template.format(
            screening_summary=screening_summary,
            timestamp=now,
            learning_insights_section=learning_insights_section,
        )

    def create_final_report(self, screening_data: str, turnaround_data: str) -> str:
        """Combine both analysis stages into a unified report."""
        timestamp = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        prefs = self.preferences
        return f"""# Undervalued Stocks Analysis Report
Generated Time : {timestamp}

## Executive Summary
This analysis identifies fundamentally strong but underperforming stocks under ${prefs.max_price:.2f}
with significant recovery potential.

## Screening Criteria
- Price Range: ${prefs.min_price:.2f} - ${prefs.max_price:.2f}
- Minimum Volume: {prefs.min_volume:,.0f} shares/day
- Maximum P/E: {prefs.max_pe:.1f}
- Minimum Market Cap: ${prefs.min_market_cap:,.0f}

## Initial Screening Results
{screening_data}

## Turnaround Analysis
{turnaround_data}

## Implementation Strategy
1. Position Sizing
- Initial Position: 3-4% of portfolio
- Maximum Position: 5% at cost
- Average Down Strategy: Add 1% at 10% below entry

2. Risk Management
- Stop Loss: 25% below entry
- Position Review: Any fundamental thesis violation
- Catalyst Monitoring: Quarterly review of turnaround progress

3. Profit Taking
- First Target: 25% of position at 50% gain
- Second Target: 25% of position at 100% gain
- Hold Remainder: Until thesis completion or violation

## Monitoring Guidelines
1. Weekly:
- Price and volume action
- News and catalyst updates
- Insider activity

2. Monthly:
- Technical trend analysis
- Industry group strength
- Institutional ownership changes

3. Quarterly:
- Financial results review
- Management execution analysis
- Catalyst progression
- Thesis validation

## Risk Warnings
- Use strict position sizing
- Monitor stop levels
- Review thesis quarterly
- Track catalyst timeline
"""

    async def save_phase_output(self, phase: str, content: str, description: str):
        """Persist each analysis phase as markdown."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{phase}_{timestamp}.md"

        cleaned_content = content.strip()
        if cleaned_content.startswith("```markdown"):
            cleaned_content = cleaned_content[len("```markdown") :].strip()
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[len("```") :].strip()
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3].strip()

        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"# {description}\n")
            file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            file.write(cleaned_content)


async def get_value_preferences() -> ValueScreeningPreferences:
    """Get randomized value screening preferences within reasonable ranges."""
    import random

    ranges = {
        "max_price": (50, 200),
        "min_price": (2, 10),
        "min_volume": (300000, 1000000),
        "max_pe": (15, 30),
        "min_market_cap_millions": (100, 500),
        "min_current_ratio": (1.2, 2.0),
        "max_debt_equity": (1.5, 2.5),
        "price_vs_high_percent": (30, 50),
    }

    max_price = random.uniform(*ranges["max_price"])
    min_price = random.uniform(*ranges["min_price"])
    min_volume = random.uniform(*ranges["min_volume"])
    max_pe = random.uniform(*ranges["max_pe"])
    min_market_cap = random.uniform(*ranges["min_market_cap_millions"]) * 1_000_000
    min_current_ratio = random.uniform(*ranges["min_current_ratio"])
    max_debt_equity = 1.5
    price_vs_high = random.uniform(*ranges["price_vs_high_percent"]) / 100

    print("\n=== Randomly Selected Screening Parameters ===")
    print(f"Maximum Price: ${max_price:.2f}")
    print(f"Minimum Price: ${min_price:.2f}")
    print(f"Minimum Volume: {min_volume:,.0f}")
    print(f"Maximum P/E: {max_pe:.1f}")
    print(f"Minimum Market Cap: ${min_market_cap / 1_000_000:.1f}M")
    print(f"Minimum Current Ratio: {min_current_ratio:.2f}")
    print(f"Maximum Debt/Equity: {max_debt_equity:.2f}")
    print(f"Maximum Decline from 52-week High: {price_vs_high * 100:.1f}%\n")

    return ValueScreeningPreferences(
        max_price=max_price,
        min_price=min_price,
        min_volume=min_volume,
        max_pe=max_pe,
        min_market_cap=min_market_cap,
        min_current_ratio=min_current_ratio,
        max_debt_equity=max_debt_equity,
        price_vs_high=price_vs_high,
    )


async def main():
    """Execute the undervalued stock analysis."""
    try:
        print("\n=== Starting Analysis Run ===")
        preferences = await get_value_preferences()
        analysis_flow = UndervaluedAnalysisFlow(preferences)
        final_report = await analysis_flow.execute_undervalued_analysis()

        print("\nRun completed successfully!")
        print(f"Results saved in: {analysis_flow.output_dir}")
        return final_report
    except Exception as exc:  # pragma: no cover - integration runtime
        print(f"\nError during analysis: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
