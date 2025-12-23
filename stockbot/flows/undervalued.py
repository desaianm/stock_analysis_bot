"""Undervalued stock analysis flow powered by Agno agents and lightweight tools."""

from __future__ import annotations

import asyncio
import json
import os
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

from stockbot.tools.data import (
    ChartingTool,
    CompanyInfoTool,
    FinancialReportTool,
    OptionsChainTool,
    RealTimeQuoteTool,
    StockNewsTool,
    StockPriceDataTool,
    TavilySearchTool,
    WebSearchTool
)

ny_timezone = pytz.timezone("America/New_York")


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "undervalued" / f"{prompt_name}.txt"
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
    min_volume: float = Field(default=500000, description="Minimum daily trading volume")
    max_pe: float = Field(default=25.0, description="Maximum P/E ratio")
    min_market_cap: float = Field(
        default=300000000, description="Minimum market cap ($300M)"
    )
    min_current_ratio: float = Field(default=1.5, description="Minimum current ratio")
    max_debt_equity: float = Field(
        default=2.0, description="Maximum debt/equity ratio"
    )
    price_vs_high: float = Field(
        default=0.4, description="Maximum decline from 52-week high (40%)"
    )


class UndervaluedAnalysisFlow:
    """Coordinates the undervalued stock analysis using an Agno agent."""

    SCREENING_TOKEN_LIMIT = 900_000

    def __init__(self, preferences: ValueScreeningPreferences):
        self.preferences = preferences
        self.output_dir = Path("outputs/undervalued_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = "gpt-4.1"

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
        self._reddit_headers = {"User-Agent": "stock-analysis-bot/1.0"}

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
        ]
        shared_model = OpenAIChat(
            id=self.model_id,
            temperature=1,
            max_completion_tokens=10000,
        )

        screening_instructions = load_prompt("screening_agent_instructions").strip().split('\n')
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

        turnaround_instructions = load_prompt("turnaround_agent_instructions").strip().split('\n')
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

        reddit_instructions = load_prompt("reddit_sentiment_agent_instructions").strip().split('\n')
        self.reddit_sentiment_agent = Agent(
            name="BayStreet Reddit Scout",
            model=OpenAIChat(id="gpt-4.1", temperature=1, max_completion_tokens=5000),
            instructions=reddit_instructions,
            tools=[self.reddit_sentiment_scan],
            markdown=True,
            add_datetime_to_context=True,
            timezone_identifier="America/New_York",
            debug_mode=True,
        )

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

    def get_stock_price_history(self, symbol: str, period: str = "1y") -> str:
        """Return historical closing prices for a symbol and period."""
        try:
            prices = self._stock_price_tool.run(symbol, period)
        except Exception as exc:  # pragma: no cover - depends on API
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
        except Exception as exc:  # pragma: no cover - depends on API
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
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error("get_financial_facts", symbol=symbol, error=str(exc))

        metrics_to_track = [
            "revenue",
            "net_income",
            "operating_cash_flow",
            "free_cash_flow",
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
            "condensed_metrics": metric_summaries,
            "note": "Raw historical tables trimmed to last 3 periods per metric to control token usage.",
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def get_real_time_quote(self, symbol: str) -> str:
        """Fetch intraday price, volume, and market cap."""
        try:
            price, volume, market_cap = self._real_time_quote_tool.run(symbol)
        except Exception as exc:  # pragma: no cover - depends on API
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
            return self._format_error("search_market_events", query=query, error=str(exc))

        payload = {
            "query": query,
            "results": results,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def search_global_research(self, query: str) -> str:
        """Use Tavily to capture broader macro or thematic research."""
        try:
            results = self._tavily_tool.run(query)
        except Exception as exc:  # pragma: no cover - depends on API
            return self._format_error("search_global_research", query=query, error=str(exc))

        payload = {
            "query": query,
            "results": results,
            "retrieved_at": datetime.now(ny_timezone).isoformat(),
        }
        return self._format_json(payload)

    def reddit_sentiment_scan(self, query: str, max_posts: int = 30) -> str:
        """Collect Reddit posts for the LLM to analyze (no heuristic scoring)."""
        subs = ["Baystreetbets", "wallstreetbets"]
        aggregated_posts: List[Dict[str, Any]] = []
        normalized_query = query.upper()

        for subreddit in subs:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": f"{normalized_query} TSX" if subreddit == "Baystreetbets" else normalized_query,
                "restrict_sr": "1",
                "sort": "new",
                "limit": str(max_posts // len(subs)),
                "t": "week",
            }
            try:
                response = requests.get(
                    url, headers=self._reddit_headers, params=params, timeout=15
                )
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
            url = f"https://www.reddit.com/r/{subreddit}/new.json"
            params = {"limit": str(max_posts // len(subs))}
            try:
                response = requests.get(
                    url, headers=self._reddit_headers, params=params, timeout=15
                )
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
                summary_model = OpenAIChat(id="gpt-4.1", temperature=0.4, max_completion_tokens=2000)
                summarizer = Agent(model=summary_model, markdown=True)
                for idx in range(0, len(collected_posts), 50):
                    batch = collected_posts[idx : idx + 50]
                    batch_payload = json.dumps(batch, indent=2)
                    summary = summarizer.run(
                        f"Summarize Reddit posts batch {idx // 50 + 1}. Highlight emerging tickers, "
                        "catalysts, sentiment, and risks.\n\n"
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
            "calls": calls,
            "puts": puts,
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
    # Analysis workflow
    # -------------------------------------------------------------------------
    async def run_value_screening(self) -> str:
        """Execute only the initial screening prompt."""
        reddit_summary = self._get_reddit_discovery_summary()
        screening_prompt = self._build_screening_prompt(reddit_summary)
        screening_result = await self.screening_agent.arun(
            screening_prompt,
            images=self._build_image_payload(),
        )
        return self._extract_content(screening_result)

    async def execute_undervalued_analysis(self) -> str:
        """Run the screening and turnaround prompts sequentially."""
        print("\nExecuting undervalued stock screening with Agno...")
        screening_raw = await self.run_value_screening()
        sanitized_screening = self._sanitize_agent_output(screening_raw)
        screening_context = self._truncate_for_context(
            sanitized_screening, self.SCREENING_TOKEN_LIMIT
        )
        await self.save_phase_output(
            "initial_screening",
            sanitized_screening,
            "Initial value stock screening results",
        )

        print("\nAnalyzing turnaround potential...")
        turnaround_prompt = self._build_turnaround_prompt(screening_context)
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

        final_report = self.create_final_report(sanitized_screening, turnaround_content)
        await self.save_phase_output(
            "final_report",
            final_report,
            "Final undervalued stocks analysis",
        )
        return final_report

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
            url = f"{permalink}.json"
            response = requests.get(url, headers=self._reddit_headers, timeout=15)
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
                    return combined
            llm_summary = payload.get("llm_summary")
            if llm_summary:
                return llm_summary
            return discovery_json
        except Exception:
            return ""

    def _build_image_payload(self, limit: int = 8) -> List[Image]:
        """Convert stored chart paths into Agno Image objects."""
        images: List[Image] = []
        for path in self._chart_paths[-limit:]:
            try:
                if os.path.exists(path):
                    images.append(Image(path=path))
            except Exception:
                continue
        return images

    def _register_chart_path(self, path: str) -> None:
        """Track chart files so they can be attached to future prompts."""
        if path and path not in self._chart_paths:
            self._chart_paths.append(path)

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

    def _build_screening_prompt(self, reddit_summary: str | None = None) -> str:
        """Generate the detailed screening instructions for the agent."""
        now = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        prefs = self.preferences

        reddit_section = ""
        if reddit_summary:
            reddit_section = f"\n\n## PRIORITY INTELLIGENCE: Recent Reddit Community Insights\n{reddit_summary}\n\n**CRITICAL**: Prioritize TSX-listed stocks and Canadian equities mentioned in r/Baystreetbets. These community-identified opportunities should be thoroughly investigated first, as they represent emerging value plays with potential retail momentum. Cross-reference Reddit catalysts with fundamental data to validate investment thesis.\n"

        template = load_prompt("screening_prompt")
        return template.format(
            timestamp=now,
            reddit_section=reddit_section,
            max_price=f"{prefs.max_price:.2f}",
            min_price=f"{prefs.min_price:.2f}",
            min_volume=f"{prefs.min_volume:,.0f}",
            price_vs_high_pct=f"{prefs.price_vs_high * 100:.0f}",
            max_pe=f"{prefs.max_pe:.1f}",
            min_market_cap=f"{prefs.min_market_cap:,.0f}",
            min_current_ratio=f"{prefs.min_current_ratio:.2f}",
            max_debt_equity=f"{prefs.max_debt_equity:.2f}"
        )

    def _build_turnaround_prompt(self, screening_summary: str) -> str:
        """Build the second-phase prompt referencing screening output."""
        now = datetime.now(ny_timezone).strftime("%Y-%m-%d %H:%M:%S")
        template = load_prompt("turnaround_prompt")
        return template.format(
            screening_summary=screening_summary,
            timestamp=now
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
    print(f"Minimum Market Cap: ${min_market_cap/1_000_000:.1f}M")
    print(f"Minimum Current Ratio: {min_current_ratio:.2f}")
    print(f"Maximum Debt/Equity: {max_debt_equity:.2f}")
    print(f"Maximum Decline from 52-week High: {price_vs_high*100:.1f}%\n")

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
