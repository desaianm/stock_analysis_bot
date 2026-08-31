"""Resolve a company name to a ticker + summary using a small Agno agent."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from stockbot.tools.data import TavilySearchTool, WebSearchTool
from stockbot.tickers import normalize_ticker

EXTRACTION_MODEL_ID = "gpt-4.1-nano"


def _load_instructions() -> str:
    path = (
        Path(__file__).parent.parent.parent
        / "prompts"
        / "agents"
        / "company_lookup_agent.txt"
    )
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return (
        "You are a Company Lookup Agent. Resolve a company name to its primary "
        "exchange-listed ticker and provide a one-paragraph business summary."
    )


_tavily_tool = TavilySearchTool()
_web_search_tool = WebSearchTool()


def _search_tavily(query: str) -> str:
    try:
        results = _tavily_tool.run(query)
    except Exception as exc:
        return json.dumps({"source": "tavily", "error": str(exc)})
    return json.dumps({"source": "tavily", "results": results[:8] if isinstance(results, list) else results}, default=str)


def _search_web(query: str) -> str:
    try:
        results = _web_search_tool.run(query)
    except Exception as exc:
        return json.dumps({"source": "web", "error": str(exc)})
    return json.dumps({"source": "web", "results": results[:8] if isinstance(results, list) else results}, default=str)


_lookup_agent = Agent(
    name="Company Lookup Agent",
    model=OpenAIChat(id=EXTRACTION_MODEL_ID, temperature=1, max_completion_tokens=2000),
    instructions=[
        _load_instructions(),
        "",
        "When given a company name, return a JSON object with EXACTLY these keys:",
        '  {"ticker": "<symbol>", "company_name": "<full name>", "company_info": "<one paragraph>"}',
        "Use the search tools to find the correct primary exchange ticker.",
        "For dual-listed companies, prefer NYSE/NASDAQ. For Canadian-only listings use the .TO suffix.",
        "Respond with the JSON object only, no commentary, no markdown fences.",
    ],
    tools=[_search_tavily, _search_web],
    markdown=False,
    add_datetime_to_context=True,
    timezone_identifier="America/New_York",
)


def _parse_json_payload(text: str) -> Optional[Dict[str, str]]:
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
    if isinstance(payload, dict) and "ticker" in payload:
        try:
            ticker = normalize_ticker(str(payload.get("ticker", "")))
        except ValueError:
            return None
        return {
            "ticker": ticker,
            "company_name": str(payload.get("company_name", "")).strip(),
            "company_info": str(payload.get("company_info", "")).strip(),
        }
    return None


async def lookup_company(company_name: str) -> Optional[Dict[str, str]]:
    """Return {ticker, company_name, company_info} or None if not found."""
    if not company_name or not company_name.strip():
        return None

    prompt = f"Find the publicly traded ticker for: {company_name.strip()}"
    response = await _lookup_agent.arun(prompt)
    content = getattr(response, "content", None) or str(response)
    return _parse_json_payload(content)
