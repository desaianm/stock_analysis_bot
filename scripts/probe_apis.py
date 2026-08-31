"""Safely diagnose external services used by Stockbot."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class ProbeResult:
    service: str
    required: bool
    configured: bool
    status: str
    http_status: Optional[int]
    secret_name: Optional[str]


def classify_response(status_code: int, body: str = "") -> str:
    lowered = body.lower()
    if 200 <= status_code < 300:
        return "working"
    if status_code in {401, 403}:
        return "auth_or_permission"
    if status_code == 402 or any(
        marker in lowered
        for marker in ("quota", "billing", "payment required", "insufficient_credit")
    ):
        return "quota_or_billing"
    if status_code == 429:
        return "rate_limit"
    return "network_or_endpoint_failure"


def safe_report(
    results: list[ProbeResult], configured_values: Optional[Mapping[str, str]] = None
) -> dict[str, Any]:
    """Return report fields only; secret values are intentionally discarded."""
    del configured_values
    return {"services": [asdict(result) for result in results]}


def report_exit_code(results: list[ProbeResult]) -> int:
    return int(any(
        result.required and result.status != "working"
        for result in results
    ))


def validate_payload(service: str, payload: Any) -> bool:
    """Confirm that a successful HTTP response has the service's expected shape."""
    if not isinstance(payload, dict):
        return False
    if service == "OpenAI":
        output = payload.get("output")
        return (
            isinstance(payload.get("id"), str)
            and payload.get("status") == "completed"
            and payload.get("error") is None
            and isinstance(output, list)
            and any(
                isinstance(item, dict)
                and item.get("type") == "message"
                and item.get("role") == "assistant"
                and isinstance(item.get("content"), list)
                and any(
                    isinstance(content, dict)
                    and content.get("type") == "output_text"
                    and isinstance(content.get("text"), str)
                    and bool(content["text"].strip())
                    for content in item["content"]
                )
                for item in output
            )
        )
    if service == "Financial Datasets":
        return isinstance(payload.get("insider_trades"), list)
    if service in {"Tavily", "EXA"}:
        return isinstance(payload.get("results"), list)
    if service == "Discord":
        return isinstance(payload.get("id"), str) and isinstance(payload.get("username"), str)
    if service == "Reddit token":
        return (
            isinstance(payload.get("access_token"), str)
            and bool(payload["access_token"])
            and payload.get("token_type") == "bearer"
            and isinstance(payload.get("expires_in"), (int, float))
        )
    if service == "Reddit listing":
        data = payload.get("data")
        return payload.get("kind") == "Listing" and isinstance(data, dict) and isinstance(data.get("children"), list)
    if service == "Yahoo Finance":
        chart = payload.get("chart")
        return (
            isinstance(chart, dict)
            and chart.get("error") is None
            and isinstance(chart.get("result"), list)
            and bool(chart["result"])
        )
    return False


def _from_response(service: str, required: bool, configured: bool,
                   secret_name: Optional[str], response: Any,
                   validator_service: Optional[str] = None) -> ProbeResult:
    status = classify_response(response.status_code, getattr(response, "text", ""))
    if status == "working":
        try:
            valid = validate_payload(validator_service or service, response.json())
        except (TypeError, ValueError):
            valid = False
        if not valid:
            status = "invalid_payload"
    return ProbeResult(
        service, required, configured,
        status,
        response.status_code, secret_name,
    )


def _probe(session: Any, service: str, required: bool, configured: bool,
           secret_name: Optional[str], method: str, url: str, **kwargs: Any) -> ProbeResult:
    if not configured:
        return ProbeResult(service, required, False, "not_configured", None, secret_name)
    try:
        response = session.request(method, url, timeout=15, **kwargs)
        return _from_response(service, required, True, secret_name, response)
    except requests.RequestException:
        return ProbeResult(service, required, True, "network_or_endpoint_failure", None, secret_name)


def run_probes(env: Mapping[str, str], session: Any = requests) -> list[ProbeResult]:
    """Run bounded probes without putting credentials in URLs or output."""
    results: list[ProbeResult] = []
    openai_key = env.get("OPENAI_API_KEY")
    results.append(_probe(session, "OpenAI", True, bool(openai_key), "OPENAI_API_KEY", "POST",
                          "https://api.openai.com/v1/responses",
                          headers={"Authorization": f"Bearer {openai_key}"},
                          json={"model": "gpt-4.1-nano", "input": "Reply OK", "max_output_tokens": 16}))
    fd_key = env.get("FINANCIAL_DATASETS_API_KEY")
    results.append(_probe(session, "Financial Datasets", True, bool(fd_key),
                          "FINANCIAL_DATASETS_API_KEY", "GET",
                          "https://api.financialdatasets.ai/insider-trades/",
                          headers={"X-API-KEY": fd_key}, params={"ticker": "AAPL", "limit": 1}))
    tavily_key = env.get("TAVILY_API_KEY")
    results.append(_probe(session, "Tavily", False, bool(tavily_key), "TAVILY_API_KEY", "POST",
                          "https://api.tavily.com/search",
                          headers={"Authorization": f"Bearer {tavily_key}"},
                          json={"query": "AAPL", "max_results": 1}))
    exa_key = env.get("EXA_API_KEY")
    results.append(_probe(session, "EXA", False, bool(exa_key), "EXA_API_KEY", "POST",
                          "https://api.exa.ai/search", headers={"x-api-key": exa_key},
                          json={"query": "AAPL", "numResults": 1}))
    discord_token = env.get("DISCORD_TOKEN")
    results.append(_probe(session, "Discord", True, bool(discord_token), "DISCORD_TOKEN", "GET",
                          "https://discord.com/api/v10/users/@me",
                          headers={"Authorization": f"Bot {discord_token}"}))

    reddit_id = env.get("REDDIT_CLIENT_ID")
    reddit_secret = env.get("REDDIT_CLIENT_SECRET")
    reddit_names = "REDDIT_CLIENT_ID,REDDIT_CLIENT_SECRET"
    if reddit_id and reddit_secret:
        try:
            token_response = session.post(
                "https://www.reddit.com/api/v1/access_token", auth=(reddit_id, reddit_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": env.get("REDDIT_USER_AGENT", "stock-analysis-bot/probe")},
                timeout=15,
            )
            token_status = classify_response(token_response.status_code, getattr(token_response, "text", ""))
            try:
                token_payload = token_response.json()
            except (TypeError, ValueError):
                token_payload = None
            if token_status == "working" and validate_payload("Reddit token", token_payload):
                listing = session.get(
                    "https://oauth.reddit.com/r/stocks/new", params={"limit": 1},
                    headers={"Authorization": f"bearer {token_payload['access_token']}",
                             "User-Agent": env.get("REDDIT_USER_AGENT", "stock-analysis-bot/probe")},
                    timeout=15,
                )
                results.append(_from_response("Reddit", False, True, reddit_names, listing,
                                              validator_service="Reddit listing"))
            else:
                results.append(ProbeResult("Reddit", False, True,
                                           token_status if token_status != "working" else "invalid_payload",
                                           token_response.status_code, reddit_names))
        except (requests.RequestException, KeyError, ValueError):
            results.append(ProbeResult("Reddit", False, True,
                                       "network_or_endpoint_failure", None, reddit_names))
    else:
        results.append(ProbeResult("Reddit", False, False, "not_configured", None, reddit_names))

    results.append(_probe(session, "Yahoo Finance", True, True, None, "GET",
                          "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                          params={"range": "1d", "interval": "1d"},
                          headers={"User-Agent": "stock-analysis-bot/probe"}))
    polygon_configured = bool(env.get("POLYGON_API_KEY"))
    results.append(ProbeResult("Polygon (optional/unused)", False, polygon_configured,
                               "unused" if polygon_configured else "not_configured",
                               None, "POLYGON_API_KEY"))
    return results


def main() -> int:
    load_dotenv()
    results = run_probes(os.environ)
    print(json.dumps(safe_report(results), indent=2, sort_keys=True))
    return report_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
