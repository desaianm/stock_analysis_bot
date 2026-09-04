import json

import pytest

from stockbot.flows.undervalued import UndervaluedAnalysisFlow
from stockbot.screening.universe import Ticker


def _payload():
    return {
        "researched_at": "2026-09-04T15:00:00-04:00",
        "market_regime_summary": "Physical infrastructure is tightening.",
        "themes": [
            {
                "name": "Transformer capacity",
                "constrained_input": "large power transformers",
                "demand_driver": "grid and data-center construction",
                "change_signal": "Lead times worsened during the latest quarter.",
                "supply_response_lag": "New plants take several years.",
                "time_horizon_months": 18,
                "evidence": ["Lead times doubled", "Prices rose 75%"],
                "source_urls": [
                    "https://www.iea.org/example",
                    "https://www.energy.gov/example",
                ],
            }
        ],
        "candidates": [
            {
                "ticker": "NYSE:GRID",
                "company_name": "Grid Equipment Corporation",
                "theme": "Transformer capacity",
                "role": "equipment supplier",
                "confidence_score": 8.2,
                "earnings_transmission": "Backlog supports price and utilization.",
                "evidence": ["Backlog expanded", "New capacity takes years"],
                "source_urls": [
                    "https://grid.example/filing",
                    "https://regulator.example/data",
                ],
                "already_repriced_risk": "The order boom may be priced in.",
            }
        ],
    }


def test_bottleneck_research_is_validated_and_tickers_are_normalized():
    parsed = UndervaluedAnalysisFlow._parse_bottleneck_research_json(
        f"```json\n{json.dumps(_payload())}\n```"
    )

    assert parsed["candidates"][0]["ticker"] == "GRID"


def test_research_rejects_candidate_with_unknown_theme():
    payload = _payload()
    payload["candidates"][0]["theme"] = "Invented theme"

    with pytest.raises(ValueError, match="unknown themes"):
        UndervaluedAnalysisFlow._parse_bottleneck_research_json(json.dumps(payload))


def test_research_requires_multiple_web_sources():
    payload = _payload()
    payload["themes"][0]["source_urls"] = ["https://one.example"]

    with pytest.raises(ValueError, match="at least 2"):
        UndervaluedAnalysisFlow._parse_bottleneck_research_json(json.dumps(payload))


def test_researched_tickers_are_added_outside_base_universe():
    research = _payload()
    research["candidates"].append(
        {**research["candidates"][0], "ticker": "SHOP.TO"}
    )

    augmented = UndervaluedAnalysisFlow._augment_universe_with_research(
        [Ticker("GRID", "US", "sp500")], research
    )

    assert [(ticker.symbol, ticker.exchange, ticker.source) for ticker in augmented] == [
        ("GRID", "US", "sp500"),
        ("SHOP.TO", "TSX", "bottleneck_research"),
    ]
