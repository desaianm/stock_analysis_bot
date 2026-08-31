import json

import pytest

from stockbot.flows.undervalued import UndervaluedAnalysisFlow


def _stock(ticker, verdict="accept"):
    row = {
        "ticker": ticker, "company_name": ticker, "sector": "Tech", "verdict": verdict,
        "confidence_score": 8, "thesis": "A specific bull case with supporting evidence.",
        "key_risks": ["competition"], "primary_catalyst": "earnings",
        "entry_strategy": "below $10", "stop_loss_pct": 0.2, "position_size_pct": 0.03,
    }
    if verdict == "reject":
        row["rejection_reason"] = "weak quality"
    return row


def test_fenced_deep_dive_json_is_validated():
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 1,
                                    "candidates_accepted": 1}, "stocks": [_stock("SHOP.TO")]}
    parsed = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
        f"```json\n{json.dumps(payload)}\n```", ["SHOP.TO"]
    )
    assert parsed["stocks"][0]["ticker"] == "SHOP.TO"


def test_contradictory_accepted_count_fails_explicitly():
    stocks = [_stock(f"S{i}") for i in range(7)]
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 7,
                                    "candidates_accepted": 6}, "stocks": stocks}

    with pytest.raises(ValueError, match="candidates_accepted"):
        UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(json.dumps(payload), [f"S{i}" for i in range(7)])


def test_rejected_stock_requires_zero_position_size():
    rejected = _stock("NOPE", verdict="reject")
    rejected["position_size_pct"] = 0.0
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 1,
                                    "candidates_accepted": 0}, "stocks": [rejected]}

    parsed = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(json.dumps(payload), ["NOPE"])

    assert parsed["stocks"][0]["position_size_pct"] == 0.0


@pytest.mark.parametrize(
    ("verdict", "position_size"),
    [("accept", 0.0), ("reject", 0.03)],
)
def test_position_size_must_match_verdict(verdict, position_size):
    stock = _stock("RULE", verdict=verdict)
    stock["position_size_pct"] = position_size
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 1,
                                    "candidates_accepted": int(verdict == "accept")}, "stocks": [stock]}

    with pytest.raises(ValueError, match="position_size_pct"):
        UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(json.dumps(payload), ["RULE"])


@pytest.mark.parametrize(
    ("expected", "actual", "message"),
    [
        (["BRK-B"], ["BRK.B", "BRK-B"], "duplicate"),
        (["AAPL", "MSFT"], ["AAPL"], "omitted"),
        (["AAPL"], ["AAPL", "MSFT"], "unexpected"),
    ],
)
def test_deep_dive_requires_one_verdict_per_expected_candidate(expected, actual, message):
    stocks = [_stock(ticker) for ticker in actual]
    payload = {
        "shortlist_review": {
            "reviewed_at": "2026-08-31T12:00:00Z",
            "candidates_reviewed": len(stocks),
            "candidates_accepted": len(stocks),
        },
        "stocks": stocks,
    }

    with pytest.raises(ValueError, match=message):
        UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
            json.dumps(payload), expected
        )
