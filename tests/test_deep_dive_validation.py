import json
import asyncio

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


def test_contradictory_counts_are_derived_from_validated_stocks():
    stocks = [_stock("KEEP"), _stock("DROP", verdict="reject")]
    stocks[1]["position_size_pct"] = 0.0
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 99,
                                    "candidates_accepted": 98}, "stocks": stocks}

    parsed = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
        json.dumps(payload), ["KEEP", "DROP"]
    )

    assert parsed["shortlist_review"]["candidates_reviewed"] == 2
    assert parsed["shortlist_review"]["candidates_accepted"] == 1


def test_incorrect_zero_counts_are_derived_from_validated_stocks():
    stocks = [_stock("ONE"), _stock("TWO")]
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 0,
                                    "candidates_accepted": 0}, "stocks": stocks}

    parsed = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
        json.dumps(payload), ["ONE", "TWO"]
    )

    assert parsed["shortlist_review"]["candidates_reviewed"] == 2
    assert parsed["shortlist_review"]["candidates_accepted"] == 2


def test_count_normalization_does_not_allow_malformed_stock_objects():
    stock = _stock("BROKEN")
    del stock["thesis"]
    payload = {"shortlist_review": {"reviewed_at": "2026-08-31T12:00:00Z", "candidates_reviewed": 0,
                                    "candidates_accepted": 0}, "stocks": [stock]}

    with pytest.raises(ValueError, match="thesis"):
        UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
            json.dumps(payload), ["BROKEN"]
        )


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


def test_identical_duplicate_deep_dive_stock_is_collapsed():
    stock = _stock("VLTO")
    payload = {
        "shortlist_review": {
            "reviewed_at": "2026-09-03T10:01:58-04:00",
            "candidates_reviewed": 2,
            "candidates_accepted": 2,
        },
        "stocks": [stock, dict(stock)],
    }

    parsed = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
        json.dumps(payload), ["VLTO"]
    )

    assert [row["ticker"] for row in parsed["stocks"]] == ["VLTO"]
    assert parsed["shortlist_review"]["candidates_reviewed"] == 1
    assert parsed["shortlist_review"]["candidates_accepted"] == 1


def test_invalid_duplicate_placeholder_is_discarded_before_validation():
    valid = _stock("CTSH")
    placeholder = {
        **_stock("CTSH", verdict="reject"),
        "confidence_score": 0,
        "thesis": "",
        "key_risks": [],
        "primary_catalyst": "",
        "entry_strategy": "",
        "stop_loss_pct": 0,
        "position_size_pct": 0,
        "rejection_reason": (
            "Accepted in the narrative review; duplicate placeholder removed "
            "from final shortlist."
        ),
    }
    payload = {
        "shortlist_review": {
            "reviewed_at": "2026-09-04T10:02:26-04:00",
            "candidates_reviewed": 2,
            "candidates_accepted": 1,
        },
        "stocks": [valid, placeholder],
    }

    parsed = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
        json.dumps(payload), ["CTSH"]
    )

    assert parsed["stocks"] == [{**valid, "rejection_reason": None}]
    assert parsed["shortlist_review"]["candidates_reviewed"] == 1
    assert parsed["shortlist_review"]["candidates_accepted"] == 1


def test_invalid_sole_stock_still_fails_validation():
    stock = _stock("BROKEN")
    stock["thesis"] = ""
    payload = {
        "shortlist_review": {
            "reviewed_at": "2026-09-04T10:02:26-04:00",
            "candidates_reviewed": 1,
            "candidates_accepted": 1,
        },
        "stocks": [stock],
    }

    with pytest.raises(ValueError, match="thesis"):
        UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)._parse_deep_dive_json(
            json.dumps(payload), ["BROKEN"]
        )


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


@pytest.mark.parametrize("deep_dive", [None, {}])
def test_nonempty_shortlist_requires_parseable_deep_dive(deep_dive):
    with pytest.raises(RuntimeError, match="empty or unparseable"):
        UndervaluedAnalysisFlow._require_deep_dive(deep_dive, [object()])


def test_cancelled_run_is_persisted_failed_before_reraise(monkeypatch):
    completed = []
    audit = []
    flow = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)
    flow.current_run_id = 42
    flow.db = type("DB", (), {
        "complete_analysis_run": lambda _self, **kwargs: completed.append(kwargs)
    })()
    monkeypatch.setattr(
        "stockbot.flows.undervalued.write_state",
        lambda name, **kwargs: audit.append((name, kwargs)),
    )

    cancellation = asyncio.CancelledError("cancelled")
    flow._persist_failed_run(cancellation, phase="cancelled")

    assert completed == [{"run_id": 42, "status": "failed"}]
    assert audit == [("undervalued", {
        "run_id": 42, "phase": "cancelled", "error": "cancelled"
    })]


def test_execute_explicitly_persists_cancelled_run_before_reraising(monkeypatch):
    completed = []
    audit = []
    flow = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)
    flow.current_run_id = None
    flow.preferences = type("Preferences", (), {
        "__dict__": {}, "model_dump": lambda _self: {}
    })()
    flow.db = type("DB", (), {
        "create_analysis_run": lambda _self, **_kwargs: 42,
        "complete_analysis_run": lambda _self, **kwargs: completed.append(kwargs),
    })()
    flow._configure_agno_debug_logging = lambda: "test.log"
    flow._run_startup_ritual = lambda: None
    flow._preferences_to_gates = lambda: object()

    async def no_research():
        return None

    flow._research_bottlenecks = no_research

    class CancellingFunnel:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            raise asyncio.CancelledError("worker terminated")

    monkeypatch.setattr("stockbot.flows.undervalued.QuantFunnel", CancellingFunnel)
    monkeypatch.setattr(
        "stockbot.flows.undervalued.write_state",
        lambda name, **kwargs: audit.append((name, kwargs)),
    )

    with pytest.raises(asyncio.CancelledError, match="worker terminated"):
        asyncio.run(flow.execute_undervalued_analysis(universe=[]))

    assert completed == [{"run_id": 42, "status": "failed"}]
    assert audit[-1] == ("undervalued", {
        "run_id": 42, "phase": "cancelled", "error": "worker terminated"
    })
