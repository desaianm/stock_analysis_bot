import pytest
import asyncio
from types import SimpleNamespace

from stockbot.flows.undervalued import UndervaluedAnalysisFlow
from stockbot.tickers import normalize_ticker


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        ("TSX: SHOP", "SHOP.TO"),
        ("shop.to", "SHOP.TO"),
        ("NYSE: RY", "RY"),
        ("BRK.B", "BRK-B"),
        ("NYSE: BRK.B", "BRK-B"),
        ("BF.B", "BF-B"),
        ("NYSE: BF.B", "BF-B"),
    ],
)
def test_normalize_ticker_accepts_us_tsx_and_suffix_forms(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["../AAPL", "AAPL/../../x", "TSX:", "AAPL$", "AAPL.", "AAPL-", ".AAPL", "-AAPL"],
)
def test_normalize_ticker_rejects_malformed_or_path_like_input(raw):
    with pytest.raises(ValueError):
        normalize_ticker(raw)


def test_funnel_persistence_preserves_market_data_suffix():
    saved = []
    flow = UndervaluedAnalysisFlow.__new__(UndervaluedAnalysisFlow)
    flow.current_run_id = 1
    flow.db = SimpleNamespace(save_stock_find=lambda row: saved.append(row))
    snapshot = SimpleNamespace(exchange="TSX", sector="Tech", industry="Software", price=10,
                               market_cap=100, trailing_pe=12, forward_pe=13, debt_to_equity=1,
                               current_ratio=2, price_to_book=3)
    candidate = SimpleNamespace(snapshot=snapshot)
    output = {"stocks": [{"ticker": "SHOP.TO", "company_name": "Shopify", "verdict": "accept",
                           "confidence_score": 8, "thesis": "thesis"}]}

    asyncio.run(flow._save_funnel_stocks(output, {"SHOP.TO": candidate}))

    assert saved[0]["ticker"] == "SHOP.TO"
