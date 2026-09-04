from stockbot.screening.numeric_screen import ScreeningGates, StockSnapshot
from stockbot.screening.scarcity import score_scarcity_candidate
from stockbot.tools.institutional import InstitutionalHolding, InstitutionalPortfolio


def _snapshot(**overrides):
    values = {
        "symbol": "SNDK",
        "exchange": "US",
        "source": "test",
        "company_name": "SanDisk Corporation",
        "sector": "Technology",
        "industry": "Computer Hardware",
        "price": 600.0,
        "market_cap": 90_000_000_000,
        "volume": 2_000_000,
        "trailing_pe": 60.0,
        "free_cash_flow": 2_000_000_000,
        "revenue_growth": 0.22,
        "earnings_growth": 0.35,
        "gross_margins": 0.42,
        "fifty_two_week_high": 700.0,
        "target_mean_price": 750.0,
    }
    values.update(overrides)
    return StockSnapshot(**values)


def test_scarcity_lane_can_find_expensive_growth_stock_outside_value_gates():
    gates = ScreeningGates(max_price=100.0, max_pe=25.0)
    portfolio = InstitutionalPortfolio(
        holdings=[
            InstitutionalHolding("SANDISK CORP", "80004C200", 5_700_000_000, 2_500_000)
        ]
    )

    signal = score_scarcity_candidate(_snapshot(), gates, portfolio)

    assert signal is not None
    assert signal.score >= 4
    assert signal.institutional_direct_long_value == 5_700_000_000
    assert any("Situational Awareness" in reason for reason in signal.reasons)


def test_option_position_does_not_create_direct_long_boost():
    portfolio = InstitutionalPortfolio(
        holdings=[
            InstitutionalHolding("NVIDIA CORP", "x", 1_000_000_000, 5_000_000, "put")
        ]
    )

    signal = score_scarcity_candidate(
        _snapshot(symbol="NVDA", company_name="NVIDIA Corporation"),
        ScreeningGates(),
        portfolio,
    )

    assert signal is not None
    assert signal.institutional_direct_long_value == 0
    assert signal.institutional_positions[0]["option_type"] == "put"


def test_generic_company_is_not_admitted_by_growth_alone():
    assert score_scarcity_candidate(
        _snapshot(company_name="Retail Co", industry="Apparel Retail"),
        ScreeningGates(),
    ) is None


def test_scarcity_lane_keeps_basic_liquidity_guardrails():
    assert score_scarcity_candidate(
        _snapshot(volume=100), ScreeningGates(min_volume=200_000)
    ) is None


def test_scarcity_lane_requires_cash_flow_or_strong_revenue_growth():
    assert score_scarcity_candidate(
        _snapshot(free_cash_flow=-1, revenue_growth=0.05), ScreeningGates()
    ) is None
