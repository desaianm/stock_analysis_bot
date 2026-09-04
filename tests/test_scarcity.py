from stockbot.screening.numeric_screen import ScreeningGates, StockSnapshot
from stockbot.screening.scarcity import score_scarcity_candidate


def _snapshot(**overrides):
    values = {
        "symbol": "BOTL",
        "exchange": "US",
        "source": "bottleneck_research",
        "company_name": "Bottleneck Corporation",
        "sector": "Industrials",
        "industry": "Any Industry",
        "price": 600.0,
        "market_cap": 20_000_000_000,
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


def _research(**overrides):
    values = {
        "ticker": "BOTL",
        "company_name": "Bottleneck Corporation",
        "theme": "Emerging constraint",
        "role": "capacity owner",
        "confidence_score": 8.0,
        "evidence": ["Lead times doubled", "Capacity is sold out"],
        "source_urls": ["https://one.example", "https://two.example"],
        "already_repriced_risk": "The stock has rerated.",
    }
    values.update(overrides)
    return values


def test_researched_company_can_qualify_outside_classic_industries_and_gates():
    signal = score_scarcity_candidate(
        _snapshot(industry="Agricultural Inputs"),
        ScreeningGates(max_price=100.0, max_pe=25.0),
        _research(theme="Fertilizer feedstock shortage"),
    )

    assert signal is not None
    assert signal.score >= 4
    assert signal.research["theme"] == "Fertilizer feedstock shortage"


def test_company_is_not_admitted_without_web_research():
    assert score_scarcity_candidate(_snapshot(), ScreeningGates(), None) is None


def test_low_confidence_research_is_rejected():
    assert score_scarcity_candidate(
        _snapshot(), ScreeningGates(), _research(confidence_score=5.9)
    ) is None


def test_research_lane_keeps_basic_liquidity_guardrails():
    assert score_scarcity_candidate(
        _snapshot(volume=100),
        ScreeningGates(min_volume=200_000),
        _research(),
    ) is None


def test_research_lane_requires_cash_flow_or_strong_revenue_growth():
    assert score_scarcity_candidate(
        _snapshot(free_cash_flow=-1, revenue_growth=0.05),
        ScreeningGates(),
        _research(),
    ) is None
