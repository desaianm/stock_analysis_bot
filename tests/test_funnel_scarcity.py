from stockbot.screening.funnel import QuantFunnel
from stockbot.screening.numeric_screen import ScreeningGates, StockSnapshot
from stockbot.screening.universe import Ticker
from stockbot.tools.insider import InsiderSummary


def _snapshot(symbol, **overrides):
    values = {
        "symbol": symbol,
        "exchange": "US",
        "source": "test",
        "company_name": f"{symbol} Corporation",
        "sector": "Industrials",
        "industry": "Any Industry",
        "price": 50.0,
        "market_cap": 10_000_000_000,
        "volume": 1_000_000,
        "trailing_pe": 15.0,
        "price_to_book": 2.0,
        "debt_to_equity": 0.2,
        "current_ratio": 2.0,
        "free_cash_flow": 500_000_000,
        "enterprise_value": 9_000_000_000,
        "revenue_growth": 0.05,
        "earnings_growth": 0.05,
        "gross_margins": 0.30,
        "return_on_equity": 0.20,
        "fifty_two_week_high": 60.0,
        "target_mean_price": 65.0,
    }
    values.update(overrides)
    return StockSnapshot(**values)


def test_funnel_merges_researched_bottleneck_with_value_candidates(monkeypatch):
    classic = _snapshot("VALUE")
    bottleneck = _snapshot(
        "BOTL",
        price=600.0,
        trailing_pe=60.0,
        revenue_growth=0.25,
        earnings_growth=0.40,
        gross_margins=0.45,
        target_mean_price=800.0,
    )
    monkeypatch.setattr(
        "stockbot.screening.funnel.scan_universe",
        lambda *_args, **_kwargs: [classic, bottleneck],
    )
    monkeypatch.setattr(
        "stockbot.screening.funnel.reverse_dcf", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "stockbot.screening.funnel.fetch_insider_summary",
        lambda symbol, **_kwargs: InsiderSummary(
            ticker=symbol,
            window_days=180,
            net_value_usd=0,
            buy_count=0,
            sell_count=0,
            distinct_buyers=0,
            largest_buy_usd=0,
            most_recent_buy_date=None,
        ),
    )
    research = {
        "ticker": "BOTL",
        "company_name": "BOTL Corporation",
        "theme": "Grid component shortage",
        "role": "capacity owner",
        "confidence_score": 8.0,
        "evidence": ["Lead times doubled", "Backlog rose"],
        "source_urls": ["https://one.example", "https://two.example"],
        "already_repriced_risk": "Multiple expansion",
    }
    funnel = QuantFunnel(
        gates=ScreeningGates(max_price=100.0, max_pe=25.0),
        universe=[Ticker("VALUE", "US", "test"), Ticker("BOTL", "US", "test")],
        bottleneck_research={"BOTL": research},
        top_n_final=10,
    )

    result = funnel.run()
    candidates = {candidate.symbol: candidate for candidate in result["candidates"]}

    assert set(candidates) == {"VALUE", "BOTL"}
    assert candidates["VALUE"].discovery_lanes == ["classic_value"]
    assert candidates["BOTL"].discovery_lanes == ["scarcity_capacity"]
    assert candidates["BOTL"].scarcity.research == research
    assert result["stats"]["stage_2_gate_survivors"] == 1
    assert result["stats"]["stage_2_scarcity_candidates_added"] == 1
