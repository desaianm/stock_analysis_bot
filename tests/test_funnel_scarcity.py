from stockbot.screening.funnel import QuantFunnel
from stockbot.screening.numeric_screen import ScreeningGates, StockSnapshot
from stockbot.screening.universe import Ticker
from stockbot.tools.insider import InsiderSummary
from stockbot.tools.institutional import InstitutionalHolding, InstitutionalPortfolio


def _snapshot(symbol, **overrides):
    values = {
        "symbol": symbol,
        "exchange": "US",
        "source": "test",
        "company_name": f"{symbol} Corporation",
        "sector": "Technology",
        "industry": "Software",
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


def test_funnel_merges_scarcity_lane_with_classic_value_candidates(monkeypatch):
    classic = _snapshot("VALUE")
    scarcity = _snapshot(
        "SNDK",
        company_name="SanDisk Corporation",
        industry="Computer Hardware",
        price=600.0,
        trailing_pe=60.0,
        revenue_growth=0.25,
        earnings_growth=0.40,
        gross_margins=0.45,
        target_mean_price=800.0,
    )
    monkeypatch.setattr(
        "stockbot.screening.funnel.scan_universe",
        lambda *_args, **_kwargs: [classic, scarcity],
    )
    monkeypatch.setattr("stockbot.screening.funnel.reverse_dcf", lambda *_args, **_kwargs: None)
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
    portfolio = InstitutionalPortfolio(
        holdings=[
            InstitutionalHolding("SANDISK CORP", "x", 5_700_000_000, 2_500_000)
        ]
    )
    funnel = QuantFunnel(
        gates=ScreeningGates(max_price=100.0, max_pe=25.0),
        universe=[Ticker("VALUE", "US", "test"), Ticker("SNDK", "US", "test")],
        institutional_portfolio=portfolio,
        top_n_final=10,
    )

    result = funnel.run()
    candidates = {candidate.symbol: candidate for candidate in result["candidates"]}

    assert set(candidates) == {"VALUE", "SNDK"}
    assert candidates["VALUE"].discovery_lanes == ["classic_value"]
    assert candidates["SNDK"].discovery_lanes == ["scarcity_capacity"]
    assert candidates["SNDK"].scarcity is not None
    assert result["stats"]["stage_2_gate_survivors"] == 1
    assert result["stats"]["stage_2_scarcity_candidates_added"] == 1
