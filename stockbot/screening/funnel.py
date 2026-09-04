"""Dual-lane funnel: value + researched bottlenecks → ranking → DCF → insider.

Run ``QuantFunnel(...).run()`` to get a list of FunnelCandidate objects, each
annotated with sector-relative ranking, reverse-DCF margin of safety, recent
insider activity, and any validated scarcity signal. Hand this list to the
agent for narrative deep-dive.

Web research is performed by the owning flow and passed in as structured
candidate evidence. Reddit is intentionally NOT consulted here — it's added in
``stockbot/flows/undervalued.py`` as a Stage-5 catalyst overlay alongside
analyst targets and recent news.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from stockbot.screening.numeric_screen import (
    ScreeningGates,
    StockSnapshot,
    apply_gates,
    scan_universe,
)
from stockbot.screening.ranking import RankedStock, rank_by_sector, sector_medians
from stockbot.screening.scarcity import ScarcitySignal, score_scarcity_candidate
from stockbot.screening.universe import Ticker, load_universe
from stockbot.screening.valuation import (
    ReverseDCFResult,
    margin_of_safety_score,
    reverse_dcf,
)
from stockbot.tools.insider import (
    InsiderSummary,
    fetch_insider_summary,
    insider_signal_score,
)


@dataclass
class FunnelCandidate:
    """Aggregated, ranked stock ready for the agent's narrative pass."""
    symbol: str
    sector: Optional[str]
    industry: Optional[str]
    snapshot: StockSnapshot
    ranking: RankedStock
    dcf: Optional[ReverseDCFResult] = None
    insider: Optional[InsiderSummary] = None
    scarcity: Optional[ScarcitySignal] = None
    discovery_lanes: List[str] = field(default_factory=list)
    composite_funnel_score: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "industry": self.industry,
            "snapshot": self.snapshot.to_dict(),
            "ranking": self.ranking.to_dict(),
            "dcf": self.dcf.to_dict() if self.dcf else None,
            "insider": self.insider.to_dict() if self.insider else None,
            "scarcity": asdict(self.scarcity) if self.scarcity else None,
            "discovery_lanes": self.discovery_lanes,
            "composite_funnel_score": self.composite_funnel_score,
        }

    def to_prompt_summary(self) -> Dict[str, Any]:
        """Condensed form suitable for an LLM prompt (smaller token footprint)."""
        s = self.snapshot
        pe = s.trailing_pe if s.trailing_pe is not None else s.forward_pe
        return {
            "ticker": self.symbol,
            "company_name": s.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "price": s.price,
            "market_cap": s.market_cap,
            "pe": pe,
            "price_to_book": s.price_to_book,
            "fcf_yield": s.fcf_yield,
            "debt_to_equity": s.debt_to_equity,
            "current_ratio": s.current_ratio,
            "revenue_growth": s.revenue_growth,
            "earnings_growth": s.earnings_growth,
            "return_on_equity": s.return_on_equity,
            "analyst_upside": s.analyst_upside,
            "historical_fcf_growth": s.historical_fcf_growth,
            "sector_value_score": self.ranking.composite_value_score,
            "sector_size": self.ranking.sector_size,
            "implied_growth_by_discount": (
                self.dcf.by_discount_rate if self.dcf and not self.dcf.error else None
            ),
            "insider_net_buying_90d": (
                self.insider.net_value_usd if self.insider and not self.insider.error else None
            ),
            "insider_distinct_buyers": (
                self.insider.distinct_buyers if self.insider and not self.insider.error else None
            ),
            "discovery_lanes": self.discovery_lanes,
            "scarcity_score": self.scarcity.score if self.scarcity else None,
            "scarcity_reasons": self.scarcity.reasons if self.scarcity else [],
            "bottleneck_research": self.scarcity.research if self.scarcity else None,
            "composite_funnel_score": self.composite_funnel_score,
        }


@dataclass
class QuantFunnel:
    """Orchestrator. Configure once, then call ``run()``."""
    gates: ScreeningGates = field(default_factory=ScreeningGates)
    top_n_for_dcf: int = 30
    top_n_for_insider: int = 20
    top_n_final: int = 10
    top_n_scarcity: int = 5
    universe: Optional[List[Ticker]] = None
    bottleneck_research: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    force_refresh: bool = False
    workers: int = 5

    def run(self) -> Dict[str, Any]:
        """Execute stages 1-5. Returns a dict with the shortlist + stats."""
        # Stage 1: Universe
        universe = self.universe if self.universe is not None else load_universe()
        stage_stats: Dict[str, Any] = {"stage_1_universe_size": len(universe)}

        # Stage 2: Numeric screen
        snaps = scan_universe(
            universe, workers=self.workers, force_refresh=self.force_refresh
        )
        ok_snaps = [s for s in snaps if not s.error]
        stage_stats["stage_2_fetched_ok"] = len(ok_snaps)
        stage_stats["stage_2_fetched_failed"] = len(snaps) - len(ok_snaps)

        classic_survivors = apply_gates(ok_snaps, self.gates)
        stage_stats["stage_2_gate_survivors"] = len(classic_survivors)

        scarcity_signals = [
            signal
            for snapshot in ok_snaps
            if (
                signal := score_scarcity_candidate(
                    snapshot,
                    self.gates,
                    self.bottleneck_research.get(snapshot.symbol),
                )
            )
        ]
        scarcity_signals.sort(key=lambda signal: (-signal.score, signal.symbol))
        selected_scarcity = scarcity_signals[: self.top_n_scarcity]
        scarcity_by_symbol = {signal.symbol: signal for signal in selected_scarcity}
        classic_symbols = {snapshot.symbol for snapshot in classic_survivors}
        scarcity_additions = [
            snapshot
            for snapshot in ok_snaps
            if snapshot.symbol in scarcity_by_symbol
            and snapshot.symbol not in classic_symbols
        ]
        survivors = classic_survivors + scarcity_additions
        stage_stats["stage_2_scarcity_lane_selected"] = len(selected_scarcity)
        stage_stats["stage_2_scarcity_candidates_added"] = len(scarcity_additions)

        if not survivors:
            return {
                "stats": stage_stats,
                "candidates": [],
                "sector_medians": {},
            }

        # Stage 3: Sector-relative ranking
        ranked = rank_by_sector(survivors)
        sectors = sector_medians(survivors)
        stage_stats["stage_3_ranked"] = len(ranked)

        # Take top N for DCF (Stage 4) and insider (Stage 5)
        ranked_by_symbol = {row.snapshot.symbol: row for row in ranked}
        scarcity_ranked = [
            ranked_by_symbol[signal.symbol]
            for signal in selected_scarcity
            if signal.symbol in ranked_by_symbol
        ]
        scarcity_symbols = {row.snapshot.symbol for row in scarcity_ranked}
        ranked_for_dcf = (
            scarcity_ranked
            + [row for row in ranked if row.snapshot.symbol not in scarcity_symbols]
        )[: self.top_n_for_dcf]

        # Stage 4: Reverse-DCF margin of safety
        candidates: List[FunnelCandidate] = []
        for r in ranked_for_dcf:
            s = r.snapshot
            dcf = reverse_dcf(
                s.symbol,
                current_fcf=s.free_cash_flow or 0.0,
                enterprise_value=s.enterprise_value or s.market_cap or 0.0,
            )
            candidates.append(
                FunnelCandidate(
                    symbol=s.symbol,
                    sector=s.sector,
                    industry=s.industry,
                    snapshot=s,
                    ranking=r,
                    dcf=dcf,
                    scarcity=scarcity_by_symbol.get(s.symbol),
                    discovery_lanes=(
                        (["classic_value"] if s.symbol in classic_symbols else [])
                        + (["scarcity_capacity"] if s.symbol in scarcity_by_symbol else [])
                    ),
                )
            )

        # Re-rank by combined value-score + DCF margin-of-safety to pick top-N for insider
        def _value_plus_mos(c: FunnelCandidate) -> float:
            v = c.ranking.composite_value_score or 0.0
            if not c.dcf or c.dcf.error or not c.snapshot.historical_fcf_growth:
                return v + (c.scarcity.score if c.scarcity else 0.0)
            implied = c.dcf.implied_growth_at(0.10)
            if implied is None:
                return v + (c.scarcity.score if c.scarcity else 0.0)
            mos = margin_of_safety_score(implied, c.snapshot.historical_fcf_growth)
            return v + mos + (c.scarcity.score if c.scarcity else 0.0)

        candidates.sort(key=_value_plus_mos, reverse=True)
        for_insider = candidates[: self.top_n_for_insider]
        stage_stats["stage_4_dcf_computed"] = sum(
            1 for c in for_insider if c.dcf and not c.dcf.error
        )

        # Stage 5: Insider trades (sequential — Financial Datasets rate-limit friendly)
        for c in for_insider:
            c.insider = fetch_insider_summary(c.symbol, days=180)
        stage_stats["stage_5_insider_fetched"] = sum(
            1 for c in for_insider if c.insider and not c.insider.error
        )

        # Composite funnel score: sector value + MOS + insider + scarcity/capacity
        for c in for_insider:
            value = c.ranking.composite_value_score or 0.0
            mos = 0.0
            if c.dcf and not c.dcf.error and c.snapshot.historical_fcf_growth:
                implied = c.dcf.implied_growth_at(0.10)
                if implied is not None:
                    mos = margin_of_safety_score(implied, c.snapshot.historical_fcf_growth)
            insider_score = insider_signal_score(c.insider) if c.insider else 0.0
            scarcity_score = c.scarcity.score if c.scarcity else 0.0
            c.composite_funnel_score = round(
                value + mos + insider_score + scarcity_score, 2
            )

        # Final shortlist
        for_insider.sort(key=lambda c: c.composite_funnel_score, reverse=True)
        shortlist = for_insider[: self.top_n_final]
        stage_stats["stage_5_shortlist_size"] = len(shortlist)

        return {
            "stats": stage_stats,
            "candidates": shortlist,
            "sector_medians": sectors,
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Smoke: tiny universe
    universe = load_universe()[:80]
    funnel = QuantFunnel(
        universe=universe,
        top_n_for_dcf=10,
        top_n_for_insider=5,
        top_n_final=3,
        force_refresh=True,
    )
    result = funnel.run()
    print("\nStats:")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
    print("\nShortlist:")
    for c in result["candidates"]:
        s = c.snapshot
        d = c.dcf
        i = c.insider
        print(f"\n  {c.symbol} ({c.sector})  composite={c.composite_funnel_score:.2f}")
        print(f"    price=${s.price:.2f}  mcap=${(s.market_cap or 0)/1e9:.1f}B")
        print(f"    sector value score={c.ranking.composite_value_score}")
        if d and not d.error:
            print(f"    implied growth: {d.by_discount_rate}  hist fcf growth={s.historical_fcf_growth}")
        if i and not i.error:
            print(f"    insider net 180d: ${i.net_value_usd:,.0f}  buyers={i.distinct_buyers}")
