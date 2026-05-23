"""Sector-relative quality + valuation ranking.

Takes the survivors of the numeric-screen gates and ranks each stock against
its sector peers, not against absolute thresholds. P/E in the bottom quartile
of Healthcare means something very different from P/E in the bottom quartile
of Utilities — and that's exactly the point.

Composite value score = average percentile across:
    - trailing P/E (lower = better)
    - price/book (lower = better)
    - FCF yield (higher = better)
    - debt/equity (lower = better)
    - ROE (higher = better, quality overlay)

Returns each survivor annotated with its sector medians and composite score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from stockbot.screening.numeric_screen import StockSnapshot


@dataclass
class RankedStock:
    snapshot: StockSnapshot
    sector_size: int
    pe_percentile: Optional[float] = None       # 0=best (cheapest), 1=worst
    pb_percentile: Optional[float] = None
    fcf_yield_percentile: Optional[float] = None  # 0=worst, 1=best
    de_percentile: Optional[float] = None        # 0=best (lowest debt)
    roe_percentile: Optional[float] = None       # 0=worst, 1=best
    composite_value_score: Optional[float] = None  # 0-10, higher = cheaper+better
    metrics_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "snapshot": self.snapshot.to_dict(),
            "sector_size": self.sector_size,
            "pe_percentile": self.pe_percentile,
            "pb_percentile": self.pb_percentile,
            "fcf_yield_percentile": self.fcf_yield_percentile,
            "de_percentile": self.de_percentile,
            "roe_percentile": self.roe_percentile,
            "composite_value_score": self.composite_value_score,
            "metrics_used": self.metrics_used,
        }


def _percentile_rank(value: Optional[float], peer_values: List[float], *, lower_is_better: bool) -> Optional[float]:
    """Return the percentile of `value` within `peer_values`.

    With lower_is_better=True, value below median returns < 0.5; with
    lower_is_better=False, value above median returns > 0.5. The funnel always
    interprets HIGHER percentile as MORE FAVORABLE for the value thesis.
    """
    if value is None or not peer_values:
        return None
    # Convert to a "favorability" percentile: fraction of peers worse than us
    if lower_is_better:
        worse = sum(1 for v in peer_values if v > value)
    else:
        worse = sum(1 for v in peer_values if v < value)
    return worse / len(peer_values)


def _sector_groups(snaps: Iterable[StockSnapshot]) -> Dict[str, List[StockSnapshot]]:
    groups: Dict[str, List[StockSnapshot]] = {}
    for s in snaps:
        sector = s.sector or "Unknown"
        groups.setdefault(sector, []).append(s)
    return groups


def _peer_metric_values(peers: List[StockSnapshot], attr: str) -> List[float]:
    out: List[float] = []
    for p in peers:
        v = getattr(p, attr, None)
        if attr == "fcf_yield":
            v = p.fcf_yield
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        out.append(v)
    return out


def rank_by_sector(snaps: List[StockSnapshot], min_sector_size: int = 5) -> List[RankedStock]:
    """Rank each snapshot against its sector peers.

    Sectors with fewer than `min_sector_size` survivors get only the global
    ranking (still computed but flagged with a small sector_size).
    """
    groups = _sector_groups(snaps)
    out: List[RankedStock] = []

    for sector, peers in groups.items():
        pe_peers = _peer_metric_values(peers, "trailing_pe")
        pb_peers = _peer_metric_values(peers, "price_to_book")
        fcf_peers = _peer_metric_values(peers, "fcf_yield")
        de_peers = _peer_metric_values(peers, "debt_to_equity")
        roe_peers = _peer_metric_values(peers, "return_on_equity")

        for s in peers:
            pe_val = s.trailing_pe or s.forward_pe
            ranked = RankedStock(snapshot=s, sector_size=len(peers))

            ranked.pe_percentile = _percentile_rank(pe_val, pe_peers, lower_is_better=True)
            ranked.pb_percentile = _percentile_rank(s.price_to_book, pb_peers, lower_is_better=True)
            ranked.fcf_yield_percentile = _percentile_rank(s.fcf_yield, fcf_peers, lower_is_better=False)
            ranked.de_percentile = _percentile_rank(s.debt_to_equity, de_peers, lower_is_better=True)
            ranked.roe_percentile = _percentile_rank(s.return_on_equity, roe_peers, lower_is_better=False)

            available = [
                (ranked.pe_percentile, 1.5),       # weight 1.5 — P/E is the headline value metric
                (ranked.fcf_yield_percentile, 1.5),# weight 1.5 — FCF yield is the cash-based value metric
                (ranked.pb_percentile, 1.0),
                (ranked.de_percentile, 1.0),
                (ranked.roe_percentile, 1.0),
            ]
            used = [(p, w) for p, w in available if p is not None]
            ranked.metrics_used = [
                name for name, val in [
                    ("pe", ranked.pe_percentile),
                    ("fcf_yield", ranked.fcf_yield_percentile),
                    ("pb", ranked.pb_percentile),
                    ("de", ranked.de_percentile),
                    ("roe", ranked.roe_percentile),
                ] if val is not None
            ]

            if used:
                weighted_sum = sum(p * w for p, w in used)
                weight_total = sum(w for _, w in used)
                ranked.composite_value_score = round((weighted_sum / weight_total) * 10, 2)

            out.append(ranked)

    # Sort by composite score (cheapest + highest-quality first), missing-score last
    out.sort(
        key=lambda r: (
            -1 if r.composite_value_score is None else -r.composite_value_score,
            r.snapshot.symbol,
        )
    )
    return out


def sector_medians(snaps: Iterable[StockSnapshot]) -> Dict[str, Dict[str, float]]:
    """Compute median P/E, P/B, FCF yield, D/E, ROE per sector (for the report)."""
    groups = _sector_groups(snaps)
    out: Dict[str, Dict[str, float]] = {}
    for sector, peers in groups.items():
        entry: Dict[str, float] = {"count": len(peers)}
        for attr, key in [
            ("trailing_pe", "pe_median"),
            ("price_to_book", "pb_median"),
            ("fcf_yield", "fcf_yield_median"),
            ("debt_to_equity", "de_median"),
            ("return_on_equity", "roe_median"),
        ]:
            vals = _peer_metric_values(peers, attr)
            if vals:
                entry[key] = round(statistics.median(vals), 4)
        out[sector] = entry
    return out


if __name__ == "__main__":
    from stockbot.screening.numeric_screen import scan_universe, apply_gates, ScreeningGates
    from stockbot.screening.universe import load_universe

    print("Loading universe + scanning 60 tickers...")
    universe = load_universe()[:60]
    snaps = scan_universe(universe, workers=10, force_refresh=True)
    survivors = apply_gates(snaps, ScreeningGates())
    print(f"\n{len(survivors)} survivors after gates")

    ranked = rank_by_sector(survivors)
    print(f"\nTop 10 by composite value score:")
    for r in ranked[:10]:
        s = r.snapshot
        print(f"  {s.symbol:8s} {s.sector or '?':22s} score={r.composite_value_score:>5.2f}  "
              f"pe%={r.pe_percentile:>4.2f}  fcf%={r.fcf_yield_percentile:>4.2f}  "
              f"metrics={','.join(r.metrics_used)}")

    print("\nSector medians:")
    for sector, m in sector_medians(survivors).items():
        print(f"  {sector:22s} n={m.get('count', 0):>3d}  pe={m.get('pe_median', float('nan')):>6.2f}  "
              f"fcfy={m.get('fcf_yield_median', float('nan')):>6.2%}")
