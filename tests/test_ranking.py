from stockbot.screening.numeric_screen import StockSnapshot
from stockbot.screening.ranking import rank_by_sector


def _snapshot(symbol, sector, pe=None):
    return StockSnapshot(symbol=symbol, exchange="US", source="test", sector=sector, trailing_pe=pe)


def test_missing_composite_scores_sort_after_numeric_scores():
    ranked = rank_by_sector([_snapshot("MISSING", "A"), _snapshot("SCORED", "A", 10)])

    assert [row.snapshot.symbol for row in ranked] == ["SCORED", "MISSING"]


def test_small_sectors_use_global_peer_distribution():
    ranked = rank_by_sector(
        [_snapshot("MID", "Tiny", 20), _snapshot("LOW", "Large", 10), _snapshot("HIGH", "Large", 100)],
        min_sector_size=2,
    )

    mid = next(row for row in ranked if row.snapshot.symbol == "MID")
    assert mid.pe_percentile == 1 / 3
