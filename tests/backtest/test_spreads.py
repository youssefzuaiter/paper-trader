"""The core's measured half-spreads (O10): the rule, and the pasted table equals a fresh measurement."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from backtest.costs import CORE_HALF_SPREAD_BPS, CORE_LEVELS
from backtest.spreads import half_spread_bps, measure, percentile

SAMPLE = Path(__file__).resolve().parents[2] / ".cache" / "backtest" / "core" / "quotes_sample.parquet"


def test_half_spread_and_bad_quotes() -> None:
    assert half_spread_bps(99.99, 100.01) == pytest.approx(1.0)
    for bid, ask in ((0, 100), (100, 100), (100.01, 100), (-1, 2)):
        assert half_spread_bps(bid, ask) is None


def test_sessions_weigh_equally_and_only_the_open_counts() -> None:
    quotes = [{"symbol": "X", "session": date(2020, 1, 2), "window_name": "open", "bid_price": 99.99,
               "ask_price": 100.01}] * 1000  # 1 bp, many quotes
    quotes += [{"symbol": "X", "session": date(2020, 1, 3), "window_name": "open", "bid_price": 99.9,
                "ask_price": 100.1}]          # 10 bp, one quote
    quotes += [{"symbol": "X", "session": date(2020, 1, 3), "window_name": "midday", "bid_price": 50,
                "ask_price": 150}]           # ignored
    got = measure(quotes)["X"]
    assert got["sessions"] == 2 and got["optimistic"] == Decimal("5.5")
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.75) == pytest.approx(3.25)


@pytest.mark.skipif(not SAMPLE.exists(), reason="needs the core's quote sample")
def test_the_pasted_table_is_the_measurement() -> None:
    from backtest import parquet
    from backtest.registry import sha256_file

    assert sha256_file(SAMPLE).startswith("3b12d115")
    measured = measure(parquet.read_rows(SAMPLE))
    assert {s: (str(r["optimistic"]), str(r["central"]), str(r["pessimistic"])) for s, r in measured.items()} \
        == CORE_HALF_SPREAD_BPS
    assert CORE_LEVELS["central"].half_spread("VTI") == Decimal("0.00024")
