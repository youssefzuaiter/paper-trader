"""Event memory: every prediction resolved at every horizon."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import ny

from backtest.data import MarketData
from backtest.outcomes import HORIZONS, resolve

DAY = date(2025, 6, 30)


def prediction(at, pid: str = "p1") -> dict:
    return {"prediction_id": pid, "symbol": "AAPL", "reference_at": at}


def by_horizon(rows: list[dict]) -> dict[str, dict]:
    return {r["horizon"]: r for r in rows}


def test_intraday_horizons_from_the_entry_bar(market: MarketData) -> None:
    rows = by_horizon(resolve([prediction(ny(DAY, 10, 2))], market))
    assert list(rows) == list(HORIZONS)
    s = market.minute["AAPL"]
    i = s.first_at_or_after(ny(DAY, 10, 2).timestamp())
    assert rows["5m"]["ret"] == pytest.approx(s.c[i + 4] / s.o[i] - 1)          # the bar ending 10:07
    assert rows["5m"]["resolved_at"] == ny(DAY, 10, 7)
    assert rows["5m"]["max_favourable"] == pytest.approx(s.h[i:i + 5].max() / s.o[i] - 1)
    assert rows["5m"]["max_adverse"] == pytest.approx(s.l[i:i + 5].min() / s.o[i] - 1)
    last = s.session_last(i)
    assert rows["close"]["ret"] == pytest.approx(s.c[last] / s.o[i] - 1) and rows["close"]["status"] == "ok"
    assert all(rows[h]["price_space"] == "raw" for h in ("5m", "30m", "2h", "close"))


def test_a_horizon_past_the_close_is_truncated(market: MarketData) -> None:
    rows = by_horizon(resolve([prediction(ny(DAY, 15))], market))
    assert rows["2h"]["status"] == "truncated" and rows["2h"]["resolved_at"] == ny(DAY, 16)
    assert rows["30m"]["status"] == "ok"


def test_daily_horizons_are_adjusted_and_missing_data_is_flagged(market: MarketData) -> None:
    rows = by_horizon(resolve([prediction(ny(DAY, 10, 2))], market))
    s = market.minute["AAPL"]
    i = s.first_at_or_after(ny(DAY, 10, 2).timestamp())
    next_close = next(b.c for b in market.daily["AAPL"] if b.t.date() == date(2025, 7, 1))
    entry_adj = s.o[i] * market.factor("AAPL", DAY)
    assert rows["1d"]["ret"] == pytest.approx(next_close / entry_adj - 1)
    assert rows["1d"]["resolved_at"] == ny(date(2025, 7, 1), 16) and rows["1d"]["price_space"] == "adjusted"
    assert rows["1mo"]["status"] == "ok"                                        # 21 sessions later: 07-30
    late = by_horizon(resolve([prediction(ny(date(2025, 7, 15), 10, 2), "p2")], market))
    assert late["1mo"]["status"] == "no_data" and late["1mo"]["ret"] is None    # the fixture ends 2025-07-31
