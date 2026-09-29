"""Synthetic markets for the backtester's unit tests.

A few weeks in 2025 around Independence Day: Thursday 07-03 is a 13:00
half-day and Friday 07-04 a holiday. Two symbols with deterministic
1-minute random walks, daily bars back to March (enough history for
features), and a 4-for-1 split on AAPL effective 2025-06-16 so the
raw/adjusted factor is not 1.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pytest

from backtest.calendar import Calendar, Session
from backtest.data import MarketData, regular_series
from backtest.dataset import Samples
from swarm.alpaca_data import Bar
from swarm.common import NEW_YORK
from swarm.features import FEATURE_NAMES

SYMBOLS = ("AAPL", "TSLA")
HOLIDAY = date(2025, 7, 4)
HALF_DAY = date(2025, 7, 3)
SPLIT_DAY = date(2025, 6, 16)
SPLIT = 4.0


def ny(d: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm, ss), NEW_YORK).astimezone(UTC)


def trading_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5 and d != HOLIDAY:
            days.append(d)
        d += timedelta(days=1)
    return days


def make_calendar(start: date = date(2025, 3, 3), end: date = date(2025, 7, 31)) -> Calendar:
    return Calendar([Session(d, ny(d, 9, 30), ny(d, 13 if d == HALF_DAY else 16), d == HALF_DAY)
                     for d in trading_days(start, end)])


def minute_arrays(symbol: str, calendar: Calendar, first: date, *, seed: int) -> dict[str, np.ndarray]:
    """Regular-hours 1-minute bars plus a few pre-market ones (which must be ignored)."""
    rng = random.Random(seed)
    price = 200.0 if symbol == "AAPL" else 300.0
    t, o, h, lo, c, v = [], [], [], [], [], []
    for s in calendar.sessions:
        if s.date < first:
            continue
        pre = s.open_at - timedelta(minutes=3)
        for k in range(3):  # pre-market noise: must never be a quote or a fill
            t.append((pre + timedelta(minutes=k)).timestamp())
            o.append(price * 5), h.append(price * 5), lo.append(price * 5), c.append(price * 5), v.append(1.0)
        minute = s.open_at
        while minute < s.close_at:
            op = price
            cl = round(op * (1 + rng.gauss(0, 0.0008)), 2)
            t.append(minute.timestamp())
            o.append(op), c.append(cl)
            h.append(round(max(op, cl) + 0.05, 2)), lo.append(round(min(op, cl) - 0.05, 2)), v.append(1000.0)
            price, minute = cl, minute + timedelta(minutes=1)
    return {k: np.asarray(x, dtype=float) for k, x in zip("tohlcv", (t, o, h, lo, c, v), strict=True)}


def daily_bars(symbol: str, calendar: Calendar, *, seed: int) -> tuple[list[Bar], dict[date, float]]:
    """Adjusted daily bars (``Bar``, midnight New York) and raw closes by session."""
    rng = random.Random(seed)
    adjusted, raw = [], {}
    price = 50.0 if symbol == "AAPL" else 300.0  # adjusted scale
    for s in calendar.sessions:
        op = price
        cl = op * (1 + rng.gauss(0, 0.01))
        adjusted.append(Bar(t=datetime.combine(s.date, time(0), NEW_YORK), o=op, h=max(op, cl) * 1.005,
                            l=min(op, cl) * 0.995, c=cl, v=1e6 * (1 + rng.random())))
        factor = 1 / SPLIT if symbol == "AAPL" and s.date < SPLIT_DAY else 1.0
        raw[s.date] = cl / factor
        price = cl
    return adjusted, raw


def make_market(calendar: Calendar | None = None, *, minute_from: date = date(2025, 6, 2)) -> MarketData:
    calendar = calendar or make_calendar()
    minute, daily, raw = {}, {}, {}
    for k, symbol in enumerate(SYMBOLS):
        a = minute_arrays(symbol, calendar, minute_from, seed=k + 1)
        minute[symbol] = regular_series(symbol, 60, a["t"].astype(np.int64), a["o"], a["h"], a["l"], a["c"],
                                        a["v"], calendar)
        daily[symbol], raw[symbol] = daily_bars(symbol, calendar, seed=k + 10)
    return MarketData(calendar, minute, daily, raw)


@pytest.fixture(scope="session")
def calendar() -> Calendar:
    return make_calendar()


@pytest.fixture(scope="session")
def market(calendar: Calendar) -> MarketData:
    return make_market(calendar)


@pytest.fixture(scope="session")
def long_calendar() -> Calendar:
    return Calendar.weekday_rule(date(2023, 6, 1), date(2026, 12, 31))


def synthetic_samples(calendar: Calendar, n: int = 12_000, seed: int = 0, *, up: float = 0.006,
                      down: float = -0.002, symbols: tuple[str, ...] = ("AAPL", "TSLA", "MSFT")) -> Samples:
    """Events spread over 2024-01 → 2026-09 with one informative feature."""
    rng = np.random.default_rng(seed)
    sessions = calendar.sessions_between(date(2024, 1, 2), date(2026, 9, 17))
    pick = np.sort(rng.integers(0, len(sessions), n))
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (X[:, 7] + rng.normal(scale=2.0, size=n) > 0.4).astype(int)
    night = rng.random(n) < 0.4
    made, resolved, entry, session_idx = (np.empty(n) for _ in range(4))
    for k, j in enumerate(pick):
        s = sessions[j]
        made[k] = (s.open_at - timedelta(hours=3) if night[k] else s.open_at + timedelta(hours=2)).timestamp()
        entry[k] = (s.open_at + timedelta(minutes=2 if night[k] else 121)).timestamp()
        resolved[k] = s.close_at.timestamp()
        session_idx[k] = calendar.index[s.date]
    fwd = np.where(y == 1, up, down) + rng.normal(scale=0.001, size=n)
    symbols = rng.choice(list(symbols), n)
    return Samples(X=X, y=y, fwd=fwd, published_at=made - 60, known_at=made - 60, made_at=made, entry_at=entry,
                   resolved_at=resolved, entry_price=np.full(n, 100.0), exit_price=100.0 * (1 + fwd),
                   idealised_fwd=fwd, symbol=symbols, event_id=np.asarray([f"alpaca:{k}" for k in range(n)]),
                   story_id=np.asarray([""] * n, dtype="<U32"), session=session_idx.astype(int), night=night,
                   rolled=np.zeros(n, dtype=bool), tradeable=~night, day=np.asarray([""] * n))
