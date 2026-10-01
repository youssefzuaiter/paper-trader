"""Daily market data for the long-term core (core design §2, §3.1).

The engine works on session *indices* of one exchange calendar. Two sides:

* **Decision side**: ``DailyView``, bound to one decision instant, the session
  close + 15 minutes (phase 1's P2 rule, applied to the calendar's close, so
  a 13:00 half-day publishes at 13:15). It serves adjusted closes of sessions
  published by then and nothing later: anything beyond raises
  ``LookAheadError`` (check C2). Decision code sees only returns and weights
  (adjusted *levels* are rescaled by later dividends; ratios are not, C5).
* **Fill side**: ``open``/``close``/``factor`` on ``DailyMarket`` itself, which
  the engine reads when it fills and marks, and never passes to a decision.

BTC/USD is sampled on the NYSE grid: its close is the close of the last
5-minute bar ending at or before the session's ``close_at`` (13:00 on
half-days), its open the open of the bar starting at ``open_at``. A sample
more than 60 minutes stale fails the load: no silent forward-fill (C6, C11).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Final, Protocol

from backtest import parquet
from backtest.calendar import Calendar, Session
from backtest.core_fetch import BTC, CorePaths
from backtest.data import LookAheadError
from swarm.common import NEW_YORK

PUBLISHED_AFTER: Final[timedelta] = timedelta(minutes=15)
BAR: Final[timedelta] = timedelta(minutes=5)
MAX_STALENESS: Final[timedelta] = timedelta(minutes=60)


class StaleSample(RuntimeError):
    """A BTC sample would come from a bar too far from its instant."""


def dec(x: float) -> Decimal:
    """A bar price as the shortest decimal that round-trips the float (as ``costs.to_decimal``)."""
    return Decimal(repr(float(x)))


class Prices(Protocol):
    """What ``portfolio.simulate`` needs from a market: phase 1's and the core's both provide it."""

    sessions: tuple[Session, ...]

    def open(self, symbol: str, k: int) -> Decimal: ...
    def close(self, symbol: str, k: int) -> Decimal: ...
    def factor(self, symbol: str, k: int) -> Decimal: ...
    def asset_class(self, symbol: str) -> str: ...
    def view(self, k: int) -> DailyView: ...


class DailyMarket:
    def __init__(self, sessions: Sequence[Session], adjusted: Mapping[str, Mapping[date, tuple[float, float]]],
                 raw: Mapping[str, Mapping[date, tuple[float, float]]], *, crypto: frozenset[str] = frozenset()) -> None:
        self.sessions = tuple(sessions)
        self.dates = [s.date for s in self.sessions]
        self._index = {d: k for k, d in enumerate(self.dates)}
        self._crypto = crypto
        self._open: dict[str, list[float | None]] = {}
        self._close: dict[str, list[float | None]] = {}
        self._raw_close: dict[str, list[float | None]] = {}
        for symbol, bars in adjusted.items():
            self._open[symbol] = [bars[d][0] if d in bars else None for d in self.dates]
            self._close[symbol] = [bars[d][1] if d in bars else None for d in self.dates]
            r = raw.get(symbol, bars)
            self._raw_close[symbol] = [r[d][1] if d in r else None for d in self.dates]

    @classmethod
    def load(cls, paths: CorePaths, symbols: Sequence[str], *, last: date) -> DailyMarket:
        calendar = Calendar.load(paths.calendar)
        sessions = [s for s in calendar.sessions if s.date <= last]
        wanted = set(symbols)
        adjusted: dict[str, dict[date, tuple[float, float]]] = {}
        raw: dict[str, dict[date, tuple[float, float]]] = {}
        for adjustment, out in (("all", adjusted), ("raw", raw)):
            for row in parquet.read_rows(paths.daily(adjustment), "ORDER BY symbol, t"):
                if row["symbol"] in wanted:
                    out.setdefault(row["symbol"], {})[row["t"].astimezone(NEW_YORK).date()] = (row["o"], row["c"])
        crypto = frozenset({BTC} & wanted)
        if crypto:
            cols = parquet.read_numpy(paths.btc_5min, "epoch(t)::BIGINT AS t, o, c", "ORDER BY t")
            starts = [datetime.fromtimestamp(int(e), UTC) for e in cols["t"]]
            adjusted[BTC] = btc_samples(sessions, starts, [float(x) for x in cols["o"]], [float(x) for x in cols["c"]])
        missing = wanted - set(adjusted)
        if missing:
            raise ValueError(f"no bars for {sorted(missing)}")
        return cls(sessions, adjusted, raw, crypto=crypto)

    # --- fill side: the engine fills and marks with these, never decides ---------------------------

    def index(self, day: date) -> int:
        return self._index[day]

    def has(self, symbol: str, k: int) -> bool:
        return self._close[symbol][k] is not None and self._open[symbol][k] is not None

    def first_index(self, symbol: str) -> int:
        return next(k for k, c in enumerate(self._close[symbol]) if c is not None)

    def open(self, symbol: str, k: int) -> Decimal:
        value = self._open[symbol][k]
        if value is None:
            raise KeyError(f"{symbol} has no bar on {self.dates[k]}")
        return dec(value)

    def close(self, symbol: str, k: int) -> Decimal:
        value = self._close[symbol][k]
        if value is None:
            raise KeyError(f"{symbol} has no bar on {self.dates[k]}")
        return dec(value)

    def factor(self, symbol: str, k: int) -> Decimal:
        """Adjusted / raw close: raw shares = adjusted units × factor (per-share fees, D2)."""
        if symbol in self._crypto:
            return Decimal(1)
        raw = self._raw_close[symbol][k]
        if raw is None:
            raise KeyError(f"{symbol} has no raw bar on {self.dates[k]}")
        return self.close(symbol, k) / dec(raw)

    def asset_class(self, symbol: str) -> str:
        return "crypto" if symbol in self._crypto else "equity"

    # --- decision side ---------------------------------------------------------------------------------

    def decision_at(self, k: int) -> datetime:
        return self.sessions[k].close_at + PUBLISHED_AFTER

    def view(self, k: int) -> DailyView:
        return DailyView(self, k)

    def published_by(self, t: datetime) -> int:
        """The last session index whose bar is readable at ``t`` (-1 if none): brute force for C2."""
        return max((k for k, s in enumerate(self.sessions) if s.close_at + PUBLISHED_AFTER <= t), default=-1)


class DailyView:
    """The market as known at session ``k``'s decision instant: adjusted closes through ``k``."""

    def __init__(self, market: DailyMarket, k: int) -> None:
        self._market = market
        self.k = k
        self.at = market.decision_at(k)
        self.date = market.dates[k]

    def closes(self, symbol: str, n: int) -> list[float]:
        """The last ``n`` adjusted closes, ending at this view's session."""
        first = self.k - n + 1
        if first < 0:
            raise LookAheadError(f"{symbol}: {n} closes before {self.date} do not exist in the data")
        series = self._market._close[symbol][first:self.k + 1]
        if any(c is None for c in series):
            raise KeyError(f"{symbol} lacks a close in the {n} sessions to {self.date}")
        return [float(c) for c in series]  # type: ignore[arg-type]

    def close(self, symbol: str) -> Decimal:
        return self._market.close(symbol, self.k)

    def at_or_before(self, t: datetime) -> bool:
        return t <= self.at

    def require(self, k: int) -> None:
        """Decision code may ask about session ``k`` only if it was published by now."""
        if k >= len(self._market.sessions):
            raise LookAheadError(f"session index {k} is beyond the data: not published at {self.at}")
        if self._market.decision_at(k) > self.at:
            raise LookAheadError(f"session {self._market.dates[k]} is not published at {self.at}")


def btc_samples(sessions: Sequence[Session], t: Sequence[datetime], o: Sequence[float],
                c: Sequence[float]) -> dict[date, tuple[float, float]]:
    """BTC/USD (open, close) per NYSE session from 5-minute bars (design §2), for sessions the bars cover."""
    out: dict[date, tuple[float, float]] = {}
    if not t:
        return out
    starts = list(t)
    for s in sessions:
        if s.close_at < starts[0] + BAR or s.open_at > starts[-1]:
            continue
        j = bisect_left(starts, s.open_at)
        if j >= len(starts) or starts[j] - s.open_at > MAX_STALENESS:
            raise StaleSample(f"BTC open on {s.date}: no bar within {MAX_STALENESS} after {s.open_at}")
        i = bisect_right(starts, s.close_at - BAR) - 1
        if i < 0 or s.close_at - (starts[i] + BAR) > MAX_STALENESS:
            raise StaleSample(f"BTC close on {s.date}: last bar ends more than {MAX_STALENESS} before {s.close_at}")
        out[s.date] = (o[j], c[i])
    return out


class Phase1Prices:
    """Phase 1's adjusted daily bars as ``Prices``, so ``run_hold`` runs through the core's engine (C7).
    Fees there are charged on adjusted units (``factor`` 1), as phase 1 did."""

    def __init__(self, daily: Mapping[str, Sequence[object]], symbols: Sequence[str], sessions: Sequence[Session]) -> None:
        self.sessions = tuple(sessions)
        self._bars = {s: {b.t.astimezone(NEW_YORK).date(): b for b in daily[s]} for s in symbols}  # type: ignore[attr-defined]

    def open(self, symbol: str, k: int) -> Decimal:
        return dec(self._bars[symbol][self.sessions[k].date].o)  # type: ignore[attr-defined]

    def close(self, symbol: str, k: int) -> Decimal:
        return dec(self._bars[symbol][self.sessions[k].date].c)  # type: ignore[attr-defined]

    def factor(self, symbol: str, k: int) -> Decimal:
        return Decimal(1)

    def asset_class(self, symbol: str) -> str:
        return "equity"

    def view(self, k: int) -> DailyView:
        raise LookAheadError("phase 1's fixed weights need no view")
