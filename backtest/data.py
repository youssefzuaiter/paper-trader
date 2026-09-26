"""Point-in-time market data (design §2: rules P2-P4 and P7).

Decision code (features, strategies, the router's quotes) sees the market
only through a ``PitView`` bound to one instant. It can read:

* ``daily(symbol)``: adjusted daily bars of sessions completed and
  published by then (P2, ``features.completed_sessions``' 16:15 rule);
* ``quote_bar(symbol)``: the last regular 1-minute bar that *ended* at or
  before the instant (P3: live quotes are real time, so a bar is usable
  from its end);
* ``input_bars(symbol, since)``: intraday bars as a model input, only
  those that ended 16 minutes before the instant (P3; no model uses any today).

Anything beyond the instant raises ``LookAheadError``. Fill code reads
``fill_index`` instead, which never feeds a decision (P4 is asserted by
the simulated broker). Lock-box sessions (P7) are quarantined at load:
without a ``LockBoxKey`` they are not in memory at all.

Prices: 1-minute bars are **raw** (what traded), daily bars **adjusted**
(what training and serving compute features from). ``factor`` converts.
"""

from __future__ import annotations

import gzip
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np

from backtest import parquet
from backtest.calendar import Calendar
from backtest.fetch import Paths
from backtest.lockbox import LOCKBOX_SESSIONS_FROM, LockBoxError, LockBoxKey, session_readable
from swarm.alpaca_data import SIP_DELAY, Bar
from swarm.common import NEW_YORK
from swarm.features import SESSION_PUBLISHED_AT, session_date

#: An intraday bar may be a model input only this long after it ended (the free plan's SIP delay).
INPUT_DELAY: Final[timedelta] = SIP_DELAY
TRAINING_CACHE: Final[Path] = Path(".cache/training")
DAILY_CACHE: Final[str] = "bars_1Day_AAPL-MSFT-NVDA-TSLA-AMZN-GOOGL-META-AMD_2023-09-04_2026-09-25.json.gz"
INTRADAY_CACHE: Final[str] = "bars_30Min_AAPL-MSFT-NVDA-TSLA-AMZN-GOOGL-META-AMD_2024-01-02_2026-09-25.json.gz"
NEWS_CACHE: Final[str] = "news_AAPL-MSFT-NVDA-TSLA-AMZN-GOOGL-META-AMD_2024-01-02_2026-09-18.jsonl.gz"


class LookAheadError(AssertionError):
    """Decision code asked for data that did not exist yet."""


@dataclass(frozen=True, slots=True)
class PriceBar:
    symbol: str
    start: datetime  # UTC
    seconds: int
    o: float
    h: float
    l: float
    c: float
    v: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.seconds)


@dataclass(frozen=True)
class BarSeries:
    """One symbol's regular-session bars of one length, as arrays.

    ``session`` is each bar's session index (``Calendar.index``), so "the
    same session" and "the session's last bar" are array lookups."""

    symbol: str
    seconds: int
    start: np.ndarray    # int64, epoch seconds
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    session: np.ndarray  # int32

    def __len__(self) -> int:
        return len(self.start)

    def first_at_or_after(self, epoch: float) -> int:
        """Index of the first bar starting at or after ``epoch`` (``len`` if none)."""
        return int(np.searchsorted(self.start, epoch, side="left"))

    def last_ended_by(self, epoch: float) -> int:
        """Index of the last bar that ended at or before ``epoch`` (-1 if none)."""
        return int(np.searchsorted(self.start + self.seconds, epoch, side="right")) - 1

    def session_last(self, i: int) -> int:
        """Index of the last bar in bar ``i``'s session."""
        return int(np.searchsorted(self.session, self.session[i], side="right")) - 1

    def bar(self, i: int) -> PriceBar:
        return PriceBar(self.symbol, datetime.fromtimestamp(int(self.start[i]), UTC), self.seconds,
                        float(self.o[i]), float(self.h[i]), float(self.l[i]), float(self.c[i]), float(self.v[i]))

    def restricted(self, keep: np.ndarray) -> BarSeries:
        return BarSeries(self.symbol, self.seconds, self.start[keep], self.o[keep], self.h[keep], self.l[keep],
                         self.c[keep], self.v[keep], self.session[keep])


def regular_series(symbol: str, seconds: int, start: np.ndarray, o: np.ndarray, h: np.ndarray, lo: np.ndarray,
                   c: np.ndarray, v: np.ndarray, calendar: Calendar) -> BarSeries:
    """Keep the bars that start inside a calendar session's regular hours."""
    opens = np.array([s.open_at.timestamp() for s in calendar.sessions])
    closes = np.array([s.close_at.timestamp() for s in calendar.sessions])
    idx = np.searchsorted(opens, start, side="right") - 1
    ok = (idx >= 0) & (start < closes[np.clip(idx, 0, None)])
    order = np.argsort(start[ok], kind="stable")
    return BarSeries(symbol, seconds, start[ok][order].astype(np.int64), o[ok][order], h[ok][order],
                     lo[ok][order], c[ok][order], v[ok][order], idx[ok][order].astype(np.int32))


class MarketData:
    def __init__(self, calendar: Calendar, minute: Mapping[str, BarSeries], daily: Mapping[str, list[Bar]],
                 raw_close: Mapping[str, Mapping[date, float]], *, lockbox: LockBoxKey | None = None) -> None:
        self.calendar = calendar
        self._lockbox = lockbox
        self.symbols = tuple(minute)
        self.minute = {s: self._quarantine_series(b) for s, b in minute.items()}
        self.daily = {s: [b for b in bars if self._readable(session_date(b))] for s, bars in daily.items()}
        self._daily_cutoffs = {
            s: [datetime.combine(session_date(b), SESSION_PUBLISHED_AT, NEW_YORK).timestamp() for b in bars]
            for s, bars in self.daily.items()
        }
        self._adj_close = {s: {session_date(b): b.c for b in bars} for s, bars in self.daily.items()}
        self._raw_close = {s: {d: c for d, c in closes.items() if self._readable(d)}
                           for s, closes in raw_close.items()}

    @property
    def lockbox(self) -> LockBoxKey | None:
        return self._lockbox

    def _guarded(self) -> bool:
        """P7 in one place: every lock-box decision below asks this."""
        return self._lockbox is None

    def _readable(self, session: date) -> bool:
        return not self._guarded() or session_readable(session, None)

    def _quarantine_series(self, series: BarSeries) -> BarSeries:
        first_locked = self.calendar.index.get(LOCKBOX_SESSIONS_FROM)
        if not self._guarded() or first_locked is None:
            return series
        return series.restricted(series.session < first_locked)

    @classmethod
    def load(cls, paths: Paths, *, symbols: Sequence[str], training_cache: Path = TRAINING_CACHE,
             lockbox: LockBoxKey | None = None) -> MarketData:
        calendar = Calendar.load(paths.calendar)
        minute = {}
        for symbol in symbols:
            cols = parquet.read_numpy(paths.bars_1min(symbol), "epoch(t)::BIGINT AS t, o, h, l, c, v", "ORDER BY t")
            minute[symbol] = regular_series(symbol, 60, cols["t"], cols["o"], cols["h"], cols["l"], cols["c"],
                                            cols["v"], calendar)
        raw_close: dict[str, dict[date, float]] = {s: {} for s in symbols}
        for row in parquet.read_rows(paths.bars_1day_raw, "ORDER BY symbol, t"):
            if row["symbol"] in raw_close:
                raw_close[row["symbol"]][row["t"].astimezone(NEW_YORK).date()] = row["c"]
        daily = load_bar_cache(training_cache / DAILY_CACHE, symbols)
        return cls(calendar, minute, daily, raw_close, lockbox=lockbox)

    # --- decision side --------------------------------------------------------------

    def as_of(self, t: datetime) -> PitView:
        if t.tzinfo is None:
            raise ValueError("PitView needs an aware instant")
        if self._guarded() and t >= self._lockbox_start():
            raise LockBoxError(f"{t.isoformat()} is inside the lock-box")
        return PitView(self, t)

    def _lockbox_start(self) -> datetime:
        session = self.calendar.session(LOCKBOX_SESSIONS_FROM)
        return session.open_at if session else datetime.max.replace(tzinfo=UTC)

    # --- fill side (never feeds a decision) ------------------------------------------------

    def fill_index(self, symbol: str, not_before: datetime) -> int:
        """First bar a fill may use: the first starting at or after ``not_before``."""
        return self.minute[symbol].first_at_or_after(not_before.timestamp())

    def factor(self, symbol: str, session: date) -> float:
        """Adjusted close / raw close: multiply a raw price by it to get adjusted."""
        try:
            return self._adj_close[symbol][session] / self._raw_close[symbol][session]
        except KeyError:
            if not self._readable(session):
                raise LockBoxError(f"{symbol} {session} is inside the lock-box") from None
            raise


class PitView:
    """The market as it was known at one instant."""

    __slots__ = ("_epoch", "_m", "t")

    def __init__(self, market: MarketData, t: datetime) -> None:
        self._m = market
        self.t = t
        self._epoch = t.timestamp()

    def daily(self, symbol: str) -> list[Bar]:
        """Completed, published sessions (adjusted), oldest first — exactly
        ``features.completed_sessions(bars, t)`` (a test holds them equal)."""
        n = bisect_right(self._m._daily_cutoffs[symbol], self._epoch)
        return self._m.daily[symbol][:n]

    def quote_bar(self, symbol: str) -> PriceBar | None:
        series = self._m.minute[symbol]
        i = series.last_ended_by(self._epoch)
        if i < 0:
            return None
        bar = series.bar(i)
        if bar.end > self.t:
            raise LookAheadError(f"{symbol} bar ending {bar.end.isoformat()} read at {self.t.isoformat()}")
        return bar

    def input_bars(self, symbol: str, since: datetime) -> list[PriceBar]:
        """Intraday bars as a model input: only those that ended by ``t - INPUT_DELAY``."""
        series = self._m.minute[symbol]
        lo = series.first_at_or_after(since.timestamp())
        hi = series.last_ended_by(self._epoch - INPUT_DELAY.total_seconds())
        bars = [series.bar(i) for i in range(lo, hi + 1)]
        if bars and bars[-1].end > self.t - INPUT_DELAY:
            raise LookAheadError(f"{symbol} input bar ending {bars[-1].end.isoformat()} at {self.t.isoformat()}")
        return bars


def load_bar_cache(path: Path, symbols: Sequence[str]) -> dict[str, list[Bar]]:
    """The training cache's bars (``swarm.train_return_model.fetch_bars`` format)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    return {s: [Bar.from_api(b) for b in raw[s]] for s in symbols}


def load_news_cache(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
