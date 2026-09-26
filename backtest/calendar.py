"""The exchange calendar: sessions, holidays and 13:00 half-days, from Alpaca's ``/v2/calendar``.

Sessions come from the calendar, never from weekday rules (design §2). The
simulated broker's ``clock()`` is served from here in Alpaca's own shape,
so the router's ``session_close`` logic handles a half-day exactly as it
does live: no entries from 12:30, everything sold from 12:50.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from backtest import parquet
from swarm.common import NEW_YORK


@dataclass(frozen=True, slots=True)
class Session:
    date: date
    open_at: datetime   # UTC
    close_at: datetime  # UTC
    half_day: bool

    def contains(self, ts: datetime) -> bool:
        return self.open_at <= ts < self.close_at

    @property
    def minutes(self) -> int:
        return int((self.close_at - self.open_at).total_seconds() // 60)


class Calendar:
    def __init__(self, sessions: Sequence[Session]) -> None:
        self.sessions: tuple[Session, ...] = tuple(sorted(sessions, key=lambda s: s.date))
        self._dates = [s.date for s in self.sessions]
        self._opens = [s.open_at for s in self.sessions]
        self._closes = [s.close_at for s in self.sessions]
        self._by_date = {s.date: s for s in self.sessions}
        self.index = {s.date: i for i, s in enumerate(self.sessions)}

    @classmethod
    def load(cls, path: Path) -> Calendar:
        return cls([Session(r["date"], r["open_at"].astimezone(UTC), r["close_at"].astimezone(UTC), r["half_day"])
                    for r in parquet.read_rows(path, "ORDER BY date")])

    @classmethod
    def weekday_rule(cls, start: date, end: date) -> Calendar:
        """Every weekday 09:30-16:00. Wrong on holidays and half-days: it
        exists only as the broken twin of test L11."""
        sessions, d = [], start
        while d <= end:
            if d.weekday() < 5:
                sessions.append(Session(d, datetime.combine(d, time(9, 30), NEW_YORK).astimezone(UTC),
                                        datetime.combine(d, time(16, 0), NEW_YORK).astimezone(UTC), False))
            d += timedelta(days=1)
        return cls(sessions)

    def session(self, day: date) -> Session | None:
        return self._by_date.get(day)

    def session_at(self, ts: datetime) -> Session | None:
        """The session whose regular hours contain ``ts``."""
        i = bisect_right(self._opens, ts) - 1
        return self.sessions[i] if i >= 0 and self.sessions[i].contains(ts) else None

    def next_session(self, ts: datetime) -> Session | None:
        """The first session opening strictly after ``ts``."""
        i = bisect_right(self._opens, ts)
        return self.sessions[i] if i < len(self.sessions) else None

    def previous_session(self, ts: datetime) -> Session | None:
        """The last session that closed at or before ``ts``."""
        i = bisect_right(self._closes, ts) - 1
        return self.sessions[i] if i >= 0 else None

    def sessions_between(self, start: date, end: date) -> list[Session]:
        """Sessions with ``start <= date <= end``."""
        return list(self.sessions[bisect_left(self._dates, start):bisect_right(self._dates, end)])

    def shift(self, session: Session, n: int) -> Session | None:
        """The session ``n`` sessions after (``n < 0``: before) ``session``."""
        i = self.index[session.date] + n
        return self.sessions[i] if 0 <= i < len(self.sessions) else None

    def clock(self, ts: datetime) -> dict[str, Any]:
        """Alpaca's ``/v2/clock`` at ``ts``: New York offsets, as Alpaca sends them."""
        current = self.session_at(ts)
        upcoming = self.next_session(ts)
        next_close = current.close_at if current is not None else (upcoming.close_at if upcoming else None)
        return {
            "timestamp": ts.astimezone(NEW_YORK).isoformat(),
            "is_open": current is not None,
            "next_open": upcoming.open_at.astimezone(NEW_YORK).isoformat() if upcoming else None,
            "next_close": next_close.astimezone(NEW_YORK).isoformat() if next_close else None,
        }
