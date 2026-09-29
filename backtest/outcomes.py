"""What happened next, for every prediction, traded or not (design §7: event memory).

Each prediction is resolved from its v2 entry, the open of the first bar the
live path could fill on (``reference_at`` / ``reference_price``), at six
horizons:

* ``5m``, ``30m``, ``2h``: the close of the bar ending that long after the
  entry, raw prices; a horizon past the session's close stops at the close
  and is flagged ``truncated``;
* ``close``: the session's last regular close (the v2 label's own exit);
* ``1d``, ``1mo``: the adjusted close 1 and 21 sessions later, against the
  entry converted to adjusted dollars (dividends reinvested, as S0/S1).

Each carries the maximum favourable and adverse excursion on the way.
Horizons beyond the data (or inside the lock-box) are ``no_data``.

Nothing that builds features, fits models or chooses parameters may import
this module (test L9): outcomes are for accountability, not for tuning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import numpy as np

from backtest.data import MarketData
from swarm.common import NEW_YORK

INTRADAY: Final[dict[str, timedelta]] = {"5m": timedelta(minutes=5), "30m": timedelta(minutes=30),
                                         "2h": timedelta(hours=2)}
DAILY: Final[dict[str, int]] = {"1d": 1, "1mo": 21}
HORIZONS: Final[tuple[str, ...]] = ("5m", "30m", "2h", "close", "1d", "1mo")


def resolve(predictions: Sequence[Mapping[str, Any]], market: MarketData) -> list[dict[str, Any]]:
    calendar = market.calendar
    daily = {s: {b.t.astimezone(NEW_YORK).date(): b for b in bars} for s, bars in market.daily.items()}
    out: list[dict[str, Any]] = []
    for p in predictions:
        symbol, pid = p["symbol"], p["prediction_id"]
        series = market.minute[symbol]
        i = series.first_at_or_after(p["reference_at"].timestamp())
        if i >= len(series) or series.start[i] != int(p["reference_at"].timestamp()):
            out.extend(_empty(pid, h) for h in HORIZONS)
            continue
        entry = float(series.o[i])
        last = series.session_last(i)
        session = calendar.sessions[int(series.session[i])]

        for name, span in INTRADAY.items():
            target = p["reference_at"].timestamp() + span.total_seconds()
            j = min(series.last_ended_by(target), last)
            truncated = target > series.start[last] + series.seconds
            out.append(_intraday(pid, name, series, i, j, entry, "truncated" if truncated else "ok"))
        out.append(_intraday(pid, "close", series, i, last, entry, "ok"))

        factor = market.factor(symbol, session.date)
        entry_adj = entry * factor
        rest_high = float(series.h[i:last + 1].max()) * factor
        rest_low = float(series.l[i:last + 1].min()) * factor
        for name, n in DAILY.items():
            later = [calendar.shift(session, k) for k in range(1, n + 1)]
            bars = [daily[symbol].get(s.date) if s is not None else None for s in later]
            if any(b is None for b in bars):
                out.append(_empty(pid, name))
                continue
            high = max(rest_high, *(b.h for b in bars))
            low = min(rest_low, *(b.l for b in bars))
            out.append({"prediction_id": pid, "horizon": name, "resolved_at": later[-1].close_at,
                        "ret": bars[-1].c / entry_adj - 1, "max_favourable": high / entry_adj - 1,
                        "max_adverse": low / entry_adj - 1, "price_space": "adjusted", "status": "ok"})
    return out


def _intraday(pid: str, name: str, series: Any, i: int, j: int, entry: float, status: str) -> dict[str, Any]:
    return {"prediction_id": pid, "horizon": name,
            "resolved_at": datetime.fromtimestamp(int(series.start[j] + series.seconds), UTC),
            "ret": float(series.c[j]) / entry - 1, "max_favourable": float(np.max(series.h[i:j + 1])) / entry - 1,
            "max_adverse": float(np.min(series.l[i:j + 1])) / entry - 1, "price_space": "raw", "status": status}


def _empty(pid: str, horizon: str) -> dict[str, Any]:
    return {"prediction_id": pid, "horizon": horizon, "resolved_at": None, "ret": None, "max_favourable": None,
            "max_adverse": None, "price_space": "adjusted" if horizon in DAILY else "raw", "status": "no_data"}
