"""Half-spreads measured from the core's quote sample (owner decision O10, 2026-10-01).

The rule was fixed in code before any number was looked at:

* a quote's half-spread is (ask − bid) / (2 · mid), mid = (ask + bid) / 2;
* quotes with a non-positive bid or ask, or ask ≤ bid (locked or crossed),
  are dropped;
* only the ``open`` window counts: the first minute after the open, when the
  core trades (its orders fill at the open, so phase 1's opening multiplier
  *m* is not applied on top: the spread is measured where it is paid);
* each session gets equal weight: a session's median half-spread is taken
  first, so one busy session's thousands of quotes do not swamp the rest;
* the three cost levels are the 50th, 75th and 95th percentiles of those
  per-session medians (optimistic, central, pessimistic), in bps, rounded up
  to 0.1 bp.

The result is pasted into ``costs.CORE_HALF_SPREAD_BPS`` with the sample's
hash, so a change to the sample or the rule shows up in review.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import ROUND_CEILING, Decimal
from statistics import median
from typing import Any, Final

PERCENTILES: Final[dict[str, float]] = {"optimistic": 0.50, "central": 0.75, "pessimistic": 0.95}
WINDOW: Final[str] = "open"


def half_spread_bps(bid: float, ask: float) -> float | None:
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return (ask - bid) / (ask + bid) * 1e4  # (ask − bid) / (2 · mid) in bps


def percentile(sorted_xs: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default), stdlib only."""
    if not sorted_xs:
        raise ValueError("no values")
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def measure(quotes: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per symbol: the three levels (bps), and how many sessions and quotes they rest on."""
    by_session: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    dropped: dict[str, int] = defaultdict(int)
    for q in quotes:
        if q["window_name"] != WINDOW:
            continue
        h = half_spread_bps(q["bid_price"], q["ask_price"])
        if h is None:
            dropped[q["symbol"]] += 1
            continue
        by_session[q["symbol"]][q["session"]].append(h)
    out: dict[str, dict[str, Any]] = {}
    for symbol, sessions in sorted(by_session.items()):
        medians = sorted(median(v) for v in sessions.values())
        levels = {name: Decimal(repr(percentile(medians, p))).quantize(Decimal("0.1"), ROUND_CEILING)
                  for name, p in PERCENTILES.items()}
        out[symbol] = {**levels, "sessions": len(medians), "quotes": sum(len(v) for v in sessions.values()),
                       "dropped": dropped[symbol]}
    return out
