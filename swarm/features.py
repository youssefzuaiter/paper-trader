"""Feature maths shared, line for line, by training and serving.

The single rule that keeps the model honest: **a feature may only use
information that existed when the article was published.** Daily bars
enter only once their session has closed and Alpaca's free-plan SIP delay
has passed (``completed_sessions``). A bar from the article's own trading
day never leaks in before 16:15 New York time.

Every function here is pure. ``FEATURE_NAMES`` is the model's input
contract: the artifact records it at training time, and the serving side
refuses to load a model whose list differs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime, time
from typing import Final

import numpy as np

import tier0
from swarm.alpaca_data import Bar
from swarm.common import NEW_YORK

FEATURE_NAMES: Final[tuple[str, ...]] = (
    # FinBERT on the headline
    "sent_pos", "sent_neg", "sent_neu", "sent_net",
    # Quantitative Agent, from completed daily sessions
    "atr14_pct", "rv20", "gk20", "rsi14", "mom5", "mom20", "dist_sma20", "vol_z20", "gap1",
    # the article itself
    "n_symbols", "regular_hours",
)
DAILY_FEATURES: Final[tuple[str, ...]] = FEATURE_NAMES[4:13]

#: Enough history for a 20-day window plus warm-up of the Wilder averages.
MIN_DAILY_BARS: Final[int] = 30
#: Features always use exactly the most recent 40 sessions (fewer only if
#: fewer exist). Wilder averages depend on where they were seeded, so a
#: fixed window is what makes training (years of history) and serving
#: (~50 sessions fetched) compute identical numbers from identical bars.
FEATURE_WINDOW: Final[int] = 40
#: A daily bar counts once its session closed and the SIP delay passed.
SESSION_PUBLISHED_AT: Final[time] = time(16, 15)
REGULAR_OPEN: Final[time] = time(9, 30)
REGULAR_CLOSE: Final[time] = time(16, 0)
_TRADING_DAYS: Final[int] = 252


def session_date(bar: Bar) -> date:
    """Alpaca stamps a daily bar at midnight New York time of its session."""
    return bar.t.astimezone(NEW_YORK).date()


def completed_sessions(daily: Sequence[Bar], as_of: datetime) -> list[Bar]:
    """The daily bars that were final and published at ``as_of``."""
    local = as_of.astimezone(NEW_YORK)
    today = local.date()
    today_done = local.time() >= SESSION_PUBLISHED_AT
    return [b for b in daily if session_date(b) < today or (today_done and session_date(b) == today)]


def is_regular_hours(ts: datetime) -> bool:
    """Weekday 09:30–16:00 New York. Exchange holidays are not modelled:
    no news-driven entry happens on one, because the router checks
    Alpaca's clock before buying."""
    local = ts.astimezone(NEW_YORK)
    return local.weekday() < 5 and REGULAR_OPEN <= local.time() < REGULAR_CLOSE


def is_tradeable(ts: datetime) -> bool:
    """Published when the router would still open a position: regular
    hours, and not in the session's last ``tier0.LAST_ENTRY_BEFORE_CLOSE``.
    Overnight news goes stale (the router refuses signals over 120 s old)
    long before the open. Half-days are not modelled here; the router
    itself uses Alpaca's clock."""
    local = ts.astimezone(NEW_YORK)
    cutoff = (datetime.combine(local.date(), REGULAR_CLOSE) - tier0.LAST_ENTRY_BEFORE_CLOSE).time()
    return local.weekday() < 5 and REGULAR_OPEN <= local.time() < cutoff


def _wilder(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing: seed with the simple mean, then
    ``avg = (avg * (n - 1) + x) / n``."""
    out = np.empty(len(values) - period + 1)
    out[0] = values[:period].mean()
    for i in range(1, len(out)):
        out[i] = (out[i - 1] * (period - 1) + values[period - 1 + i]) / period
    return out


def daily_features(bars: Sequence[Bar]) -> dict[str, float]:
    """Volatility and technical state as of the last bar in ``bars``.

    Raises ``ValueError`` with fewer than ``MIN_DAILY_BARS`` bars.
    """
    if len(bars) < MIN_DAILY_BARS:
        raise ValueError(f"need {MIN_DAILY_BARS} daily bars, got {len(bars)}")
    bars = bars[-FEATURE_WINDOW:]
    o = np.array([b.o for b in bars])
    h = np.array([b.h for b in bars])
    lo = np.array([b.l for b in bars])
    c = np.array([b.c for b in bars])
    v = np.array([b.v for b in bars])

    prev_c = c[:-1]
    true_range = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - prev_c), np.abs(lo[1:] - prev_c)])
    atr14 = _wilder(true_range, 14)[-1]

    log_ret = np.diff(np.log(c))
    rv20 = float(np.std(log_ret[-20:], ddof=1)) * math.sqrt(_TRADING_DAYS) * 100

    # Garman–Klass: uses the whole day's range, a tighter volatility
    # estimate than close-to-close for the same window.
    gk_var = 0.5 * np.log(h / lo) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2
    gk20 = math.sqrt(max(float(gk_var[-20:].mean()), 0.0) * _TRADING_DAYS) * 100

    changes = np.diff(c)
    avg_gain = _wilder(np.clip(changes, 0, None), 14)[-1]
    avg_loss = _wilder(np.clip(-changes, 0, None), 14)[-1]
    rsi14 = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)

    prior_vol = v[-21:-1]
    vol_std = float(prior_vol.std(ddof=1))
    vol_z20 = 0.0 if vol_std == 0 else float((v[-1] - prior_vol.mean()) / vol_std)

    return {
        "atr14_pct": float(atr14 / c[-1] * 100),
        "rv20": rv20,
        "gk20": gk20,
        "rsi14": float(rsi14),
        "mom5": float((c[-1] / c[-6] - 1) * 100),
        "mom20": float((c[-1] / c[-21] - 1) * 100),
        "dist_sma20": float((c[-1] / c[-20:].mean() - 1) * 100),
        "vol_z20": vol_z20,
        "gap1": float((o[-1] / c[-2] - 1) * 100),
    }


def feature_vector(
    *, sentiment: Sequence[float], daily: dict[str, float], n_symbols: int, published_at: datetime,
) -> list[float]:
    """Assemble the model input in ``FEATURE_NAMES`` order."""
    p_pos, p_neg, p_neu = (float(x) for x in sentiment)
    row = {
        "sent_pos": p_pos, "sent_neg": p_neg, "sent_neu": p_neu, "sent_net": p_pos - p_neg,
        **{name: float(daily[name]) for name in DAILY_FEATURES},
        "n_symbols": float(n_symbols),
        "regular_hours": 1.0 if is_regular_hours(published_at) else 0.0,
    }
    return [row[name] for name in FEATURE_NAMES]
