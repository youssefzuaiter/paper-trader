"""Samples: one row per (event, watched symbol), features as of ``made_at``, a label (design §2, §6).

**Legacy mode** reproduces ``models/return_model/report.md``: it calls the
training script's own ``build_dataset`` on the cached articles and bars,
in the cache's order (ties in publication time keep file order, which the
sample digest depends on). It breaks P4 on purpose — the label enters at
the open of the 30-minute bar *starting* at the publication time — and
serves only the reproduction.

**v2** labels what the live path could have traded:

* ``made_at = known_at + signal_latency`` (P1); the router decides at the
  next minute boundary and an order can fill ``latency_bars`` later (P4);
* **night** events, known between a session's close and the next open, are
  S2's: its pre-open scheduler scores them at open + 1 minute, so entry =
  open of the 09:32 bar;
* **session** events: entry = open of the first bar at or after the fill
  time, in the same session; one too late to fill before the close rolls
  to the next session's scheduler entry (``rolled``);
* exit = the session's last regular close; ``y = exit / entry - 1 > 0.25%``.

Both are raw prices: intraday returns do not depend on the adjustment.
Features are the production ``daily_features`` / ``feature_vector`` on a
``PitView`` at ``made_at``; ``published_at`` sets ``regular_hours`` as the
live Inference Agent does.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np

from backtest.calendar import Calendar
from backtest.data import (
    DAILY_CACHE,
    INTRADAY_CACHE,
    NEWS_CACHE,
    TRAINING_CACHE,
    MarketData,
    load_bar_cache,
    load_news_cache,
)
from backtest.events import Event
from backtest.lockbox import LOCKBOX_SESSIONS_FROM, event_readable
from swarm.common import NEW_YORK
from swarm.features import FEATURE_NAMES, MIN_DAILY_BARS, daily_features, feature_vector
from swarm.train_return_model import LABEL_MIN_RETURN, MAX_SYMBOLS_PER_ARTICLE, Dataset, RegularBars, build_dataset

logger = logging.getLogger("backtest.dataset")

LEGACY: Final[str] = "legacy"
V2: Final[str] = "v2"
FINBERT_CACHE: Final[str] = "finbert_scores.json.gz"


@dataclass(frozen=True)
class DatasetConfig:
    mode: str = V2
    signal_latency: timedelta = timedelta(seconds=60)
    latency_bars: int = 1
    scheduler_offset: timedelta = timedelta(minutes=1)
    label_min_return: float = LABEL_MIN_RETURN

    def as_params(self) -> dict[str, object]:
        return {"mode": self.mode, "signal_latency_s": self.signal_latency.total_seconds(),
                "latency_bars": self.latency_bars, "scheduler_offset_s": self.scheduler_offset.total_seconds(),
                "label_min_return": self.label_min_return}


@dataclass
class Samples:
    X: np.ndarray               # (n, len(FEATURE_NAMES))
    y: np.ndarray               # int
    fwd: np.ndarray             # label return, fraction
    published_at: np.ndarray    # epoch seconds
    known_at: np.ndarray
    made_at: np.ndarray
    entry_at: np.ndarray        # start of the entry bar
    resolved_at: np.ndarray     # end of the exit bar: when the label is known
    entry_price: np.ndarray     # raw (legacy: adjusted 30-minute)
    exit_price: np.ndarray
    idealised_fwd: np.ndarray   # no latency at all: night events at the 09:30 open
    symbol: np.ndarray
    event_id: np.ndarray
    story_id: np.ndarray
    session: np.ndarray         # calendar index of the entry session (-1 in legacy mode)
    night: np.ndarray           # S2's population: known between a close and the next open
    rolled: np.ndarray          # known in session too late to fill; entered next session
    tradeable: np.ndarray       # S3's population: known in session before close - 30 min
    day: np.ndarray             # New York date of publication, ISO (legacy's symbol-day key)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> Samples:
        return Samples(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    def digest(self) -> str:
        """As the training script names its model: sha256 of X and y."""
        return hashlib.sha256(self.X.tobytes() + self.y.tobytes()).hexdigest()

    def to_legacy_dataset(self) -> Dataset:
        return Dataset(X=self.X, y=self.y, fwd=self.fwd, ts=self.published_at, symbol=self.symbol, day=self.day,
                       tradeable=self.tradeable)


def load_sentiment(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    """FinBERT probabilities by headline, read-only (scoring is ``score_new_headlines``)."""
    scores: dict[str, list[float]] = {}
    for path in paths:
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                scores.update(json.load(f))
    return {h: np.asarray(v) for h, v in scores.items()}


# --- legacy -----------------------------------------------------------------------------

def build_legacy(watch: Sequence[str], training_cache: Path = TRAINING_CACHE) -> Samples:
    """``report.md``'s 33,312 samples, via ``train_return_model.build_dataset`` itself."""
    articles = load_news_cache(training_cache / NEWS_CACHE)
    # P7: nothing from lock-box sessions enters, even as label padding.
    daily = {s: [b for b in bars if b.t.astimezone(NEW_YORK).date() < LOCKBOX_SESSIONS_FROM]
             for s, bars in load_bar_cache(training_cache / DAILY_CACHE, watch).items()}
    intraday = {s: [b for b in bars if b.t.astimezone(NEW_YORK).date() < LOCKBOX_SESSIONS_FROM]
                for s, bars in load_bar_cache(training_cache / INTRADAY_CACHE, watch).items()}
    # main_async's filter, verbatim: roundups and empty headlines never reach build_dataset
    focused = [a for a in articles if 0 < len(a.get("symbols") or []) <= MAX_SYMBOLS_PER_ARTICLE
               and (a.get("headline") or "").strip()]
    cached = load_sentiment([training_cache / FINBERT_CACHE])
    missing = {a["headline"].strip() for a in focused} - set(cached)
    if missing:
        raise KeyError(f"{len(missing)} headlines have no cached FinBERT score; legacy mode never scores")
    ds = build_dataset(focused, daily, intraday, cached, frozenset(watch))

    regular = {s: RegularBars.build(b) for s, b in intraday.items()}
    n = len(ds.y)
    entry_at, resolved_at, entry_price, exit_price = (np.empty(n) for _ in range(4))
    for i in range(n):
        bars = regular[str(ds.symbol[i])]
        _, entry, exit_ = bars.forward_return(datetime.fromtimestamp(ds.ts[i], UTC))
        k = bisect_left(bars.starts, entry)
        entry_at[i] = entry.timestamp()
        resolved_at[i] = (exit_ + timedelta(minutes=30)).timestamp()
        entry_price[i], exit_price[i] = bars.opens[k], bars.closes[bars.last_index[bars.session_of[k]]]
    empty = np.array([""] * n)
    night = ds.X[:, FEATURE_NAMES.index("regular_hours")] == 0.0
    return Samples(X=ds.X, y=ds.y, fwd=ds.fwd, published_at=ds.ts, known_at=ds.ts, made_at=ds.ts,
                   entry_at=entry_at, resolved_at=resolved_at, entry_price=entry_price, exit_price=exit_price,
                   idealised_fwd=ds.fwd, symbol=ds.symbol, event_id=empty, story_id=empty,
                   session=np.full(n, -1), night=night, rolled=np.zeros(n, dtype=bool), tradeable=ds.tradeable,
                   day=ds.day)


# --- v2 -----------------------------------------------------------------------------------

def _ceil_minute(ts: datetime) -> datetime:
    floor = ts.replace(second=0, microsecond=0)
    return floor if floor == ts else floor + timedelta(minutes=1)


@dataclass(frozen=True)
class Entry:
    session: int      # calendar index
    bar: int          # index into the symbol's minute series
    night: bool
    rolled: bool


def entry_for(known_at: datetime, made_at: datetime, calendar: Calendar, market: MarketData, symbol: str,
              cfg: DatasetConfig) -> Entry | None:
    """The bar the live path could first fill on."""
    series = market.minute[symbol]
    known_session = calendar.session_at(known_at)
    night = known_session is None
    rolled = False
    if night:
        session = calendar.next_session(known_at)
    else:
        session = known_session
        fill_from = _ceil_minute(made_at) + timedelta(minutes=cfg.latency_bars)
        if fill_from >= session.close_at:
            rolled, session = True, calendar.shift(session, 1)
    if session is None:
        return None
    if night or rolled:
        fill_from = session.open_at + cfg.scheduler_offset + timedelta(minutes=cfg.latency_bars)
    i = series.first_at_or_after(fill_from.timestamp())
    s = calendar.index[session.date]
    if i >= len(series) or series.session[i] != s:
        return None
    return Entry(s, i, night, rolled)


def build_v2(events: Sequence[Event], market: MarketData, sentiment: Mapping[str, np.ndarray],
             cfg: DatasetConfig | None = None) -> tuple[Samples, dict[str, int]]:
    cfg = cfg or DatasetConfig()
    if cfg.mode != V2:
        raise ValueError("build_v2 builds v2 samples")
    watch = frozenset(market.symbols)
    calendar = market.calendar
    memo: dict[tuple[str, int], dict[str, float] | None] = {}
    cols: dict[str, list] = {f.name: [] for f in fields(Samples) if f.name != "X"}
    rows: list[list[float]] = []
    dropped: Counter[str] = Counter()
    for event in events:
        if not event_readable(event.known_at, market.lockbox):
            dropped["not_development"] += 1
            continue
        if event.n_symbols > MAX_SYMBOLS_PER_ARTICLE:
            dropped["roundup"] += 1
            continue
        if not event.headline:
            dropped["no_headline"] += 1
            continue
        made_at = event.known_at + cfg.signal_latency
        view = market.as_of(made_at)
        for symbol in (s for s in dict.fromkeys(event.symbols) if s in watch):
            history = view.daily(symbol)
            key = (symbol, len(history))
            if key not in memo:
                memo[key] = daily_features(history) if len(history) >= MIN_DAILY_BARS else None
            feats = memo[key]
            if feats is None:
                dropped["history"] += 1
                continue
            entry = entry_for(event.known_at, made_at, calendar, market, symbol, cfg)
            if entry is None:
                dropped["no_forward_data"] += 1
                continue
            series = market.minute[symbol]
            last = series.session_last(entry.bar)
            session = calendar.sessions[entry.session]
            entry_price, exit_price = float(series.o[entry.bar]), float(series.c[last])
            ret = exit_price / entry_price - 1
            first = series.first_at_or_after(session.open_at.timestamp())
            ideal_bar = first if entry.night or entry.rolled else series.first_at_or_after(event.known_at.timestamp())
            rows.append(feature_vector(sentiment=sentiment[event.headline], daily=feats,
                                       n_symbols=event.n_symbols, published_at=event.published_at))
            cols["y"].append(int(ret > cfg.label_min_return))
            cols["fwd"].append(ret)
            cols["published_at"].append(event.published_at.timestamp())
            cols["known_at"].append(event.known_at.timestamp())
            cols["made_at"].append(made_at.timestamp())
            cols["entry_at"].append(float(series.start[entry.bar]))
            cols["resolved_at"].append(float(series.start[last] + series.seconds))
            cols["entry_price"].append(entry_price)
            cols["exit_price"].append(exit_price)
            cols["idealised_fwd"].append(exit_price / float(series.o[ideal_bar]) - 1)
            cols["symbol"].append(symbol)
            cols["event_id"].append(event.event_id)
            cols["story_id"].append(event.story_id or "")
            cols["session"].append(entry.session)
            cols["night"].append(entry.night)
            cols["rolled"].append(entry.rolled)
            cols["tradeable"].append(not entry.night and not entry.rolled and event.known_at
                                     < calendar.session_at(event.known_at).close_at - timedelta(minutes=30))
            cols["day"].append(event.published_at.astimezone(NEW_YORK).date().isoformat())
    X = np.asarray(rows, dtype=float).reshape(-1, len(FEATURE_NAMES))
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays["y"] = arrays["y"].astype(int)
    for flag in ("night", "rolled", "tradeable"):
        arrays[flag] = arrays[flag].astype(bool)
    order = np.lexsort((arrays["symbol"], arrays["event_id"], arrays["made_at"])) if len(X) else np.arange(0)
    samples = Samples(X=X[order], **{k: v[order] for k, v in arrays.items()})
    logger.info("v2 dataset: %d samples; dropped %s", len(samples), dict(dropped))
    return samples, dict(dropped)
