"""Settings every swarm agent shares."""

from __future__ import annotations

import os
from typing import Final
from zoneinfo import ZoneInfo

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")

#: Redis Streams between the stages.
NEWS_RAW: Final[str] = "news.raw"
NEWS_ENRICHED: Final[str] = "news.enriched"
SIGNALS_EVALUATED: Final[str] = "signals.evaluated"
#: Approximate cap per stream; news volume for 8 names is a few hundred a day.
STREAM_MAXLEN: Final[int] = 10_000

#: Alpaca's free plan allows 30 symbols per stream subscription. These
#: are the names the monolith's scenarios and ATR calibration used.
DEFAULT_WATCHLIST: Final[tuple[str, ...]] = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD")
MAX_WATCHLIST: Final[int] = 30


def watchlist() -> tuple[str, ...]:
    raw = os.getenv("WATCHLIST", "").strip()
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in raw.split(",") if s.strip())) or DEFAULT_WATCHLIST
    if len(symbols) > MAX_WATCHLIST:
        raise ValueError(f"WATCHLIST has {len(symbols)} symbols; the free data plan streams at most {MAX_WATCHLIST}")
    return symbols


def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")
