"""The long-term core's free fetches (core design §2), cached under ``.cache/backtest/core/``.

Phase 1's files under ``.cache/backtest/`` are never read or written here, so
their manifests stay valid.

* **The exchange calendar**, 2016-01-01 → 2026-12-31.
* **Daily bars**, SIP, every instrument of the brief's §3 table, twice:
  ``all`` (split- and dividend-adjusted: total return) and ``raw`` (share
  counts, cent rounding and per-share fees). The adjustment is proportional:
  the adjusted/raw ratio steps only on ex-dates (checked 2026-10-01 on VTI and
  BIL), which is what lets decisions use adjusted *returns* point in time.
* **BTC/USD 5-minute bars** from 2021-01-01, the first day Alpaca has. The
  engine samples them on the NYSE grid; Alpaca's daily crypto bar is not used
  (its day ends at midnight UTC, not at the close).
* **A quote sample** (owner decision O10, 2026-10-01): the first minute after
  the open, when the core trades, and 12:00, on evenly spaced full sessions.
  ETF quotes span the whole history; BTC quotes exist from 2023 only.

Everything ends at ``CORE_END``, a constant: the files are hashed into every
experiment's manifest, and a later fetch must not silently extend them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

from backtest import parquet
from backtest.fetch import (
    BAR_TYPES,
    CALENDAR_TYPES,
    QUOTE_TYPES,
    ResearchData,
    calendar_columns,
    fetch_calendar,
)
from swarm.alpaca_data import _rfc3339
from swarm.common import NEW_YORK

logger = logging.getLogger("backtest.core_fetch")

#: One primary per class (O2), then the registered robustness substitutes.
PRIMARY: Final[dict[str, str]] = {"us_stocks": "VTI", "intl_stocks": "VXUS", "bonds": "BND", "gold": "IAU",
                                  "real_estate": "VNQ", "cash": "BIL"}
SUBSTITUTES: Final[tuple[str, ...]] = ("SPY", "IEF", "TLT", "GLD", "SGOV")
ETFS: Final[tuple[str, ...]] = (*PRIMARY.values(), *SUBSTITUTES)
BTC: Final[str] = "BTC/USD"

CORE_START: Final[date] = date(2016, 1, 1)
BTC_START: Final[date] = date(2021, 1, 1)
#: Through the 2026-09-30 session, after-hours included.
CORE_END: Final[datetime] = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
CALENDAR_END: Final[date] = date(2026, 12, 31)

QUOTE_SESSIONS: Final[int] = 24
BTC_QUOTES_FROM: Final[date] = date(2023, 3, 1)  # Alpaca returns no crypto quotes for 2022 (probed 2026-10-01)
#: (name, minutes after the open) of each one-minute window; ``None`` is 12:00 New York.
QUOTE_WINDOWS: Final[tuple[tuple[str, int | None], ...]] = (("open", 0), ("midday", None))
#: Pages of 10,000 per window. A truncated window keeps its earliest quotes, the ones nearest the open.
QUOTE_MAX_PAGES: Final[int] = 5

DAILY_TYPES: Final[dict[str, str]] = {"symbol": "VARCHAR", **BAR_TYPES}


@dataclass(frozen=True)
class CorePaths:
    root: Path

    @classmethod
    def under(cls, backtest_cache: Path) -> CorePaths:
        return cls(backtest_cache / "core")

    @property
    def calendar(self) -> Path:
        return self.root / "calendar.parquet"

    def daily(self, adjustment: str) -> Path:
        return self.root / f"bars_1day_{adjustment}.parquet"

    @property
    def btc_5min(self) -> Path:
        return self.root / "btc_usd_5min.parquet"

    @property
    def quotes_sample(self) -> Path:
        return self.root / "quotes_sample.parquet"

    def market_files(self) -> list[Path]:
        """What the simulation reads: every core experiment's manifest."""
        return [self.calendar, self.daily("all"), self.daily("raw"), self.btc_5min]

    def all_files(self) -> list[Path]:
        return [*self.market_files(), self.quotes_sample]


class CoreData(ResearchData):
    """Crypto bars and quotes: Alpaca's ``v1beta3`` endpoints, same pacing and retries."""

    async def crypto_bars(self, symbol: str, *, timeframe: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"symbols": symbol, "timeframe": timeframe, "start": _rfc3339(start),
                                  "end": _rfc3339(end), "limit": 10_000, "sort": "asc"}
        out: list[dict[str, Any]] = []
        while True:
            page = await self._get("/v1beta3/crypto/us/bars", params)
            out.extend((page.get("bars") or {}).get(symbol, []))
            if len(out) and len(out) % 100_000 < 10_000:
                logger.info("%s %s: %d bars, at %s", symbol, timeframe, len(out), out[-1]["t"])
            token = page.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    async def crypto_quotes(self, symbol: str, *, start: datetime, end: datetime,
                            max_pages: int) -> tuple[list[dict[str, Any]], bool]:
        params: dict[str, Any] = {"symbols": symbol, "start": _rfc3339(start), "end": _rfc3339(end),
                                  "limit": 10_000, "sort": "asc"}
        out: list[dict[str, Any]] = []
        for _ in range(max_pages):
            page = await self._get("/v1beta3/crypto/us/quotes", params)
            out.extend((page.get("quotes") or {}).get(symbol, []))
            token = page.get("next_page_token")
            if not token:
                return out, False
            params["page_token"] = token
        return out, True


def _local(day: date, hh: int, mm: int) -> datetime:
    return datetime.combine(day, time(hh, mm), NEW_YORK).astimezone(UTC)


def quote_sessions(calendar: dict[str, list[Any]], first: date, n: int = QUOTE_SESSIONS) -> list[int]:
    """``n`` evenly spaced full sessions from ``first`` to the last fetched session (row indices)."""
    full = [i for i, (d, half) in enumerate(zip(calendar["date"], calendar["half_day"], strict=True))
            if first <= d < CORE_END.date() and not half]
    step = len(full) / n
    return [full[int(step * k + step / 2)] for k in range(n)]


def quote_window(open_at: datetime, day: date, offset: int | None) -> tuple[datetime, datetime]:
    start = open_at + timedelta(minutes=offset) if offset is not None else _local(day, 12, 0)
    return start, start + timedelta(minutes=1)


async def fetch_quote_sample(data: CoreData, calendar: dict[str, list[Any]]) -> tuple[dict[str, list[Any]], list[str]]:
    cols: dict[str, list[Any]] = {k: [] for k in QUOTE_TYPES}
    truncated: list[str] = []

    def keep(symbol: str, day: date, name: str, q: dict[str, Any]) -> None:
        cols["symbol"].append(symbol)
        cols["session"].append(day)
        cols["window_name"].append(name)
        cols["t"].append(datetime.fromisoformat(q["t"]))
        cols["bid_price"].append(float(q["bp"]))
        cols["ask_price"].append(float(q["ap"]))
        cols["bid_size"].append(float(q.get("bs", 0)))
        cols["ask_size"].append(float(q.get("as", 0)))
        cols["bid_exchange"].append(q.get("bx") or "")
        cols["ask_exchange"].append(q.get("ax") or "")

    plan = [(s, i) for i in quote_sessions(calendar, CORE_START) for s in ETFS]
    plan += [(BTC, i) for i in quote_sessions(calendar, BTC_QUOTES_FROM)]
    for symbol, i in plan:
        day, open_at = calendar["date"][i], calendar["open_at"][i]
        for name, offset in QUOTE_WINDOWS:
            start, end = quote_window(open_at, day, offset)
            if symbol == BTC:
                quotes, cut = await data.crypto_quotes(symbol, start=start, end=end, max_pages=QUOTE_MAX_PAGES)
            else:
                quotes, cut = await data.quotes(symbol, start=start, end=end, max_pages=QUOTE_MAX_PAGES)
            if cut:
                truncated.append(f"{symbol} {day} {name}")
            for q in quotes:
                keep(symbol, day, name, q)
        logger.info("quotes: %s %s done (%d quotes so far)", symbol, day, len(cols["t"]))
    return cols, truncated


async def fetch_core(paths: CorePaths, *, data: CoreData, calendar_keys: tuple[str, str] | None = None,
                     calendar_transport: Any = None) -> dict[str, Any]:
    """Fetch every missing file; skip the ones present (an interrupted run resumes)."""
    report: dict[str, Any] = {"fetched": [], "skipped": []}

    def note(path: Path, fetched: bool) -> None:
        report["fetched" if fetched else "skipped"].append(str(path.relative_to(paths.root)))

    missing_calendar = not paths.calendar.exists()
    if missing_calendar:
        keys = calendar_keys or (None, None)
        raw = await fetch_calendar(CORE_START, CALENDAR_END, transport=calendar_transport,
                                   key_id=keys[0], secret_key=keys[1])
        parquet.write(paths.calendar, calendar_columns(raw), CALENDAR_TYPES, order_by="date")
    note(paths.calendar, missing_calendar)
    calendar = {k: [r[k] for r in parquet.read_rows(paths.calendar, "ORDER BY date")] for k in CALENDAR_TYPES}

    for adjustment in ("all", "raw"):
        path = paths.daily(adjustment)
        missing = not path.exists()
        if missing:
            got = await data.bars(list(ETFS), timeframe="1Day", start=datetime.combine(CORE_START, time(0), UTC),
                                  end=CORE_END, adjustment=adjustment)
            cols: dict[str, list[Any]] = {k: [] for k in DAILY_TYPES}
            for symbol in ETFS:
                bars = got.get(symbol, [])
                if not bars:
                    raise RuntimeError(f"no {adjustment} daily bars for {symbol}")
                cols["symbol"].extend([symbol] * len(bars))
                for k in BAR_TYPES:
                    cols[k].extend(getattr(b, k) for b in bars)
            parquet.write(path, cols, DAILY_TYPES, order_by="symbol, t")
            logger.info("daily %s: %d bars", adjustment, len(cols["t"]))
        note(path, missing)

    if not paths.btc_5min.exists():
        raw_bars = await data.crypto_bars(BTC, timeframe="5Min", start=datetime.combine(BTC_START, time(0), UTC),
                                          end=CORE_END)
        cols = {k: [] for k in BAR_TYPES}
        for b in raw_bars:
            cols["t"].append(datetime.fromisoformat(b["t"]))
            for k in ("o", "h", "l", "c", "v"):
                cols[k].append(float(b[k]))
        parquet.write(paths.btc_5min, cols, BAR_TYPES, order_by="t")
        logger.info("BTC 5-min: %d bars", len(cols["t"]))
        note(paths.btc_5min, True)
    else:
        note(paths.btc_5min, False)

    if not paths.quotes_sample.exists():
        cols, truncated = await fetch_quote_sample(data, calendar)
        parquet.write(paths.quotes_sample, cols, QUOTE_TYPES, order_by="symbol, session, window_name, t")
        report["quote_windows_truncated"] = truncated
        note(paths.quotes_sample, True)
    else:
        note(paths.quotes_sample, False)
    return report


def run(paths: CorePaths) -> dict[str, Any]:
    async def main() -> dict[str, Any]:
        data = CoreData.from_settings()
        try:
            return await fetch_core(paths, data=data)
        finally:
            await data.aclose()

    return asyncio.run(main())


def missing(paths: CorePaths, files: Sequence[Path] | None = None) -> list[Path]:
    return [p for p in (files if files is not None else paths.all_files()) if not p.exists()]
