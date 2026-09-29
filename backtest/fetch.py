"""The new free fetches (design §2, D9), cached as Parquet under ``.cache/backtest/``.

* **Raw 1-minute SIP bars**: quotes, fills, stops, intraday outcomes and v2
  labels are priced in the dollars that actually traded.
* **Raw daily bars**: with the cached adjusted ones, the adjustment factor
  (adjusted close / raw close) per session.
* **The exchange calendar** (``/v2/calendar``): sessions, holidays, 13:00 half-days.
* **News with ``updated_at``**: re-fetched to attach revision times to the
  cached articles (the revision diagnostic, §9). Only development-window
  articles are kept; anything created later is lock-box and is not stored.
* **An NBBO quote sample** (D9): a few one-minute windows on a few sessions,
  to measure the half-spread the cost levels assume.

Free plan, paced by ``swarm.alpaca_data.AlpacaData`` (3 requests/s). Every
file is written atomically and skipped when present, so an interrupted
run resumes where it stopped. The calendar is the one read from the
trading API, and it goes to ``config.PAPER_BASE_URL``, the only trading
host this project may reach.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import httpx

from backtest import parquet
from backtest.lockbox import DEV_EVENTS_END
from config import PAPER_BASE_URL, get_broker_settings
from swarm.alpaca_data import AlpacaData, _rfc3339
from swarm.common import DEFAULT_WATCHLIST, NEW_YORK

logger = logging.getLogger("backtest.fetch")

SYMBOLS: Final[tuple[str, ...]] = DEFAULT_WATCHLIST
BARS_START: Final[date] = date(2024, 1, 2)
DAILY_START: Final[date] = date(2023, 9, 4)          # as the adjusted daily cache
NEWS_START: Final[date] = date(2024, 1, 2)            # as the news cache
#: Through the 2026-09-18 embargo session, after-hours included. Lock-box
#: sessions (from 09-21) are fetched when the lock-box opens, not before.
BARS_END: Final[datetime] = datetime(2026, 9, 19, 0, 0, tzinfo=UTC)
CALENDAR_START: Final[date] = date(2023, 9, 1)
CALENDAR_END: Final[date] = date(2026, 12, 31)

QUOTE_SESSIONS: Final[int] = 10
#: (name, minutes after the open) of each one-minute quote window; "midday" is 12:00 New York.
QUOTE_WINDOWS: Final[tuple[tuple[str, int | None], ...]] = (("open", 0), ("third_minute", 2), ("midday", None))
QUOTE_MAX_PAGES: Final[int] = 20

BAR_TYPES: Final[dict[str, str]] = {"t": "TIMESTAMPTZ", "o": "DOUBLE", "h": "DOUBLE", "l": "DOUBLE",
                                    "c": "DOUBLE", "v": "DOUBLE"}


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def calendar(self) -> Path:
        return self.root / "calendar.parquet"

    @property
    def news(self) -> Path:
        return self.root / "news.parquet"

    def bars_1min(self, symbol: str) -> Path:
        return self.root / "bars_1min_raw" / f"{symbol}.parquet"

    @property
    def bars_1day_raw(self) -> Path:
        return self.root / "bars_1day_raw.parquet"

    @property
    def quotes_sample(self) -> Path:
        return self.root / "quotes_sample.parquet"

    def all_files(self, symbols: Sequence[str] = SYMBOLS) -> list[Path]:
        return [self.calendar, self.news, *(self.bars_1min(s) for s in symbols), self.bars_1day_raw,
                self.quotes_sample]


class ResearchData(AlpacaData):
    """``AlpacaData`` plus the one endpoint only research needs: historical quotes."""

    async def quotes(self, symbol: str, *, start: datetime, end: datetime, max_pages: int,
                     feed: str = "sip") -> tuple[list[dict[str, Any]], bool]:
        """``(quotes, truncated)``: at most ``max_pages`` pages of 10,000."""
        params: dict[str, Any] = {"symbols": symbol, "start": _rfc3339(start), "end": _rfc3339(end),
                                  "feed": feed, "limit": 10_000, "sort": "asc"}
        out: list[dict[str, Any]] = []
        for _ in range(max_pages):
            page = await self._get("/v2/stocks/quotes", params)
            out.extend((page.get("quotes") or {}).get(symbol, []))
            token = page.get("next_page_token")
            if not token:
                return out, False
            params["page_token"] = token
        return out, True


# --- the calendar -------------------------------------------------------------------

async def fetch_calendar(start: date, end: date, *, transport: httpx.AsyncBaseTransport | None = None,
                         key_id: str | None = None, secret_key: str | None = None) -> list[dict[str, Any]]:
    if key_id is None or secret_key is None:
        settings = get_broker_settings()
        key_id, secret_key = settings.api_key_id, settings.api_secret_key
    async with httpx.AsyncClient(base_url=PAPER_BASE_URL, transport=transport, timeout=30.0,
                                 headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}) as client:
        if client.base_url.host != httpx.URL(PAPER_BASE_URL).host:
            raise RuntimeError(f"refusing to call {client.base_url}: only the paper host is allowed")
        response = await client.get("/v2/calendar", params={"start": start.isoformat(), "end": end.isoformat()})
        response.raise_for_status()
        return response.json()


def _local(day: date, hhmm: str) -> datetime:
    hhmm = hhmm.replace(":", "")
    return datetime.combine(day, time(int(hhmm[:2]), int(hhmm[2:])), NEW_YORK).astimezone(UTC)


def calendar_columns(raw: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Alpaca gives New York wall-clock times; the cache holds UTC instants."""
    cols: dict[str, list[Any]] = {k: [] for k in ("date", "open_at", "close_at", "session_open_at",
                                                  "session_close_at", "half_day")}
    for row in raw:
        day = date.fromisoformat(row["date"])
        close_at = _local(day, row["close"])
        cols["date"].append(day)
        cols["open_at"].append(_local(day, row["open"]))
        cols["close_at"].append(close_at)
        cols["session_open_at"].append(_local(day, row.get("session_open", "0400")))
        cols["session_close_at"].append(_local(day, row.get("session_close", "2000")))
        cols["half_day"].append(close_at.astimezone(NEW_YORK).time() < time(16, 0))
    return cols


CALENDAR_TYPES: Final[dict[str, str]] = {"date": "DATE", "open_at": "TIMESTAMPTZ", "close_at": "TIMESTAMPTZ",
                                         "session_open_at": "TIMESTAMPTZ", "session_close_at": "TIMESTAMPTZ",
                                         "half_day": "BOOLEAN"}

# --- news ------------------------------------------------------------------------------

NEWS_TYPES: Final[dict[str, str]] = {"id": "BIGINT", "created_at": "TIMESTAMPTZ", "updated_at": "TIMESTAMPTZ",
                                     "headline": "VARCHAR", "symbols": "VARCHAR", "source": "VARCHAR",
                                     "author": "VARCHAR"}


async def fetch_news(data: AlpacaData, symbols: Sequence[str], start: date,
                     end: datetime) -> tuple[dict[str, list[Any]], int]:
    """Development-window articles as columns, and how many later ones were dropped."""
    cols: dict[str, list[Any]] = {k: [] for k in NEWS_TYPES}
    dropped = 0
    async for item in _news(data, symbols, start, end):
        created = datetime.fromisoformat(item["created_at"])
        if created >= DEV_EVENTS_END:
            dropped += 1  # lock-box side: never stored
            continue
        cols["id"].append(int(item["id"]))
        cols["created_at"].append(created)
        cols["updated_at"].append(datetime.fromisoformat(item["updated_at"]) if item.get("updated_at") else None)
        cols["headline"].append(item.get("headline") or "")
        cols["symbols"].append(",".join(item.get("symbols") or []))
        cols["source"].append(item.get("source") or "")
        cols["author"].append(item.get("author") or "")
        if len(cols["id"]) % 10_000 == 0:
            logger.info("news: %d articles", len(cols["id"]))
    return cols, dropped


async def _news(data: AlpacaData, symbols: Sequence[str], start: date, end: datetime) -> AsyncIterator[dict]:
    async for item in data.news(symbols, start=datetime.combine(start, time(0), UTC), end=end):
        yield item


# --- bars -------------------------------------------------------------------------------

async def fetch_bars(data: AlpacaData, symbol: str, timeframe: str, start: date, end: datetime,
                     adjustment: str) -> dict[str, list[Any]]:
    got = await data.bars([symbol], timeframe=timeframe, start=datetime.combine(start, time(0), UTC), end=end,
                          adjustment=adjustment)
    bars = got.get(symbol, [])
    return {"t": [b.t for b in bars], "o": [b.o for b in bars], "h": [b.h for b in bars],
            "l": [b.l for b in bars], "c": [b.c for b in bars], "v": [b.v for b in bars]}


# --- the quote sample -----------------------------------------------------------------------

QUOTE_TYPES: Final[dict[str, str]] = {
    "symbol": "VARCHAR", "session": "DATE", "window_name": "VARCHAR", "t": "TIMESTAMPTZ", "bid_price": "DOUBLE",
    "ask_price": "DOUBLE", "bid_size": "DOUBLE", "ask_size": "DOUBLE", "bid_exchange": "VARCHAR",
    "ask_exchange": "VARCHAR",
}


def quote_sessions(calendar: dict[str, list[Any]], n: int = QUOTE_SESSIONS) -> list[int]:
    """``n`` evenly spaced full sessions of the development window (row indices)."""
    full = [i for i, (d, half) in enumerate(zip(calendar["date"], calendar["half_day"], strict=True))
            if BARS_START <= d < DEV_EVENTS_END.date() and not half]
    step = len(full) / n
    return [full[int(step * k + step / 2)] for k in range(n)]


def quote_window(open_at: datetime, day: date, offset: int | None) -> tuple[datetime, datetime]:
    start = open_at + timedelta(minutes=offset) if offset is not None else _local(day, "1200")
    return start, start + timedelta(minutes=1)


async def fetch_quote_sample(data: ResearchData, symbols: Sequence[str],
                             calendar: dict[str, list[Any]]) -> tuple[dict[str, list[Any]], list[str]]:
    cols: dict[str, list[Any]] = {k: [] for k in QUOTE_TYPES}
    truncated: list[str] = []
    for i in quote_sessions(calendar):
        day, open_at = calendar["date"][i], calendar["open_at"][i]
        for symbol in symbols:
            for name, offset in QUOTE_WINDOWS:
                start, end = quote_window(open_at, day, offset)
                quotes, cut = await data.quotes(symbol, start=start, end=end, max_pages=QUOTE_MAX_PAGES)
                if cut:
                    truncated.append(f"{symbol} {day} {name}")
                for q in quotes:
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
        logger.info("quotes: %s done (%d quotes so far)", day, len(cols["t"]))
    return cols, truncated


# --- the whole ingest --------------------------------------------------------------------------

async def fetch_all(paths: Paths, *, data: ResearchData, now: datetime,
                    calendar_transport: httpx.AsyncBaseTransport | None = None,
                    calendar_keys: tuple[str, str] | None = None,
                    symbols: Sequence[str] = SYMBOLS) -> dict[str, Any]:
    """Fetch every missing file; return what was fetched and what was skipped."""
    report: dict[str, Any] = {"fetched": [], "skipped": []}

    def note(path: Path, fetched: bool) -> None:
        report["fetched" if fetched else "skipped"].append(str(path.relative_to(paths.root)))

    if not paths.calendar.exists():
        keys = calendar_keys or (None, None)
        raw = await fetch_calendar(CALENDAR_START, CALENDAR_END, transport=calendar_transport,
                                   key_id=keys[0], secret_key=keys[1])
        parquet.write(paths.calendar, calendar_columns(raw), CALENDAR_TYPES, order_by="date")
        note(paths.calendar, True)
    else:
        note(paths.calendar, False)
    calendar = {k: [r[k] for r in parquet.read_rows(paths.calendar, "ORDER BY date")] for k in CALENDAR_TYPES}

    if not paths.news.exists():
        cols, dropped = await fetch_news(data, symbols, NEWS_START, now)
        parquet.write(paths.news, cols, NEWS_TYPES, order_by="created_at, id")
        report["news_after_dev_window_not_stored"] = dropped
        note(paths.news, True)
    else:
        note(paths.news, False)

    for symbol in symbols:
        path = paths.bars_1min(symbol)
        missing = not path.exists()
        if missing:
            cols = await fetch_bars(data, symbol, "1Min", BARS_START, BARS_END, "raw")
            parquet.write(path, cols, BAR_TYPES, order_by="t")
            logger.info("1-min raw %s: %d bars", symbol, len(cols["t"]))
        note(path, missing)
    if not paths.bars_1day_raw.exists():
        cols = {k: [] for k in ("symbol", *BAR_TYPES)}
        for symbol in symbols:
            got = await fetch_bars(data, symbol, "1Day", DAILY_START, BARS_END, "raw")
            cols["symbol"].extend([symbol] * len(got["t"]))
            for k in BAR_TYPES:
                cols[k].extend(got[k])
        parquet.write(paths.bars_1day_raw, cols, {"symbol": "VARCHAR", **BAR_TYPES}, order_by="symbol, t")
        note(paths.bars_1day_raw, True)
    else:
        note(paths.bars_1day_raw, False)

    if not paths.quotes_sample.exists():
        cols, truncated = await fetch_quote_sample(data, symbols, calendar)
        parquet.write(paths.quotes_sample, cols, QUOTE_TYPES, order_by="symbol, session, window_name, t")
        report["quote_windows_truncated"] = truncated
        note(paths.quotes_sample, True)
    else:
        note(paths.quotes_sample, False)
    return report


def run(paths: Paths, now: datetime | None = None) -> dict[str, Any]:
    async def main() -> dict[str, Any]:
        data = ResearchData.from_settings()
        try:
            return await fetch_all(paths, data=data, now=now or datetime.now(UTC))
        finally:
            await data.aclose()

    return asyncio.run(main())
