"""The new fetches, against a scripted Alpaca behind ``httpx.MockTransport``."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from backtest import fetch, parquet
from backtest.fetch import Paths, ResearchData
from backtest.lockbox import DEV_EVENTS_END
from swarm.alpaca_data import AlpacaData

SYMBOLS = ("AAPL", "TSLA")


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class FakeAlpaca:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.news = [
            {"id": 1, "headline": "old article, revised", "symbols": ["AAPL"], "source": "benzinga",
             "author": "a", "created_at": "2021-04-08T14:02:55Z", "updated_at": "2024-03-01T10:00:00Z"},
            {"id": 2, "headline": "TSLA beats", "symbols": ["TSLA", "F"], "source": "benzinga", "author": "b",
             "created_at": "2025-02-03T15:00:00Z", "updated_at": "2025-02-03T15:00:00Z"},
            {"id": 3, "headline": "lock-box side", "symbols": ["AAPL"], "source": "benzinga", "author": "c",
             "created_at": _iso(DEV_EVENTS_END), "updated_at": _iso(DEV_EVENTS_END)},
        ]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, params = request.url.path, request.url.params
        if path == "/v2/calendar":
            return httpx.Response(200, json=[
                {"date": "2025-01-02", "open": "09:30", "close": "16:00", "session_open": "0400",
                 "session_close": "2000", "settlement_date": "2025-01-03"},
                {"date": "2025-07-03", "open": "09:30", "close": "13:00", "session_open": "0400",
                 "session_close": "1700", "settlement_date": "2025-07-07"},
            ] + [{"date": (date(2025, 1, 6) + timedelta(days=i)).isoformat(), "open": "09:30", "close": "16:00"}
                 for i in range(12)])
        if path == "/v1beta1/news":  # one article per page, to exercise paging
            page = int(params.get("page_token", "0"))
            body: dict[str, Any] = {"news": [self.news[page]]}
            if page + 1 < len(self.news):
                body["next_page_token"] = str(page + 1)
            return httpx.Response(200, json=body)
        if path == "/v2/stocks/bars":
            symbol = params["symbols"]
            t0 = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
            bars = [{"t": _iso(t0 + timedelta(minutes=i)), "o": 100 + i, "h": 101 + i, "l": 99 + i,
                     "c": 100.5 + i, "v": 1000, "n": 10, "vw": 100.2} for i in range(3)]
            return httpx.Response(200, json={"bars": {symbol: bars}, "next_page_token": None})
        if path == "/v2/stocks/quotes":
            page = int(params.get("page_token", "0"))
            quote = {"t": "2025-01-02T14:30:00.123456789Z", "bp": 187.1, "ap": 187.12, "bs": 2, "as": 3,
                     "bx": "V", "ax": "Q", "c": ["R"], "z": "C"}
            body = {"quotes": {params["symbols"]: [quote]}}
            if page < 2:
                body["next_page_token"] = str(page + 1)
            return httpx.Response(200, json=body)
        return httpx.Response(404, text=f"unhandled {path}")


@pytest.fixture
def fake() -> FakeAlpaca:
    return FakeAlpaca()


def _data(fake: FakeAlpaca) -> ResearchData:
    return ResearchData("key", "secret", transport=httpx.MockTransport(fake.handler), rate=1000.0)


def test_production_bars_stay_adjusted_and_raw_is_opt_in(fake: FakeAlpaca) -> None:
    async def go() -> None:
        data = AlpacaData("key", "secret", transport=httpx.MockTransport(fake.handler), rate=1000.0)
        start = datetime(2025, 1, 2, tzinfo=UTC)
        await data.bars(["AAPL"], timeframe="1Day", start=start)
        await data.bars(["AAPL"], timeframe="1Min", start=start, adjustment="raw")
        await data.aclose()

    asyncio.run(go())
    assert [r.url.params["adjustment"] for r in fake.requests] == ["all", "raw"]


def test_calendar_is_stored_as_utc_instants_with_half_days(fake: FakeAlpaca) -> None:
    raw = asyncio.run(fetch.fetch_calendar(date(2025, 1, 1), date(2025, 12, 31), key_id="k", secret_key="s",
                                           transport=httpx.MockTransport(fake.handler)))
    cols = fetch.calendar_columns(raw)
    assert fake.requests[0].url.host == "paper-api.alpaca.markets"
    assert cols["open_at"][0] == datetime(2025, 1, 2, 14, 30, tzinfo=UTC)     # EST: UTC-5
    assert cols["close_at"][1] == datetime(2025, 7, 3, 17, 0, tzinfo=UTC)     # EDT half-day: 13:00 New York
    assert cols["half_day"][:2] == [False, True]


def test_news_keeps_updated_at_and_never_stores_the_lockbox(fake: FakeAlpaca) -> None:
    async def go() -> tuple[dict[str, list[Any]], int]:
        data = _data(fake)
        try:
            return await fetch.fetch_news(data, SYMBOLS, date(2024, 1, 2), datetime(2026, 9, 26, tzinfo=UTC))
        finally:
            await data.aclose()

    cols, dropped = asyncio.run(go())
    assert cols["id"] == [1, 2] and dropped == 1
    assert cols["updated_at"][0] == datetime(2024, 3, 1, 10, tzinfo=UTC)
    assert cols["symbols"][1] == "TSLA,F"  # every ticker, so n_symbols stays derivable
    starts = {r.url.params["start"] for r in fake.requests}
    assert starts == {"2024-01-02T00:00:00Z"}


def test_quotes_page_until_the_cap_and_report_truncation(fake: FakeAlpaca) -> None:
    async def go(max_pages: int) -> tuple[list[dict[str, Any]], bool]:
        data = _data(fake)
        try:
            start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
            return await data.quotes("AAPL", start=start, end=start + timedelta(minutes=1), max_pages=max_pages)
        finally:
            await data.aclose()

    quotes, truncated = asyncio.run(go(10))
    assert (len(quotes), truncated) == (3, False)
    quotes, truncated = asyncio.run(go(2))
    assert (len(quotes), truncated) == (2, True)
    # SIP stamps quotes in nanoseconds; Python keeps microseconds
    assert datetime.fromisoformat(quotes[0]["t"]) == datetime(2025, 1, 2, 14, 30, 0, 123456, tzinfo=UTC)


def test_fetch_all_writes_every_file_once_and_resumes(fake: FakeAlpaca, tmp_path: Path) -> None:
    paths = Paths(tmp_path)

    async def go() -> dict[str, Any]:
        data = _data(fake)
        try:
            return await fetch.fetch_all(paths, data=data, now=datetime(2026, 9, 26, tzinfo=UTC),
                                         calendar_transport=httpx.MockTransport(fake.handler),
                                         calendar_keys=("k", "s"), symbols=SYMBOLS)
        finally:
            await data.aclose()

    first = asyncio.run(go())
    assert all(p.exists() for p in paths.all_files(SYMBOLS))
    assert len(first["fetched"]) == len(paths.all_files(SYMBOLS)) and first["news_after_dev_window_not_stored"] == 1
    bars = parquet.read_rows(paths.bars_1min("AAPL"), "ORDER BY t")
    assert [b["o"] for b in bars] == [100.0, 101.0, 102.0]
    assert all(r.url.params["adjustment"] == "raw" for r in fake.requests if r.url.path == "/v2/stocks/bars")
    daily = parquet.read_rows(paths.bars_1day_raw)
    assert {d["symbol"] for d in daily} == set(SYMBOLS)

    n_requests = len(fake.requests)
    second = asyncio.run(go())
    assert len(fake.requests) == n_requests  # nothing fetched twice
    assert second["fetched"] == [] and len(second["skipped"]) == len(paths.all_files(SYMBOLS))
    assert not list(tmp_path.rglob(".*.tmp"))  # atomic writes leave no debris
