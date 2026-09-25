"""Async Alpaca market-data client: news and bars, within the free plan.

Free-plan rules this client enforces so callers can't trip them:

* **200 requests/minute.** A token bucket paces every request at 3/s
  (180/min) and a 429 backs off and retries.
* **No SIP data from the last 15 minutes.** ``bars`` clamps ``end`` to
  16 minutes ago. Features only ever use completed sessions anyway.

Everything returned is plain data (dicts / ``Bar``), so training and
serving share it without an SDK.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx

from config import get_broker_settings

logger = logging.getLogger("swarm.alpaca_data")

DATA_BASE_URL: Final[str] = "https://data.alpaca.markets"
SIP_DELAY: Final[timedelta] = timedelta(minutes=16)
_REQUESTS_PER_SECOND: Final[float] = 3.0
_MAX_RETRIES: Final[int] = 5


@dataclass(frozen=True, slots=True)
class Bar:
    t: datetime  # bar start, UTC
    o: float
    h: float
    l: float
    c: float
    v: float

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Bar:
        return cls(t=datetime.fromisoformat(raw["t"]), o=float(raw["o"]), h=float(raw["h"]),
                   l=float(raw["l"]), c=float(raw["c"]), v=float(raw["v"]))

    def to_json(self) -> dict[str, Any]:
        return {"t": self.t.isoformat(), "o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v}


class _Pacer:
    """Spaces requests at least ``1/rate`` seconds apart, across tasks."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


class AlpacaData:
    def __init__(self, key_id: str, secret_key: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 rate: float = _REQUESTS_PER_SECOND) -> None:
        self._client = httpx.AsyncClient(
            base_url=DATA_BASE_URL,
            headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key},
            timeout=httpx.Timeout(30.0, connect=5.0),
            transport=transport,
        )
        self._pacer = _Pacer(rate)

    @classmethod
    def from_settings(cls) -> AlpacaData:
        settings = get_broker_settings()
        return cls(settings.api_key_id, settings.api_secret_key)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, _MAX_RETRIES + 1):
            await self._pacer.wait()
            try:
                response = await self._client.get(path, params=params)
            except httpx.TransportError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning("GET %s failed (%s); retry %d", path, exc, attempt)
                await asyncio.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == _MAX_RETRIES:
                    response.raise_for_status()
                await asyncio.sleep(min(60, 2 ** attempt))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    async def news(
        self, symbols: Sequence[str], *, start: datetime, end: datetime | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Every article touching ``symbols`` in ``[start, end]``, oldest first."""
        params: dict[str, Any] = {
            "symbols": ",".join(symbols), "start": _rfc3339(start), "limit": 50,
            "sort": "asc", "include_content": "false", "exclude_contentless": "false",
        }
        if end is not None:
            params["end"] = _rfc3339(end)
        while True:
            page = await self._get("/v1beta1/news", params)
            for item in page.get("news", []):
                yield item
            token = page.get("next_page_token")
            if not token:
                return
            params["page_token"] = token

    async def bars(
        self, symbols: Sequence[str], *, timeframe: str, start: datetime, end: datetime | None = None,
        feed: str = "sip",
    ) -> dict[str, list[Bar]]:
        """Split- and dividend-adjusted bars per symbol, oldest first."""
        latest_allowed = datetime.now(UTC) - SIP_DELAY
        end = min(end or latest_allowed, latest_allowed)
        params: dict[str, Any] = {
            "symbols": ",".join(symbols), "timeframe": timeframe, "start": _rfc3339(start),
            "end": _rfc3339(end), "feed": feed, "adjustment": "all", "limit": 10_000, "sort": "asc",
        }
        out: dict[str, list[Bar]] = {s: [] for s in symbols}
        while True:
            page = await self._get("/v2/stocks/bars", params)
            for symbol, raw_bars in (page.get("bars") or {}).items():
                out.setdefault(symbol, []).extend(Bar.from_api(b) for b in raw_bars)
            token = page.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token


def _rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
