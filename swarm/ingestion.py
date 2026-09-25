"""Ingestion Agent: Alpaca's live news stream → ``news.raw``.

One connection to ``wss://stream.data.alpaca.markets/v1beta1/news``,
subscribed to the watchlist. Every (re)connect first subscribes, then
backfills over REST from the newest article it has already published (or
``BACKFILL_HOURS`` ago on a cold start). Subscribing first means there is
no gap between the backfill and the live feed; the overlap is harmless
because publication is de-duplicated on Alpaca's article id with
``SET NX`` in Redis.

Runs as exactly one pod (Deployment, ``Recreate``). A second connection
would be refused by the free data plan and would publish everything twice
anyway.

    uvicorn swarm.ingestion:app --port 8081
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import certifi
import websockets
from fastapi import FastAPI
from redis.asyncio import Redis

from config import get_broker_settings
from swarm.alpaca_data import AlpacaData
from swarm.bus import publish
from swarm.common import NEWS_RAW, redis_url, watchlist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
logger = logging.getLogger("swarm.ingestion")

NEWS_STREAM_URL: Final[str] = "wss://stream.data.alpaca.markets/v1beta1/news"
SEEN_TTL_SECONDS: Final[int] = 7 * 24 * 3600
LAST_SEEN_KEY: Final[str] = "ingestion:last_created_at"
_MAX_TEXT: Final[int] = 1_000


class NewsIngestor:
    def __init__(self, redis: Redis, data: AlpacaData, symbols: tuple[str, ...], *,
                 backfill_hours: float = 6.0, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.redis = redis
        self.data = data
        self.symbols = symbols
        self._watch = frozenset(symbols)
        self.backfill_hours = backfill_hours
        self._now = now
        self.connected = False
        self.published = 0
        self.last_published_at: str | None = None

    async def handle_article(self, article: dict[str, Any]) -> bool:
        """Publish one article if it touches the watchlist and is new."""
        all_symbols = [str(s).upper() for s in article.get("symbols") or []]
        relevant = [s for s in all_symbols if s in self._watch]
        headline = (article.get("headline") or "").strip()
        if not relevant or not headline:
            return False
        if not await self.redis.set(f"news:seen:{article['id']}", "1", nx=True, ex=SEEN_TTL_SECONDS):
            return False
        created_at = article["created_at"]
        await publish(self.redis, NEWS_RAW, {
            "news_id": str(article["id"]),
            "headline": headline[:_MAX_TEXT],
            "summary": (article.get("summary") or "")[:_MAX_TEXT],
            "symbols": relevant,
            "n_symbols": len(all_symbols),
            "created_at": created_at,
            "source": article.get("source") or "",
            "url": article.get("url") or "",
        })
        await self._remember(created_at)
        self.published += 1
        self.last_published_at = created_at
        return True

    async def _remember(self, created_at: str) -> None:
        current = await self.redis.get(LAST_SEEN_KEY)
        current_s = current.decode() if isinstance(current, bytes) else current
        if current_s is None or datetime.fromisoformat(created_at) > datetime.fromisoformat(current_s):
            await self.redis.set(LAST_SEEN_KEY, created_at)

    async def backfill(self) -> int:
        last = await self.redis.get(LAST_SEEN_KEY)
        floor = self._now() - timedelta(hours=self.backfill_hours)
        start = floor
        if last:
            start = max(floor, datetime.fromisoformat(last.decode() if isinstance(last, bytes) else last))
        count = 0
        async for article in self.data.news(self.symbols, start=start):
            count += await self.handle_article(article)
        logger.info("Backfill from %s published %d new article(s)", start.isoformat(), count)
        return count

    async def handle_frame(self, frame: str | bytes) -> None:
        for message in json.loads(frame):
            kind = message.get("T")
            if kind == "n":
                await self.handle_article(message)
            elif kind == "error":
                # e.g. 406 "connection limit exceeded" — surface it and let
                # the reconnect loop back off.
                raise ConnectionError(f"news stream error {message.get('code')}: {message.get('msg')}")

    async def stream_forever(self, connect: Callable[..., Any] | None = None) -> None:
        settings = get_broker_settings()
        headers = {"APCA-API-KEY-ID": settings.api_key_id, "APCA-API-SECRET-KEY": settings.api_secret_key}
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connect = connect or websockets.connect
        delay = 1
        while True:
            try:
                async with connect(NEWS_STREAM_URL, additional_headers=headers, ssl=ssl_context,
                                   ping_interval=20, ping_timeout=60) as ws:
                    for _ in range(2):  # "connected", then "authenticated"
                        reply = json.loads(await asyncio.wait_for(ws.recv(), 15))
                        if any(m.get("T") == "error" for m in reply):
                            raise ConnectionError(f"news stream refused: {reply}")
                        if any(m.get("msg") == "authenticated" for m in reply):
                            break
                    await ws.send(json.dumps({"action": "subscribe", "news": list(self.symbols)}))
                    self.connected = True
                    logger.info("News stream subscribed to %s", ",".join(self.symbols))
                    await self.backfill()
                    delay = 1
                    async for frame in ws:
                        await self.handle_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("News stream dropped (%s: %s); reconnecting in %ss", type(exc).__name__, exc, delay)
            finally:
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis = Redis.from_url(redis_url())
    data = AlpacaData.from_settings()
    app.state.ingestor = NewsIngestor(redis, data, watchlist(),
                                      backfill_hours=float(os.getenv("BACKFILL_HOURS", "6")))
    task = asyncio.create_task(app.state.ingestor.stream_forever())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await data.aclose()
    await redis.aclose()


app = FastAPI(title="Ingestion Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    ingestor: NewsIngestor = app.state.ingestor
    return {"status": "ok", "connected": ingestor.connected, "published": ingestor.published,
            "last_published_at": ingestor.last_published_at, "watchlist": list(ingestor.symbols)}
