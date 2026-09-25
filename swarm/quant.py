"""Quantitative Agent: ``news.raw`` → ``news.enriched``.

For each article, and each watchlist symbol it mentions, computes the
volatility and technical state of that symbol *as of the article's
publication* (``features.completed_sessions``) and publishes one enriched
message per symbol. Stateless and horizontally scalable: replicas share
the ``quant`` consumer group, and daily bars are cached in Redis per
symbol, so N replicas don't make N copies of the same request.

    uvicorn swarm.quant:app --port 8081
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from fastapi import FastAPI
from redis.asyncio import Redis

from swarm.alpaca_data import AlpacaData, Bar
from swarm.bus import StreamConsumer, publish
from swarm.common import NEWS_ENRICHED, NEWS_RAW, redis_url
from swarm.features import (
    MIN_DAILY_BARS,
    completed_sessions,
    daily_features,
    is_regular_hours,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
logger = logging.getLogger("swarm.quant")

#: ~70 calendar days covers MIN_DAILY_BARS sessions across holidays.
_HISTORY: Final[timedelta] = timedelta(days=75)
_CACHE_TTL_SECONDS: Final[int] = 30 * 60


class QuantAgent:
    def __init__(self, redis: Redis, data: AlpacaData, *,
                 now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.redis = redis
        self.data = data
        self._now = now
        self.enriched = 0
        self.skipped = 0

    async def daily_bars(self, symbol: str) -> list[Bar]:
        key = f"bars:1d:{symbol}"
        cached = await self.redis.get(key)
        if cached:
            return [Bar.from_api(b) for b in json.loads(cached)]
        now = self._now()
        bars = (await self.data.bars([symbol], timeframe="1Day", start=now - _HISTORY, end=now)).get(symbol, [])
        # Cache only sessions already final at fetch time: today's bar,
        # fetched mid-session, must not be served after the close as if final.
        bars = completed_sessions(bars, now)
        await self.redis.set(key, json.dumps([b.to_json() for b in bars]), ex=_CACHE_TTL_SECONDS)
        return bars

    async def handle(self, article: dict[str, Any]) -> None:
        published_at = datetime.fromisoformat(article["created_at"])
        for symbol in article["symbols"]:
            history = completed_sessions(await self.daily_bars(symbol), published_at)
            if len(history) < MIN_DAILY_BARS:
                self.skipped += 1
                logger.warning("%s: only %d completed sessions before %s; skipped",
                               symbol, len(history), article["created_at"])
                continue
            features = daily_features(history)
            await publish(self.redis, NEWS_ENRICHED, {
                "news_id": article["news_id"],
                "symbol": symbol,
                "headline": article["headline"],
                "created_at": article["created_at"],
                "n_symbols": article["n_symbols"],
                "regular_hours": is_regular_hours(published_at),
                "features_as_of": history[-1].t.isoformat(),
                "features": features,
                # For the router's volatility-parity sizing (dollars, not a
                # model input). It can only ever shrink an order.
                "atr_usd": round(features["atr14_pct"] / 100 * history[-1].c, 4),
            })
            self.enriched += 1


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis = Redis.from_url(redis_url())
    data = AlpacaData.from_settings()
    agent = QuantAgent(redis, data)
    consumer = StreamConsumer(redis, NEWS_RAW, "quant", os.getenv("CONSUMER_NAME", socket.gethostname()), agent.handle)
    app.state.agent, app.state.consumer = agent, consumer
    task = asyncio.create_task(consumer.run_forever())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await data.aclose()
    await redis.aclose()


app = FastAPI(title="Quantitative Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "enriched": app.state.agent.enriched, "skipped": app.state.agent.skipped,
            "consumer": vars(app.state.consumer.stats)}
