"""Redis Streams: publish, and an at-least-once consumer-group loop.

Every message is one field, ``data``, holding JSON. A consumer:

1. reclaims messages another consumer took but never acknowledged (a pod
   that crashed mid-message) once they've been idle ``claim_idle_ms``;
2. reads new messages for its group;
3. acknowledges a message only after its handler returns, so a crash
   means redelivery, never loss;
4. moves a message that keeps failing (``max_deliveries``) to
   ``<stream>.dead`` and acknowledges it, so one poison message can't
   wedge the stage.

Handlers must therefore be idempotent. Ours are: Ingestion de-duplicates on
Alpaca's news id, and the router de-duplicates signals on ``signal_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from swarm.common import STREAM_MAXLEN

logger = logging.getLogger("swarm.bus")

Handler = Callable[[dict[str, Any]], Awaitable[None]]

DEFAULT_CLAIM_IDLE_MS: Final[int] = 60_000
DEFAULT_MAX_DELIVERIES: Final[int] = 5


async def publish(redis: Redis, stream: str, payload: dict[str, Any], *, maxlen: int = STREAM_MAXLEN) -> str:
    message_id = await redis.xadd(stream, {"data": json.dumps(payload, separators=(",", ":"))},
                                  maxlen=maxlen, approximate=True)
    return message_id.decode() if isinstance(message_id, bytes) else str(message_id)


async def ensure_group(redis: Redis, stream: str, group: str) -> None:
    """Create the group at the start of the stream, so a stage that comes
    up after its producer still processes the backlog."""
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


@dataclass
class ConsumerStats:
    processed: int = 0
    failed: int = 0
    dead_lettered: int = 0


class StreamConsumer:
    def __init__(
        self, redis: Redis, stream: str, group: str, consumer: str, handler: Handler, *,
        count: int = 10, block_ms: int = 5_000,
        claim_idle_ms: int = DEFAULT_CLAIM_IDLE_MS, max_deliveries: int = DEFAULT_MAX_DELIVERIES,
    ) -> None:
        self.redis, self.stream, self.group, self.consumer = redis, stream, group, consumer
        self.handler = handler
        self.count, self.block_ms = count, block_ms
        self.claim_idle_ms, self.max_deliveries = claim_idle_ms, max_deliveries
        self.stats = ConsumerStats()

    async def _deliveries(self, message_id: str) -> int:
        pending = await self.redis.xpending_range(self.stream, self.group, min=message_id, max=message_id, count=1)
        return int(pending[0]["times_delivered"]) if pending else 1

    async def _handle(self, message_id: Any, fields: dict[Any, Any], *, reclaimed: bool) -> None:
        mid = message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        raw = fields.get(b"data", fields.get("data"))
        if reclaimed and await self._deliveries(mid) > self.max_deliveries:
            await self.redis.xadd(f"{self.stream}.dead", {"data": raw or "", "source_id": mid},
                                  maxlen=STREAM_MAXLEN, approximate=True)
            await self.redis.xack(self.stream, self.group, mid)
            self.stats.dead_lettered += 1
            logger.error("%s %s: dead-lettered after %d deliveries", self.stream, mid, self.max_deliveries)
            return
        try:
            payload = json.loads(raw)
            await self.handler(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stats.failed += 1
            logger.exception("%s %s: handler failed; will retry", self.stream, mid)
            return
        await self.redis.xack(self.stream, self.group, mid)
        self.stats.processed += 1

    async def run_once(self) -> int:
        """Reclaim stale messages, then read new ones. Returns messages seen."""
        seen = 0
        claimed = await self.redis.xautoclaim(self.stream, self.group, self.consumer,
                                              min_idle_time=self.claim_idle_ms, start_id="0-0", count=self.count)
        for message_id, fields in claimed[1]:
            if fields:  # entries trimmed from the stream come back empty
                await self._handle(message_id, fields, reclaimed=True)
                seen += 1
        response = await self.redis.xreadgroup(self.group, self.consumer, {self.stream: ">"},
                                               count=self.count, block=self.block_ms)
        for _stream, messages in response or []:
            for message_id, fields in messages:
                await self._handle(message_id, fields, reclaimed=False)
                seen += 1
        return seen

    async def run_forever(self) -> None:
        await ensure_group(self.redis, self.stream, self.group)
        logger.info("Consuming %s as %s/%s", self.stream, self.group, self.consumer)
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s consumer loop error; retrying in 5s", self.stream)
                await asyncio.sleep(5)
