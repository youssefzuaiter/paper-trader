"""Inference Agent: ``news.enriched`` → signed ``TradeSignal`` → Risk Router.

For every enriched article-symbol it runs, side by side:

* **FinBERT** (int8 ONNX, via ``finbert_onnx``) on the headline — the
  PyTorch-lineage text classifier; and
* the **scikit-learn return model** (``return_model``), which takes
  FinBERT's probabilities plus the Quantitative Agent's features and
  outputs a calibrated probability that the stock is up more than 0.25%
  by the next session's close.

A signal goes to the router only when ``prob_up`` clears the router's own
threshold (``tier0.ROUTER_MIN_PROB_UP``) with a positive expected move, and
the article is fresh enough that the router will still accept it. Every
evaluation, sent or not, is recorded on ``signals.evaluated`` for
monitoring and later backtesting.

Needs no broker keys. Signs with ``ROUTER_SIGNAL_SECRET`` only.

    uvicorn swarm.inference_agent:app --port 8081
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

import httpx
from fastapi import FastAPI
from redis.asyncio import Redis

import finbert_onnx
import tier0
import webhook
from risk_router.schemas import TradeSignal
from swarm.bus import StreamConsumer, publish
from swarm.common import NEWS_ENRICHED, SIGNALS_EVALUATED, redis_url
from swarm.features import feature_vector
from swarm.return_model import ReturnModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
logger = logging.getLogger("swarm.inference")

#: The router refuses signals older than 120 s; leave room for the hop.
MAX_SEND_AGE: Final[timedelta] = timedelta(seconds=90)
DEFAULT_ROUTER_URL: Final[str] = "http://risk-router-0.risk-router:8080/v1/signals"

Sentiment = Callable[[str], Sequence[float]]


class RouterClient:
    def __init__(self, url: str, secret: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if len(secret) < 32:
            raise ValueError("ROUTER_SIGNAL_SECRET must be at least 32 characters")
        self.url = url
        self._secret = secret
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=3.0), transport=transport)

    async def send(self, signal: TradeSignal) -> tuple[int | None, dict[str, Any]]:
        """``(status, body)``; ``status`` is ``None`` if the router was unreachable."""
        body = signal.model_dump_json().encode()
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            webhook.TIMESTAMP_HEADER: timestamp,
            webhook.SIGNATURE_HEADER: webhook.SIGNATURE_PREFIX + webhook.sign(body, timestamp, self._secret),
        }
        try:
            response = await self._client.post(self.url, content=body, headers=headers)
        except httpx.TransportError as exc:
            return None, {"error": f"{type(exc).__name__}: {exc}"}
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"raw": response.text[:300]}

    async def aclose(self) -> None:
        await self._client.aclose()


class InferenceAgent:
    def __init__(self, redis: Redis, model: ReturnModel, sentiment: Sentiment, router: RouterClient, *,
                 source: str, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.redis = redis
        self.model = model
        self.sentiment = sentiment
        self.router = router
        self.source = source
        self._now = now
        self.counts: dict[str, int] = {}
        self.last: dict[str, Any] | None = None

    async def handle(self, message: dict[str, Any]) -> None:
        published_at = datetime.fromisoformat(message["created_at"])
        probs = await asyncio.to_thread(self.sentiment, message["headline"])
        row = feature_vector(sentiment=probs, daily=message["features"],
                             n_symbols=int(message["n_symbols"]), published_at=published_at)
        prob_up, expected_move = self.model.predict(row)
        age = self._now() - published_at

        outcome, router_reply = "below_threshold", None
        if Decimal(str(prob_up)) >= tier0.ROUTER_MIN_PROB_UP and expected_move > 0:
            if age > MAX_SEND_AGE:
                outcome = "stale"  # e.g. a backfilled article: evaluated, never traded
            else:
                status, router_reply = await self.router.send(self._signal(message, probs, prob_up, expected_move))
                outcome = self._describe(status, router_reply)

        record = {
            "news_id": message["news_id"], "symbol": message["symbol"], "headline": message["headline"],
            "created_at": message["created_at"], "evaluated_at": self._now().isoformat(),
            "age_seconds": round(age.total_seconds(), 1), "model_version": self.model.version,
            "sentiment": [round(float(p), 6) for p in probs], "prob_up": round(prob_up, 6),
            "expected_move_pct": round(expected_move, 4), "outcome": outcome, "router": router_reply,
        }
        await publish(self.redis, SIGNALS_EVALUATED, record)
        self.counts[outcome] = self.counts.get(outcome, 0) + 1
        self.last = record
        log = logger.info if outcome not in {"below_threshold", "stale"} else logger.debug
        log("%s %s prob_up=%.3f move=%+.3f%% → %s", message["symbol"], message["news_id"], prob_up,
            expected_move, outcome)

    def _signal(self, message: dict[str, Any], probs: Sequence[float], prob_up: float,
                expected_move: float) -> TradeSignal:
        atr = message.get("atr_usd")
        return TradeSignal(
            # Deterministic: a redelivered message maps to the same signal,
            # which the router turns into the same client_order_id.
            signal_id=f"{message['news_id']}-{message['symbol']}-{self.model.version}"[:64],
            symbol=message["symbol"],
            source=self.source,
            model_name=f"finbert-onnx-int8+{self.model.version}",
            prob_up=prob_up,
            predicted_move_pct=expected_move,
            confidence=max(float(p) for p in probs),
            headline=message["headline"][:500],
            atr=Decimal(str(atr)) if atr else None,
            created_at=datetime.fromisoformat(message["created_at"]),
        )

    @staticmethod
    def _describe(status: int | None, reply: dict[str, Any]) -> str:
        if status is None:
            return "router_unreachable"
        if status == 200:
            return f"router_{reply.get('decision')}:{reply.get('code')}"
        if status == 423:
            detail = reply.get("detail") or {}
            return f"router_blocked:{detail.get('code') if isinstance(detail, dict) else detail}"
        return f"router_http_{status}"


def _finbert_sentiment() -> Sentiment:
    model = finbert_onnx.load()

    def score(headline: str) -> Sequence[float]:
        return [float(p) for p in finbert_onnx.softmax(finbert_onnx.logits(model, headline))]

    score("warmup headline")  # pay the graph's first-run cost before taking traffic
    return score


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis = Redis.from_url(redis_url())
    model = ReturnModel.load()
    sentiment = await asyncio.to_thread(_finbert_sentiment)
    router = RouterClient(os.getenv("RISK_ROUTER_URL", DEFAULT_ROUTER_URL), os.getenv("ROUTER_SIGNAL_SECRET", ""))
    consumer_name = os.getenv("CONSUMER_NAME", socket.gethostname())
    agent = InferenceAgent(redis, model, sentiment, router, source=f"inference/{consumer_name}")
    consumer = StreamConsumer(redis, NEWS_ENRICHED, "inference", consumer_name, agent.handle)
    app.state.agent, app.state.consumer = agent, consumer
    logger.info("Inference Agent ready: model %s", model.version)
    task = asyncio.create_task(consumer.run_forever())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await router.aclose()
    await redis.aclose()


app = FastAPI(title="Inference Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    agent: InferenceAgent = app.state.agent
    return {"status": "ok", "model_version": agent.model.version, "outcomes": agent.counts,
            "consumer": vars(app.state.consumer.stats), "last": agent.last}
