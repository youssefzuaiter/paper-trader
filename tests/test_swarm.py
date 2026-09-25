"""The swarm's upstream agents and the maths they share with training.

Redis is ``fakeredis``; Alpaca's data API is a scripted stub; the Risk
Router is the real app over ``httpx.ASGITransport`` with the same fake
broker ``test_risk_router`` uses. The last test pushes one article through
all four agents and checks what reaches the broker.
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import fakeredis
import httpx
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

import webhook
from swarm import features
from swarm.alpaca_data import Bar
from swarm.bus import StreamConsumer, ensure_group, publish
from swarm.common import NEW_YORK, NEWS_ENRICHED, NEWS_RAW, SIGNALS_EVALUATED
from swarm.features import (
    FEATURE_NAMES,
    completed_sessions,
    daily_features,
    feature_vector,
)
from swarm.inference_agent import InferenceAgent, RouterClient
from swarm.ingestion import NewsIngestor
from swarm.quant import QuantAgent
from swarm.return_model import ArtifactMismatch, ReturnModel
from swarm.train_return_model import DailyHistory, RegularBars

SECRET = "k" * 40


def ny(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm), NEW_YORK)


def daily_series(n: int, *, end: date, seed: int = 1, start_price: float = 100.0) -> list[Bar]:
    """``n`` weekday sessions ending at ``end``, stamped like Alpaca (midnight NY)."""
    rng = random.Random(seed)
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    bars, close = [], start_price
    for d in reversed(days):
        open_ = close * (1 + rng.uniform(-0.01, 0.01))
        close = open_ * (1 + rng.uniform(-0.02, 0.02))
        high, low = max(open_, close) * 1.01, min(open_, close) * 0.99
        bars.append(Bar(t=ny(d, 0), o=open_, h=high, l=low, c=close, v=rng.uniform(1e6, 2e6)))
    return bars


@pytest.fixture
def redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis()


async def read_stream(redis: fakeredis.FakeAsyncRedis, stream: str) -> list[dict[str, Any]]:
    return [json.loads(fields[b"data"]) for _, fields in await redis.xrange(stream)]


# --- features: no look-ahead, and identical in training and serving ---------------

def test_a_session_counts_only_after_its_close_and_the_sip_delay() -> None:
    thursday, friday = date(2026, 9, 24), date(2026, 9, 25)
    bars = [Bar(t=ny(thursday, 0), o=1, h=1, l=1, c=1, v=1), Bar(t=ny(friday, 0), o=1, h=1, l=1, c=1, v=1)]
    assert [b.t for b in completed_sessions(bars, ny(friday, 10))] == [bars[0].t]
    assert len(completed_sessions(bars, ny(friday, 16, 14))) == 1
    assert len(completed_sessions(bars, ny(friday, 16, 15))) == 2


def test_daily_features_known_values() -> None:
    flat = [Bar(t=ny(date(2026, 1, 1) + timedelta(days=i), 0), o=100, h=101, l=99, c=100, v=1e6) for i in range(40)]
    f = daily_features(flat)
    assert f["atr14_pct"] == pytest.approx(2.0)  # true range is always 101 - 99
    assert f["rv20"] == pytest.approx(0.0)
    assert f["mom5"] == f["mom20"] == f["dist_sma20"] == f["gap1"] == pytest.approx(0.0)
    assert f["vol_z20"] == 0.0

    rising = [Bar(t=b.t, o=100 + i, h=101 + i, l=99 + i, c=100 + i, v=1e6) for i, b in enumerate(flat)]
    g = daily_features(rising)
    assert g["rsi14"] == 100.0
    assert g["mom5"] == pytest.approx((139 / 134 - 1) * 100)


def test_features_depend_only_on_the_fixed_window() -> None:
    long = daily_series(200, end=date(2026, 9, 24))
    assert daily_features(long) == daily_features(long[-features.FEATURE_WINDOW:])
    with pytest.raises(ValueError):
        daily_features(long[:features.MIN_DAILY_BARS - 1])


def test_training_history_matches_the_live_quant_path() -> None:
    bars = daily_series(120, end=date(2026, 9, 24), seed=3)
    history = DailyHistory(bars)
    rng = random.Random(9)
    for _ in range(50):
        as_of = ny(date(2026, 6, 1), 9) + timedelta(minutes=rng.randrange(0, 160 * 24 * 60))
        live = completed_sessions(bars, as_of)
        expected = daily_features(live) if len(live) >= features.MIN_DAILY_BARS else None
        assert history.features_at(as_of) == expected


def test_feature_vector_order_and_session_flag() -> None:
    daily = {name: float(i) for i, name in enumerate(features.DAILY_FEATURES)}
    row = feature_vector(sentiment=[0.7, 0.2, 0.1], daily=daily, n_symbols=2, published_at=ny(date(2026, 9, 24), 10))
    assert len(row) == len(FEATURE_NAMES)
    assert row[FEATURE_NAMES.index("sent_net")] == pytest.approx(0.5)
    assert row[FEATURE_NAMES.index("regular_hours")] == 1.0
    after_hours = feature_vector(sentiment=[0.7, 0.2, 0.1], daily=daily, n_symbols=2,
                                 published_at=ny(date(2026, 9, 24), 17))
    assert after_hours[FEATURE_NAMES.index("regular_hours")] == 0.0


# --- training labels ------------------------------------------------------------

def _intraday(days: list[date]) -> list[Bar]:
    bars = []
    for d in days:
        slot = ny(d, 4)  # extended hours from 04:00, as SIP returns them
        while slot < ny(d, 20):
            price = d.toordinal() + slot.hour / 100 + slot.minute / 10_000
            bars.append(Bar(t=slot.astimezone(UTC), o=price, h=price, l=price, c=price + 0.5, v=1))
            slot += timedelta(minutes=30)
    return bars


def test_labels_enter_after_publication_and_exit_at_that_sessions_close() -> None:
    thu, fri, mon = date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)
    regular = RegularBars.build(_intraday([thu, fri, mon]))

    ret, entry, exit_ = regular.forward_return(ny(thu, 10, 7))  # mid-bar: wait for 10:30
    assert entry == ny(thu, 10, 30)
    assert exit_ == ny(thu, 15, 30)  # the session's last 30-minute bar, closing at 16:00
    assert ret == pytest.approx((thu.toordinal() + 0.153 + 0.5) / (thu.toordinal() + 0.103) - 1)

    _, entry, _ = regular.forward_return(ny(thu, 10, 30))  # exactly on a boundary
    assert entry == ny(thu, 10, 30)

    _, entry, exit_ = regular.forward_return(ny(fri, 18))  # after hours Friday → Monday's session
    assert (entry, exit_) == (ny(mon, 9, 30), ny(mon, 15, 30))

    _, entry, _ = regular.forward_return(ny(mon, 6))  # pre-market bars are never an entry
    assert entry == ny(mon, 9, 30)

    assert regular.forward_return(ny(mon, 17)) is None  # no later session in the data


def test_tradeable_matches_the_router_entry_window() -> None:
    from swarm.features import is_tradeable

    thu = date(2026, 9, 24)
    assert is_tradeable(ny(thu, 9, 30)) and is_tradeable(ny(thu, 15, 29))
    assert not is_tradeable(ny(thu, 15, 30)) and not is_tradeable(ny(thu, 9, 29))
    assert not is_tradeable(ny(date(2026, 9, 26), 11))  # Saturday


# --- bus ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumer_acks_retries_and_dead_letters(redis: fakeredis.FakeAsyncRedis) -> None:
    await ensure_group(redis, "s", "g")
    for n in range(3):
        await publish(redis, "s", {"n": n})

    attempts: dict[int, int] = {}

    async def handler(payload: dict[str, Any]) -> None:
        attempts[payload["n"]] = attempts.get(payload["n"], 0) + 1
        if payload["n"] == 1 and attempts[1] < 2:
            raise RuntimeError("transient")
        if payload["n"] == 2:
            raise RuntimeError("poison")

    consumer = StreamConsumer(redis, "s", "g", "c1", handler, block_ms=10, claim_idle_ms=0, max_deliveries=3)
    for _ in range(5):
        await consumer.run_once()

    assert attempts[0] == 1 and attempts[1] == 2  # retried once, then succeeded
    assert attempts[2] == 3  # tried max_deliveries times, then parked
    assert consumer.stats.dead_lettered == 1
    assert (await redis.xpending("s", "g"))["pending"] == 0
    (dead,) = await redis.xrange("s.dead")
    assert json.loads(dead[1][b"data"]) == {"n": 2}


@pytest.mark.asyncio
async def test_a_late_consumer_group_processes_the_backlog(redis: fakeredis.FakeAsyncRedis) -> None:
    await publish(redis, "s", {"n": 0})  # published before the group exists
    seen: list[int] = []

    async def handler(payload: dict[str, Any]) -> None:
        seen.append(payload["n"])

    await ensure_group(redis, "s", "late")
    await StreamConsumer(redis, "s", "late", "c", handler, block_ms=10).run_once()
    assert seen == [0]


# --- ingestion --------------------------------------------------------------------

class FakeNewsData:
    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self.articles = articles
        self.starts: list[datetime] = []

    async def news(self, symbols, *, start, end=None) -> AsyncIterator[dict[str, Any]]:
        self.starts.append(start)
        for article in self.articles:
            yield article


def _article(article_id: int, symbols: list[str], created_at: datetime, headline: str = "TSLA beats") -> dict:
    return {"id": article_id, "headline": headline, "summary": "s", "symbols": symbols,
            "created_at": created_at.isoformat().replace("+00:00", "Z"), "source": "benzinga", "url": "u"}


@pytest.mark.asyncio
async def test_ingestion_filters_and_deduplicates(redis: fakeredis.FakeAsyncRedis) -> None:
    now = datetime.now(UTC)
    ingestor = NewsIngestor(redis, FakeNewsData([]), ("TSLA", "AAPL"))
    assert await ingestor.handle_article(_article(1, ["TSLA", "SPCX"], now)) is True
    assert await ingestor.handle_article(_article(1, ["TSLA", "SPCX"], now)) is False  # same id again
    assert await ingestor.handle_article(_article(2, ["F"], now)) is False  # not on the watchlist
    assert await ingestor.handle_article(_article(3, ["AAPL"], now, headline="  ")) is False

    (item,) = await read_stream(redis, NEWS_RAW)
    assert item["symbols"] == ["TSLA"] and item["n_symbols"] == 2 and item["news_id"] == "1"


@pytest.mark.asyncio
async def test_backfill_resumes_from_the_newest_published_article(redis: fakeredis.FakeAsyncRedis) -> None:
    now = datetime(2026, 9, 25, 15, tzinfo=UTC)
    data = FakeNewsData([_article(10, ["TSLA"], now - timedelta(hours=1))])
    ingestor = NewsIngestor(redis, data, ("TSLA",), backfill_hours=6, now=lambda: now)

    assert await ingestor.backfill() == 1
    assert data.starts[0] == now - timedelta(hours=6)  # cold start: the configured window
    assert await ingestor.backfill() == 0  # already published
    assert data.starts[1] == now - timedelta(hours=1)  # warm: from the newest article seen


@pytest.mark.asyncio
async def test_stream_errors_trigger_a_reconnect(redis: fakeredis.FakeAsyncRedis) -> None:
    ingestor = NewsIngestor(redis, FakeNewsData([]), ("TSLA",))
    frame = json.dumps([{"T": "n", **_article(5, ["TSLA"], datetime.now(UTC))}])
    await ingestor.handle_frame(frame)
    assert ingestor.published == 1
    with pytest.raises(ConnectionError):
        await ingestor.handle_frame(json.dumps([{"T": "error", "code": 406, "msg": "connection limit exceeded"}]))


# --- quant ------------------------------------------------------------------------

class FakeBars:
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars
        self.calls = 0

    async def bars(self, symbols, *, timeframe, start, end=None, feed="sip") -> dict[str, list[Bar]]:
        self.calls += 1
        return {symbols[0]: self._bars}


@pytest.mark.asyncio
async def test_quant_enriches_as_of_publication_and_caches_bars(redis: fakeredis.FakeAsyncRedis) -> None:
    today = date(2026, 9, 25)
    bars = daily_series(60, end=today)
    data = FakeBars(bars)
    agent = QuantAgent(redis, data, now=lambda: ny(today, 11))
    article = {"news_id": "7", "headline": "TSLA beats", "symbols": ["TSLA"], "n_symbols": 1,
               "created_at": ny(today, 10, 5).isoformat()}

    await agent.handle(article)
    await agent.handle(article | {"news_id": "8"})

    first, _second = await read_stream(redis, NEWS_ENRICHED)
    assert data.calls == 1  # second article served from the Redis cache
    assert first["features"] == daily_features(bars[:-1])  # today's session is still open
    assert first["features_as_of"] == bars[-2].t.isoformat()
    assert first["atr_usd"] == pytest.approx(first["features"]["atr14_pct"] / 100 * bars[-2].c, abs=1e-4)
    assert first["regular_hours"] is True


@pytest.mark.asyncio
async def test_quant_skips_symbols_without_enough_history(redis: fakeredis.FakeAsyncRedis) -> None:
    today = date(2026, 9, 25)
    agent = QuantAgent(redis, FakeBars(daily_series(10, end=today)), now=lambda: ny(today, 11))
    await agent.handle({"news_id": "1", "headline": "h", "symbols": ["NEW"], "n_symbols": 1,
                        "created_at": ny(today, 10).isoformat()})
    assert agent.skipped == 1
    assert await read_stream(redis, NEWS_ENRICHED) == []


# --- return model artifact ----------------------------------------------------------

def _artifact(tmp_path: Path, *, up_rate: float = 0.7, names: tuple[str, ...] = FEATURE_NAMES) -> Path:
    X = np.zeros((10, len(FEATURE_NAMES)))
    y = np.array([1] * int(up_rate * 10) + [0] * (10 - int(up_rate * 10)))
    joblib.dump(DummyClassifier(strategy="prior").fit(X, y), tmp_path / "model.joblib")
    import sklearn
    (tmp_path / "meta.json").write_text(json.dumps({
        "version": "test-v1", "feature_names": list(names), "sklearn_version": sklearn.__version__,
        "move_table": {"edges": [0.5, 0.65], "mean_return_pct": [-0.3, 0.1, 0.8]},
    }))
    return tmp_path


def test_return_model_predicts_probability_and_binned_move(tmp_path: Path) -> None:
    model = ReturnModel.load(_artifact(tmp_path, up_rate=0.7))
    prob_up, move = model.predict([0.0] * len(FEATURE_NAMES))
    assert prob_up == pytest.approx(0.7)
    assert move == 0.8  # 0.7 falls in the top bin (≥ 0.65)


def test_return_model_refuses_a_different_feature_contract(tmp_path: Path) -> None:
    with pytest.raises(ArtifactMismatch):
        ReturnModel.load(_artifact(tmp_path, names=(*FEATURE_NAMES[1:], FEATURE_NAMES[0])))


# --- inference agent ------------------------------------------------------------------

def _enriched(created_at: datetime, *, news_id: str = "42", symbol: str = "TSLA") -> dict[str, Any]:
    return {"news_id": news_id, "symbol": symbol, "headline": f"{symbol} soars on record deliveries",
            "created_at": created_at.isoformat(), "n_symbols": 1, "regular_hours": True,
            "features": {name: 1.0 for name in features.DAILY_FEATURES}, "atr_usd": 15.2}


class RecordingRouter:
    """A router stand-in that checks the signature like the real one."""

    def __init__(self, status: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status = status
        self.body = body or {"decision": "accepted", "code": "submitted"}
        self.signals: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        ts = request.headers[webhook.TIMESTAMP_HEADER]
        assert webhook.verify(request.content, ts, request.headers[webhook.SIGNATURE_HEADER], SECRET)
        self.signals.append(json.loads(request.content))
        return httpx.Response(self.status, json=self.body)


def _agent(redis, tmp_path: Path, router: RecordingRouter, *, up_rate: float, now: datetime) -> InferenceAgent:
    client = RouterClient("http://router/v1/signals", SECRET, transport=httpx.MockTransport(router.handler))
    return InferenceAgent(redis, ReturnModel.load(_artifact(tmp_path, up_rate=up_rate)),
                          lambda _h: [0.9, 0.05, 0.05], client, source="inference/test", now=lambda: now)


@pytest.mark.asyncio
async def test_inference_sends_only_confident_fresh_signals(redis: fakeredis.FakeAsyncRedis, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    router = RecordingRouter()

    await _agent(redis, tmp_path, router, up_rate=0.5, now=now).handle(_enriched(now - timedelta(seconds=5)))
    await _agent(redis, tmp_path, router, up_rate=0.7, now=now).handle(_enriched(now - timedelta(minutes=10)))
    assert router.signals == []  # one below threshold, one stale (e.g. backfilled)

    await _agent(redis, tmp_path, router, up_rate=0.7, now=now).handle(_enriched(now - timedelta(seconds=5)))
    (signal,) = router.signals
    assert signal["signal_id"] == "42-TSLA-test-v1"
    assert signal["prob_up"] == pytest.approx(0.7)
    assert signal["predicted_move_pct"] == 0.8
    assert signal["atr"] == "15.2"

    outcomes = [e["outcome"] for e in await read_stream(redis, SIGNALS_EVALUATED)]
    assert outcomes == ["below_threshold", "stale", "router_accepted:submitted"]


@pytest.mark.asyncio
async def test_inference_records_a_router_block(redis: fakeredis.FakeAsyncRedis, tmp_path: Path) -> None:
    now = datetime.now(UTC)
    router = RecordingRouter(423, {"detail": {"code": "circuit_breaker", "detail": "-2.6%"}})
    await _agent(redis, tmp_path, router, up_rate=0.7, now=now).handle(_enriched(now - timedelta(seconds=5)))
    (record,) = await read_stream(redis, SIGNALS_EVALUATED)
    assert record["outcome"] == "router_blocked:circuit_breaker"


# --- one article, all four agents ------------------------------------------------------

@pytest.mark.asyncio
async def test_one_article_flows_from_news_to_a_single_paper_order(
    redis: fakeredis.FakeAsyncRedis, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_risk_router import FakeAlpaca, _router

    from risk_router.app import create_app

    monkeypatch.setenv("ROUTER_SIGNAL_SECRET", SECRET)
    broker = FakeAlpaca()
    router = _router(broker, tmp_path / "router-state.json")
    router_app = create_app(router, background=False)
    router_app.state.router = router  # ASGITransport does not run the lifespan

    now = datetime.now(UTC)
    ingestor = NewsIngestor(redis, FakeNewsData([]), ("TSLA",))
    quant = QuantAgent(redis, FakeBars(daily_series(60, end=now.astimezone(NEW_YORK).date())), now=lambda: now)
    client = RouterClient("http://router/v1/signals", SECRET, transport=httpx.ASGITransport(app=router_app))
    inference = InferenceAgent(redis, ReturnModel.load(_artifact(tmp_path, up_rate=0.7)),
                               lambda _h: [0.9, 0.05, 0.05], client, source="inference/test")

    for group, stream in (("quant", NEWS_RAW), ("inference", NEWS_ENRICHED)):
        await ensure_group(redis, stream, group)
    article = _article(99, ["TSLA"], now - timedelta(seconds=3), headline="Tesla deliveries beat estimates")
    await ingestor.handle_article(article)
    await ingestor.handle_article(article)  # the stream repeats itself; ingestion must not

    await StreamConsumer(redis, NEWS_RAW, "quant", "q1", quant.handle, block_ms=10).run_once()
    await StreamConsumer(redis, NEWS_ENRICHED, "inference", "i1", inference.handle, block_ms=10).run_once()

    (order,) = broker.buy_posts()
    assert order["symbol"] == "TSLA" and order["type"] == "limit" and order["time_in_force"] == "day"
    (evaluation,) = await read_stream(redis, SIGNALS_EVALUATED)
    assert evaluation["outcome"] == "router_accepted:submitted"
    await client.aclose()
