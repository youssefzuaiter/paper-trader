"""Risk & Routing agent — the gatekeeper's guarantees, end to end.

Alpaca is simulated by ``FakeAlpaca`` behind ``httpx.MockTransport``, so
the real ``AsyncAlpaca`` client (URL building, retries, Decimal parsing)
is exercised without the network. Each test states one guarantee:
circuit breaker on every execution path, no duplicate accumulation (even
under concurrency), exits at stop/target, at-most-once submission, and a
non-blocking event loop while an order is in flight.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import config
import tier0
import webhook
from risk_router.alpaca_async import AsyncAlpaca
from risk_router.app import create_app
from risk_router.gatekeeper import (
    RiskRouter,
    buy_client_order_id,
    buy_order_payload,
    exit_reason,
)
from risk_router.guards import CircuitBreaker, ExecutionGuard, GuardBlocked, Intent
from risk_router.policy import PolicyRejection, PortfolioSnapshot, check_can_open
from risk_router.schemas import TradeSignal
from risk_router.state import StateStore

SECRET = "s" * 40


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class FakeAlpaca:
    """Just enough of Alpaca's paper REST API, with scriptable failures."""

    def __init__(self) -> None:
        self.equity = "100000"
        self.last_equity = "100000"
        self.account_fails = False
        self.is_open = True
        self.closes_at = datetime.now(UTC) + timedelta(hours=3)
        self.positions: list[dict[str, Any]] = []
        self.open_orders: list[dict[str, Any]] = []
        self.recent_orders: list[dict[str, Any]] = []
        self.quote_at = datetime.now(UTC)
        self.by_client_id: dict[str, dict[str, Any]] = {}
        self.order_posts: list[dict[str, Any]] = []  # every POST /v2/orders attempt
        self.submit_script: list[str] = []  # per attempt: "accept" | "timeout_after_accept" | "timeout" | "500"
        self.submit_delay = 0.0
        self.hosts: set[str] = set()

    def _accept(self, payload: dict[str, Any]) -> dict[str, Any]:
        order = {
            "id": f"ord-{len(self.by_client_id) + 1}",
            "client_order_id": payload["client_order_id"],
            "symbol": payload["symbol"],
            "side": payload["side"],
            "type": payload["type"],
            "qty": payload["qty"],
            "filled_qty": "0",
            "limit_price": payload.get("limit_price"),
            "notional": None,
            "status": "accepted",
            "submitted_at": _iso(datetime.now(UTC)),
        }
        self.by_client_id[payload["client_order_id"]] = order
        self.open_orders.append(order)
        self.recent_orders.append(order)
        return order

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.hosts.add(request.url.host)
        path, method = request.url.path, request.method
        if path == "/v2/account":
            if self.account_fails:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, json={"equity": self.equity, "last_equity": self.last_equity})
        if path == "/v2/clock":
            return httpx.Response(200, json={"is_open": self.is_open, "next_open": "2026-09-28T13:30:00Z",
                                             "next_close": _iso(self.closes_at)})
        if path == "/v2/positions":
            return httpx.Response(200, json=self.positions)
        if path == "/v2/orders:by_client_order_id":
            order = self.by_client_id.get(request.url.params["client_order_id"])
            return httpx.Response(200, json=order) if order else httpx.Response(404, json={"message": "not found"})
        if path == "/v2/orders" and method == "GET":
            status = request.url.params["status"]
            return httpx.Response(200, json=self.open_orders if status == "open" else self.recent_orders)
        if path == "/v2/orders" and method == "POST":
            payload = json.loads(request.content)
            self.order_posts.append(payload)
            if self.submit_delay:
                await asyncio.sleep(self.submit_delay)
            action = self.submit_script.pop(0) if self.submit_script else "accept"
            if action == "timeout_after_accept":
                self._accept(payload)
                raise httpx.ReadTimeout("response lost", request=request)
            if action == "timeout":
                raise httpx.ConnectTimeout("no route", request=request)
            if action == "500":
                return httpx.Response(500, text="internal")
            if payload["client_order_id"] in self.by_client_id:
                return httpx.Response(422, json={"message": "client_order_id must be unique"})
            return httpx.Response(200, json=self._accept(payload))
        if path == "/v2/orders" and method == "DELETE":
            canceled = [{"id": o["id"], "status": 200} for o in self.open_orders]
            self.open_orders.clear()
            return httpx.Response(207, json=canceled)
        if path.startswith("/v2/stocks/") and path.endswith("/quotes/latest"):
            symbol = path.split("/")[3]
            return httpx.Response(200, json={
                "symbol": symbol,
                "quote": {"bp": 353.80, "ap": 354.00, "t": _iso(self.quote_at)},
            })
        return httpx.Response(404, text=f"unhandled {method} {path}")

    def buy_posts(self) -> list[dict[str, Any]]:
        return [p for p in self.order_posts if p["side"] == "buy"]


def _position(symbol: str, entry: str, current: str, qty: str = "0.028") -> dict[str, Any]:
    value = Decimal(qty) * Decimal(current)
    return {"symbol": symbol, "side": "long", "qty": qty, "qty_available": qty,
            "avg_entry_price": entry, "current_price": current, "market_value": str(value)}


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> FakeAlpaca:
    return FakeAlpaca()


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "router-state.json"


def _router(fake: FakeAlpaca, state_path: Path, received: list | None = None) -> RiskRouter:
    alpaca = AsyncAlpaca("key", "secret", transport=httpx.MockTransport(fake.handler), sleep=_no_sleep)
    state = StateStore(state_path)
    guard = ExecutionGuard(state, CircuitBreaker(alpaca, state))

    async def on_submitted(plan, signal, order):
        if received is not None:
            received.append((plan, signal, order))

    return RiskRouter(alpaca, guard, on_submitted=on_submitted)


def _signal(symbol: str = "TSLA", signal_id: str = "sig-00000001", **overrides: Any) -> TradeSignal:
    fields: dict[str, Any] = {
        "signal_id": signal_id, "symbol": symbol, "source": "inference-test", "model_name": "gbm-v1",
        "prob_up": 0.71, "predicted_move_pct": 1.8, "confidence": 0.8,
        "headline": f"{symbol} beats estimates", "atr": Decimal("15.20"),
        "created_at": datetime.now(UTC),
    }
    fields.update(overrides)
    return TradeSignal(**fields)


# --- tier0: one sizing implementation for both services ------------------------

def test_plan_buy_reproduces_the_documented_tsla_example() -> None:
    plan = tier0.plan_buy(symbol="TSLA", bid=Decimal("353.80"), ask=Decimal("354.00"), atr=Decimal("15.20"))
    assert plan.limit_price == Decimal("354.89")
    assert plan.stop_price == Decimal("337.14")
    assert plan.quantity == Decimal("0.019736842")  # volatility parity binds
    assert plan.notional_usd == Decimal("7.00")
    assert plan.stop_loss_kind is tier0.StopLossKind.ENGINE_TRACKED

    without_atr = tier0.plan_buy(symbol="TSLA", bid=Decimal("353.80"), ask=Decimal("354.00"), atr=None)
    assert without_atr.quantity == Decimal("0.028177745")  # $10 cap binds
    assert without_atr.notional_usd == Decimal("10.00")


def test_monolith_gate_is_unchanged_by_the_extraction() -> None:
    import execution
    from inference import InferenceSignal, Quote

    quote = Quote(ticker="TSLA", bid=Decimal("353.80"), ask=Decimal("354.00"), as_of=datetime.now(UTC),
                  atr=Decimal("15.20"))
    base = {"ticker": "TSLA", "headline": "h", "confidence": 0.9,
            "prob_positive": 0.9, "prob_negative": 0.05, "prob_neutral": 0.05}
    plan = execution.validate_and_plan(InferenceSignal(predicted_move_pct=14.8382, **base), quote)
    assert plan.quantity == Decimal("0.019736842")
    with pytest.raises(execution.Tier0Rejection) as rejected:
        execution.validate_and_plan(InferenceSignal(predicted_move_pct=9.99, **base), quote)
    assert rejected.value.code is tier0.RejectionCode.BELOW_MIN_GAIN
    assert execution.CIRCUIT_BREAKER_DAILY_PNL_PCT is tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT


def test_whole_share_orders_carry_a_native_stop() -> None:
    plan = tier0.plan_buy(symbol="F", bid=Decimal("4.95"), ask=Decimal("4.98"), atr=None)
    assert plan.quantity == Decimal("2")  # $10 / $5.00 limit
    payload = buy_order_payload(plan)
    assert (payload["qty"], payload["limit_price"]) == ("2", "5")
    assert payload["order_class"] == "oto"
    assert payload["stop_loss"] == {"stop_price": "4.75"}


# --- opening positions -----------------------------------------------------------

@pytest.mark.asyncio
async def test_accepted_signal_submits_an_exact_limit_buy(fake: FakeAlpaca, state_path: Path) -> None:
    received: list = []
    decision = await _router(fake, state_path, received).handle_signal(_signal())

    assert decision.decision == "accepted"
    (payload,) = fake.order_posts
    assert payload == {
        "symbol": "TSLA", "qty": "0.019736842", "side": "buy", "type": "limit",
        "time_in_force": "day", "limit_price": "354.89",
        "client_order_id": buy_client_order_id("sig-00000001"),
    }
    assert fake.hosts == {"paper-api.alpaca.markets", "data.alpaca.markets"}
    assert len(received) == 1  # the receipt hook fired once


@pytest.mark.asyncio
async def test_second_signal_for_a_held_symbol_is_rejected(fake: FakeAlpaca, state_path: Path) -> None:
    router = _router(fake, state_path)
    first = await router.handle_signal(_signal(signal_id="sig-first-01"))
    second = await router.handle_signal(_signal(signal_id="sig-second-2"))

    assert first.decision == "accepted"
    assert (second.decision, second.code) == ("rejected", "duplicate_accumulation")
    assert len(fake.buy_posts()) == 1


@pytest.mark.asyncio
async def test_concurrent_signals_for_one_symbol_place_one_order(fake: FakeAlpaca, state_path: Path) -> None:
    """The race the lock exists for: both signals would otherwise read
    'no TSLA exposure' before either order was acknowledged."""
    fake.submit_delay = 0.05
    router = _router(fake, state_path)
    decisions = await asyncio.gather(*(router.handle_signal(_signal(signal_id=f"sig-race-{i:04d}")) for i in range(5)))

    assert sorted(d.decision for d in decisions) == ["accepted"] + ["rejected"] * 4
    assert len(fake.buy_posts()) == 1


@pytest.mark.asyncio
async def test_event_loop_keeps_running_while_an_order_is_in_flight(fake: FakeAlpaca, state_path: Path) -> None:
    """Measures the longest the event loop went without running a 10 ms
    heartbeat while an order spent 300 ms in flight. Any blocking call on
    the submit path shows up as a gap at least that long."""
    fake.submit_delay = 0.3
    router = _router(fake, state_path)
    beats: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            beats.append(time.monotonic())
            await asyncio.sleep(0.01)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    decision = await router.handle_signal(_signal())
    stop.set()
    await beat

    assert decision.decision == "accepted"
    longest_stall = max(b - a for a, b in itertools.pairwise(beats))
    assert beats[-1] - beats[0] >= 0.25  # the heartbeat ran across the in-flight window
    assert longest_stall < 0.1, f"event loop stalled for {longest_stall:.3f}s"


@pytest.mark.asyncio
async def test_lost_response_is_not_resubmitted(fake: FakeAlpaca, state_path: Path) -> None:
    fake.submit_script = ["timeout_after_accept"]
    decision = await _router(fake, state_path).handle_signal(_signal())

    assert decision.decision == "accepted"
    assert len(fake.order_posts) == 1  # found by client_order_id, never POSTed again
    assert decision.order["client_order_id"] == buy_client_order_id("sig-00000001")


@pytest.mark.asyncio
async def test_transient_failures_are_retried_until_accepted(fake: FakeAlpaca, state_path: Path) -> None:
    fake.submit_script = ["timeout", "500", "accept"]
    decision = await _router(fake, state_path).handle_signal(_signal())

    assert decision.decision == "accepted"
    assert len(fake.order_posts) == 3
    assert len(fake.by_client_id) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"created_at": datetime.now(UTC) - timedelta(minutes=5)}, "stale_signal"),
        ({"prob_up": 0.55}, "below_min_prob"),
        ({"predicted_move_pct": -0.4}, "non_positive_edge"),
    ],
)
async def test_entry_gate(fake: FakeAlpaca, state_path: Path, overrides: dict[str, Any], code: str) -> None:
    decision = await _router(fake, state_path).handle_signal(_signal(**overrides))
    assert (decision.decision, decision.code) == ("rejected", code)
    assert fake.order_posts == []


@pytest.mark.asyncio
async def test_no_buys_while_the_market_is_closed_or_the_quote_is_stale(fake: FakeAlpaca, state_path: Path) -> None:
    router = _router(fake, state_path)
    fake.is_open = False
    assert (await router.handle_signal(_signal())).code == "market_closed"
    fake.is_open = True
    fake.quote_at = datetime.now(UTC) - timedelta(minutes=10)
    assert (await router.handle_signal(_signal())).code == "stale_quote"
    assert fake.order_posts == []


@pytest.mark.asyncio
async def test_no_buys_in_the_last_half_hour_of_the_session(fake: FakeAlpaca, state_path: Path) -> None:
    fake.closes_at = datetime.now(UTC) + timedelta(minutes=29)  # also true of a 13:00 half-day close
    decision = await _router(fake, state_path).handle_signal(_signal())
    assert (decision.decision, decision.code) == ("rejected", "entry_window_closed")
    assert fake.order_posts == []


# --- the circuit breaker and the kill switch -------------------------------------

@pytest.mark.asyncio
async def test_breaker_blocks_opening_but_not_exiting(fake: FakeAlpaca, state_path: Path) -> None:
    fake.equity = "97400"  # -2.6% on the day
    fake.positions = [_position("NVDA", entry="120.00", current="118.00")]
    router = _router(fake, state_path)

    with pytest.raises(GuardBlocked) as blocked:
        await router.handle_signal(_signal())
    assert blocked.value.code == "circuit_breaker"

    closed = await router.close_position("NVDA", reason="manual_close")
    assert closed.decision == "accepted"
    assert [p["side"] for p in fake.order_posts] == ["sell"]


@pytest.mark.asyncio
async def test_breaker_stays_latched_for_the_day_and_across_restarts(fake: FakeAlpaca, state_path: Path) -> None:
    fake.equity = "97000"
    router = _router(fake, state_path)
    with pytest.raises(GuardBlocked):
        await router.handle_signal(_signal())

    fake.equity = "100500"  # recovered intraday
    with pytest.raises(GuardBlocked) as still:
        await _router(fake, state_path).handle_signal(_signal())  # fresh process, same volume
    assert still.value.code == "circuit_breaker"
    assert fake.order_posts == []


@pytest.mark.asyncio
async def test_breaker_fails_closed_when_pnl_is_unreadable(fake: FakeAlpaca, state_path: Path) -> None:
    fake.account_fails = True
    with pytest.raises(GuardBlocked) as blocked:
        await _router(fake, state_path).handle_signal(_signal())
    assert blocked.value.code == "breaker_unavailable"


@pytest.mark.asyncio
async def test_halt_blocks_every_order_and_survives_a_restart(fake: FakeAlpaca, state_path: Path) -> None:
    fake.positions = [_position("NVDA", entry="120.00", current="100.00")]  # deep past its stop
    router = _router(fake, state_path)
    router.guard.halt("test")

    for attempt in (router.handle_signal(_signal()), router.close_position("NVDA", reason="manual_close")):
        with pytest.raises(GuardBlocked) as blocked:
            await attempt
        assert blocked.value.code == "halted"
    assert await router.run_exit_pass() == []

    assert StateStore(state_path).halted  # what a restarted pod would read
    assert fake.order_posts == []


def test_unreadable_state_file_starts_halted(state_path: Path) -> None:
    state_path.write_text("{not json", encoding="utf-8")
    assert StateStore(state_path).halted


# --- exits -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exit_pass_enforces_stop_loss_and_take_profit(fake: FakeAlpaca, state_path: Path) -> None:
    fake.positions = [
        _position("TSLA", entry="354.89", current="337.00"),   # -5.04% → stop
        _position("AAPL", entry="228.00", current="251.00"),   # +10.09% → take profit
        _position("MSFT", entry="441.00", current="450.00"),   # +2.04% → hold
        _position("AMD", entry="142.00", current="130.00"),    # -8.45%, but a sell is already working
    ]
    fake.open_orders = [{"symbol": "AMD", "side": "sell", "qty": "0.028", "id": "x", "client_order_id": "x"}]

    decisions = await _router(fake, state_path).run_exit_pass()

    by_symbol = {d.symbol: (d.decision, d.code) for d in decisions}
    assert by_symbol == {
        "TSLA": ("accepted", "stop_loss"),
        "AAPL": ("accepted", "take_profit"),
        "AMD": ("rejected", "exit_in_flight"),
    }
    sells = {p["symbol"]: p for p in fake.order_posts}
    assert set(sells) == {"TSLA", "AAPL"}
    assert sells["TSLA"] | {"client_order_id": None} == {
        "symbol": "TSLA", "qty": "0.028", "side": "sell", "type": "market",
        "time_in_force": "day", "client_order_id": None,
    }


@pytest.mark.asyncio
async def test_exit_pass_does_nothing_while_the_market_is_closed(fake: FakeAlpaca, state_path: Path) -> None:
    fake.is_open = False
    fake.positions = [_position("TSLA", entry="354.89", current="300.00")]
    assert await _router(fake, state_path).run_exit_pass() == []
    assert fake.order_posts == []


@pytest.mark.asyncio
async def test_everything_is_sold_in_the_last_ten_minutes(fake: FakeAlpaca, state_path: Path) -> None:
    fake.positions = [_position("MSFT", entry="441.00", current="443.00"),   # +0.45%: would otherwise hold
                      _position("TSLA", entry="354.89", current="330.00")]   # past its stop anyway
    router = _router(fake, state_path)
    assert {d.symbol for d in await router.run_exit_pass()} == {"TSLA"}      # hours before the close

    fake.closes_at = datetime.now(UTC) + timedelta(minutes=9)
    fake.open_orders.clear()
    by_symbol = {d.symbol: d.code for d in await router.run_exit_pass()}
    assert by_symbol == {"MSFT": "session_close", "TSLA": "stop_loss"}


def test_exit_reason_boundaries() -> None:
    assert exit_reason(_position("X", entry="100", current="95")) == "stop_loss"      # exactly -5%
    assert exit_reason(_position("X", entry="100", current="95.01")) is None
    assert exit_reason(_position("X", entry="100", current="110")) == "take_profit"   # exactly +10%
    assert exit_reason(_position("X", entry="100", current="109.99")) is None


# --- portfolio limits ------------------------------------------------------------

def _snapshot(**overrides: Any) -> PortfolioSnapshot:
    fields: dict[str, Any] = {"positions": {}, "working_buys": {}, "working_sells": frozenset(), "buys_today": 0}
    fields.update(overrides)
    return PortfolioSnapshot(**fields)


@pytest.mark.parametrize(
    ("snapshot", "code"),
    [
        (_snapshot(working_buys={"TSLA": Decimal("7")}), "duplicate_accumulation"),
        (_snapshot(positions={s: Decimal("1") for s in "ABCDE"}), "max_open_positions"),
        (_snapshot(positions={"A": Decimal("25"), "B": Decimal("20")}), "gross_exposure"),
        (_snapshot(buys_today=tier0.MAX_NEW_BUYS_PER_DAY), "daily_buy_limit"),
        (_snapshot(last_sell_at={"TSLA": datetime.now(UTC) - timedelta(hours=1)}), "reentry_cooldown"),
    ],
)
def test_portfolio_limits(snapshot: PortfolioSnapshot, code: str) -> None:
    with pytest.raises(PolicyRejection) as rejected:
        check_can_open(snapshot, "TSLA", datetime.now(UTC))
    assert rejected.value.code == code


def test_cooldown_expires_and_other_symbols_are_unaffected() -> None:
    sold = _snapshot(last_sell_at={"TSLA": datetime.now(UTC) - timedelta(hours=5)})
    check_can_open(sold, "TSLA", datetime.now(UTC))
    check_can_open(_snapshot(positions={"AAPL": Decimal("9")}), "TSLA", datetime.now(UTC))


def test_snapshot_counts_working_orders_from_alpaca_shapes() -> None:
    day_start = datetime.now(UTC) - timedelta(hours=3)
    snap = PortfolioSnapshot.from_alpaca(
        positions=[{"symbol": "AAPL", "market_value": "-9.50"}],
        open_orders=[
            {"symbol": "TSLA", "side": "buy", "qty": "0.02", "filled_qty": "0.005", "limit_price": "354.89"},
            {"symbol": "NVDA", "side": "buy", "notional": "10"},
            {"symbol": "AMD", "side": "sell", "qty": "1"},
        ],
        recent_orders=[
            {"symbol": "TSLA", "side": "buy", "status": "accepted", "submitted_at": _iso(datetime.now(UTC))},
            {"symbol": "TSLA", "side": "buy", "status": "rejected", "submitted_at": _iso(datetime.now(UTC))},
            {"symbol": "GOOGL", "side": "buy", "status": "filled",
             "submitted_at": _iso(day_start - timedelta(hours=1))},  # yesterday: not counted
            {"symbol": "MSFT", "side": "sell", "status": "filled", "filled_at": _iso(datetime.now(UTC))},
        ],
        day_start=day_start,
    )
    assert snap.positions == {"AAPL": Decimal("9.50")}
    assert snap.working_buys == {"TSLA": Decimal("5.32335"), "NVDA": Decimal("10")}
    assert snap.working_sells == frozenset({"AMD"})
    assert snap.buys_today == 1
    assert set(snap.last_sell_at) == {"MSFT"}


# --- HTTP surface ----------------------------------------------------------------

@pytest.fixture
def secrets(monkeypatch: pytest.MonkeyPatch):
    # One value for both boundaries so the route audit can sign any route.
    monkeypatch.setenv("ROUTER_SIGNAL_SECRET", SECRET)
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    config.get_webhook_settings.cache_clear()
    yield
    config.get_webhook_settings.cache_clear()


def _signed(body: bytes) -> dict[str, str]:
    ts = str(int(time.time()))
    return {webhook.TIMESTAMP_HEADER: ts, webhook.SIGNATURE_HEADER: "sha256=" + webhook.sign(body, ts, SECRET),
            "Content-Type": "application/json"}


def _signal_body(**overrides: Any) -> bytes:
    return _signal(**overrides).model_dump_json().encode()


def test_every_execution_route_is_guarded(fake: FakeAlpaca, state_path: Path, secrets: None) -> None:
    """Enumerates the routes the app actually serves: a /v1 route added
    without the guard fails here, not in production."""
    router = _router(fake, state_path)
    router.guard.halt("route audit")
    with TestClient(create_app(router, background=False)) as client:
        paths = client.get("/openapi.json").json()["paths"]
        execution_routes = [
            (method, path) for path, methods in paths.items() for method in methods
            if path.startswith("/v1/") and not path.startswith("/v1/control/")
        ]
        assert {p for _, p in execution_routes} >= {"/v1/signals", "/v1/positions/{symbol}/close"}
        for method, path in execution_routes:
            body = _signal_body()
            response = client.request(method.upper(), path.replace("{symbol}", "TSLA"), content=body,
                                      headers=_signed(body))
            assert response.status_code == 423, (method, path, response.text)
            assert response.json()["detail"]["code"] == "halted"
    assert fake.order_posts == []


def test_signals_require_a_valid_signature(fake: FakeAlpaca, state_path: Path, secrets: None) -> None:
    with TestClient(create_app(_router(fake, state_path), background=False)) as client:
        body = _signal_body()
        assert client.post("/v1/signals", content=body).status_code == 403
        forged = _signed(body) | {webhook.SIGNATURE_HEADER: "sha256=" + "0" * 64}
        assert client.post("/v1/signals", content=body, headers=forged).status_code == 403
        accepted = client.post("/v1/signals", content=body, headers=_signed(body))
        assert accepted.status_code == 200
        assert accepted.json()["decision"] == "accepted"


def test_http_breaker_returns_423_for_signals_but_allows_manual_close(
    fake: FakeAlpaca, state_path: Path, secrets: None,
) -> None:
    fake.equity = "97000"
    fake.positions = [_position("NVDA", entry="120.00", current="118.00")]
    with TestClient(create_app(_router(fake, state_path), background=False)) as client:
        body = _signal_body()
        blocked = client.post("/v1/signals", content=body, headers=_signed(body))
        assert blocked.status_code == 423
        assert blocked.json()["detail"]["code"] == "circuit_breaker"

        closed = client.post("/v1/positions/NVDA/close", content=b"", headers=_signed(b""))
        assert closed.status_code == 200
        assert closed.json()["code"] == "manual_close"
        assert client.get("/health").json()["breaker_latched_today"] is True


def test_halt_endpoint_persists_and_cancels_orders(fake: FakeAlpaca, state_path: Path, secrets: None) -> None:
    fake.open_orders = [{"symbol": "TSLA", "side": "buy", "id": "o1"}]
    with TestClient(create_app(_router(fake, state_path), background=False)) as client:
        response = client.post("/v1/control/halt", content=b"", headers=_signed(b""))
        assert response.json() == {"halted": True, "orders_canceled": 1}
        assert client.get("/health").json()["halted"] is True
    assert StateStore(state_path).halted


def test_client_refuses_a_non_paper_host(monkeypatch: pytest.MonkeyPatch) -> None:
    import risk_router.alpaca_async as mod
    from risk_router.alpaca_async import SandboxViolationError

    monkeypatch.setattr(mod, "PAPER_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(SandboxViolationError):
        AsyncAlpaca("key", "secret")


def test_guard_intent_matrix(state_path: Path, fake: FakeAlpaca) -> None:
    router = _router(fake, state_path)
    # REDUCE never consults the breaker, so it passes even with P&L unreadable.
    fake.account_fails = True
    asyncio.run(router.guard.check(Intent.REDUCE))
