"""HTTP surface of the Risk & Routing agent.

Routes are grouped by what they are allowed to do, and the guard is
attached to the GROUP, not to each handler — a route added to
``open_routes`` or ``reduce_routes`` later inherits the kill switch and the
circuit breaker without anyone having to remember them. (The monolith's
``/signals/execute`` is what happens when they have to be remembered.)
``tests/test_risk_router.py`` fails if any ``/v1`` execution route is
mounted without the guard.

===============================  ========  ===============================
Route                            Auth      Guard
===============================  ========  ===============================
``POST /v1/signals``             signal    OPEN   (halt + breaker)
``POST /v1/positions/{s}/close`` control   REDUCE (halt only)
``POST /v1/control/halt``        control   none — must work in any state
``GET  /health``                 none      none
===============================  ========  ===============================

Two secrets, two trust boundaries: Inference Agents sign signals with
``ROUTER_SIGNAL_SECRET``; PFW's server signs control calls with the
existing ``WEBHOOK_SECRET``. An inference pod therefore cannot forge a
halt, a manual exit, or a receipt to PFW.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import ValidationError

import tier0
import webhook
from config import ConfigError, get_webhook_settings
from outbox import default_outbox
from risk_router.alpaca_async import AlpacaError, AsyncAlpaca, OrderRejected
from risk_router.gatekeeper import RiskRouter
from risk_router.guards import CircuitBreaker, ExecutionGuard, GuardBlocked, Intent
from risk_router.schemas import Decision, TradeSignal
from risk_router.state import StateStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
logger = logging.getLogger("risk_router")

REPLAY_WINDOW_SECONDS: Final[int] = 300


# --- authentication ----------------------------------------------------------

def _signal_secret() -> str:
    secret = os.getenv("ROUTER_SIGNAL_SECRET", "").strip()
    if len(secret) < 32:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ROUTER_SIGNAL_SECRET is not configured (32+ chars)")
    return secret


def _control_secret() -> str:
    try:
        return get_webhook_settings().secret
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def _hmac_dependency(secret_for: Callable[[], str]) -> Callable[[Request], Awaitable[None]]:
    """Same scheme as every other HMAC edge in this repo (webhook.py):
    sha256 over ``f"{timestamp}." + raw body``, 300 s replay window."""

    async def verify(request: Request) -> None:
        secret = secret_for()
        raw = await request.body()
        timestamp = request.headers.get(webhook.TIMESTAMP_HEADER)
        signature = request.headers.get(webhook.SIGNATURE_HEADER)
        if not timestamp or not signature:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing signature headers")
        try:
            skew = abs(time.time() - int(timestamp))
        except ValueError as exc:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid timestamp") from exc
        if skew > REPLAY_WINDOW_SECONDS:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Stale signature")
        if not webhook.verify(raw, timestamp, signature, secret):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid signature")

    return verify


verify_signal_signature = _hmac_dependency(_signal_secret)
verify_control_signature = _hmac_dependency(_control_secret)


# --- the execution guard, as a route dependency --------------------------------

def execution_guard(intent: Intent) -> Callable[[Request], Awaitable[None]]:
    async def guard(request: Request) -> None:
        router: RiskRouter = request.app.state.router
        try:
            await router.guard.check(intent)
        except GuardBlocked as exc:
            raise HTTPException(status.HTTP_423_LOCKED, {"code": exc.code, "detail": exc.detail}) from exc

    guard.execution_guard_intent = intent  # type: ignore[attr-defined] — read by the route audit test
    return guard


# Auth first, then the guard: an unauthenticated caller learns nothing
# about whether trading is halted.
open_routes = APIRouter(prefix="/v1", tags=["execution"],
                        dependencies=[Depends(verify_signal_signature), Depends(execution_guard(Intent.OPEN))])
reduce_routes = APIRouter(prefix="/v1", tags=["execution"],
                          dependencies=[Depends(verify_control_signature), Depends(execution_guard(Intent.REDUCE))])
control_routes = APIRouter(prefix="/v1/control", tags=["control"],
                           dependencies=[Depends(verify_control_signature)])
ops_routes = APIRouter(tags=["ops"])


def _blocked(exc: GuardBlocked) -> HTTPException:
    return HTTPException(status.HTTP_423_LOCKED, {"code": exc.code, "detail": exc.detail})


def _broker_failure(exc: AlpacaError) -> HTTPException:
    code = status.HTTP_422_UNPROCESSABLE_ENTITY if isinstance(exc, OrderRejected) else status.HTTP_502_BAD_GATEWAY
    return HTTPException(code, str(exc))


@open_routes.post("/signals")
async def submit_signal(request: Request) -> Decision:
    """An Inference Agent's signal. ``200`` with ``decision: rejected`` is a
    normal outcome; ``423`` means the kill switch or breaker is engaged and
    the caller should stop sending until it clears."""
    try:
        signal = TradeSignal.model_validate_json(await request.body())
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors(include_url=False)) from exc
    router: RiskRouter = request.app.state.router
    try:
        return await router.handle_signal(signal)
    except GuardBlocked as exc:  # engaged between the dependency and the decision
        raise _blocked(exc) from exc
    except AlpacaError as exc:
        raise _broker_failure(exc) from exc


@reduce_routes.post("/positions/{symbol}/close")
async def close_position(symbol: str, request: Request) -> Decision:
    """Manual exit. Allowed while the breaker is tripped; not while halted."""
    router: RiskRouter = request.app.state.router
    try:
        return await router.close_position(symbol, reason="manual_close")
    except GuardBlocked as exc:
        raise _blocked(exc) from exc
    except AlpacaError as exc:
        raise _broker_failure(exc) from exc


@control_routes.post("/halt")
async def halt(request: Request) -> dict[str, Any]:
    """Engage the kill switch and cancel every working order.

    Persisted on the pod's volume: a restart does NOT clear it. To resume,
    an operator deletes ``$ROUTER_STATE_DIR/router-state.json`` and
    restarts the pod — deliberate by design, as in the monolith.
    """
    router: RiskRouter = request.app.state.router
    router.guard.halt("Emergency halt via POST /v1/control/halt")
    try:
        canceled = await router.alpaca.cancel_all_orders()
    except Exception as exc:
        logger.exception("Halt: cancelling open orders failed")
        return {"halted": True, "orders_canceled": 0, "cancel_error": str(exc)}
    return {"halted": True, "orders_canceled": canceled}


@ops_routes.get("/health")
async def health(request: Request) -> dict[str, Any]:
    router: RiskRouter = request.app.state.router
    guard = router.guard
    latched = guard.breaker.latched_today()
    return {
        "status": "ok",
        "environment": "paper",
        "halted": guard.state.halted,
        "halt_reason": guard.state.halt_reason,
        "breaker_latched_today": latched,
        "breaker_detail": guard.state.breaker_detail if latched else None,
        "exit_loop_running": bool(getattr(request.app.state, "exit_task", None)
                                  and not request.app.state.exit_task.done()),
        "outbox_pending": default_outbox.pending_count(),
        "limits": {
            "max_notional_usd": str(tier0.MAX_NOTIONAL_USD),
            "max_gross_exposure_usd": str(tier0.MAX_GROSS_EXPOSURE_USD),
            "max_open_positions": tier0.MAX_OPEN_POSITIONS,
            "max_new_buys_per_day": tier0.MAX_NEW_BUYS_PER_DAY,
            "stop_loss_pct": str(tier0.STOP_LOSS_PCT),
            "take_profit_pct": str(tier0.TAKE_PROFIT_PCT),
            "circuit_breaker_daily_pnl_pct": str(tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT),
            "min_prob_up": str(tier0.ROUTER_MIN_PROB_UP),
        },
    }


# --- receipts to PFW -----------------------------------------------------------

async def send_pending_receipt(plan: tier0.ExecutionPlan, signal: TradeSignal, order: dict[str, Any]) -> None:
    """The same signed "pending" receipt the monolith sends, through the
    same outbox. The fill's "settled" receipt comes from the settlement
    stream / reconciler, which settle every fill on the account."""
    receipt_signal = SimpleNamespace(
        model_name=signal.model_name,
        predicted_move_pct=signal.predicted_move_pct,
        confidence=signal.confidence,
        headline=signal.headline or "",  # PFW's schema takes a string or nothing, never null
    )
    receipt_order = SimpleNamespace(id=order["id"], client_order_id=order["client_order_id"], status=order["status"])
    try:
        delivery = await webhook.send_trade_receipt(plan=plan, signal=receipt_signal, order=receipt_order,
                                                    executed_at=order_submitted_at(order))
    except ConfigError as exc:
        logger.error("Receipt for order %s not sent: %s", order["id"], exc)
        return
    if not delivery.get("delivered"):
        logger.warning("Receipt for order %s not delivered: %s", order["id"], delivery.get("error"))


def order_submitted_at(order: dict[str, Any]) -> datetime:
    raw = order.get("submitted_at") or order.get("created_at")
    return datetime.fromisoformat(raw) if raw else datetime.now(UTC)


# --- app factory -----------------------------------------------------------------

def build_router_from_env() -> RiskRouter:
    alpaca = AsyncAlpaca.from_settings()  # a gatekeeper without broker keys should not start
    state = StateStore(Path(os.getenv("ROUTER_STATE_DIR", "state")) / "router-state.json")
    guard = ExecutionGuard(state, CircuitBreaker(alpaca, state))
    return RiskRouter(alpaca, guard, on_submitted=send_pending_receipt)


def create_app(router: RiskRouter | None = None, *, background: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.router = router or build_router_from_env()
        if app.state.router.guard.state.halted:
            logger.critical("Starting HALTED: %s", app.state.router.guard.state.halt_reason)
        tasks: list[asyncio.Task[None]] = []
        if background:
            app.state.exit_task = asyncio.create_task(app.state.router.exit_loop())
            tasks = [app.state.exit_task, asyncio.create_task(default_outbox.replay_loop(webhook.send_once))]
        yield
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await app.state.router.alpaca.aclose()

    app = FastAPI(title="Risk & Routing Agent", version="0.1.0", lifespan=lifespan)
    for group in (open_routes, reduce_routes, control_routes, ops_routes):
        app.include_router(group)
    return app


app = create_app()
