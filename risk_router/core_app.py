"""HTTP surface of the long-term core's router (core design §8.1): a separate app, a separate paper
account, separate keys. The mode is fixed by which app runs, never by a request field or a flag.

===================================  ==========  ==========================================
Route                                Auth        Does
===================================  ==========  ==========================================
``POST /v1/core/plans``              allocator   the Allocator's proposal: verify, admit
``GET  /v1/core/state``              allocator   the router's facts the Allocator needs
``POST /v1/core/plans/{id}/approve`` control     the owner's approval (``fund`` for the build)
``POST /v1/core/raise-cash``         control     the owner's "raise X by date": a plan to approve
``POST /v1/control/halt``            control     kill switch + cancel open orders
``GET  /health``                     none        status, policy hash, why trading is disabled
===================================  ==========  ==========================================

The Allocator signs with ``CORE_PLAN_SECRET``; the owner's control calls with PFW's existing
``WEBHOOK_SECRET`` (the news router's control secret), so PFW's halt stops both apps. Keys:
``CORE_ALPACA_KEY_ID`` / ``CORE_ALPACA_SECRET_KEY`` for the core's own paper account; the client is
the news router's paper-locked ``AsyncAlpaca``.

At start-up the policy is validated against ``tier0_core`` and the connected account must be the
policy's. Either failure **disables** core trading (fail closed): every trading route answers 503,
no execution loop runs, and ``/health`` says why.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status

import tier0_core
from risk_router.alpaca_async import AlpacaError, AsyncAlpaca
from risk_router.app import _hmac_dependency, verify_control_signature
from risk_router.core_gatekeeper import CoreRouter, CoreStore, Journal, PlanRejected
from risk_router.guards import CircuitBreaker, ExecutionGuard
from risk_router.state import StateStore

logger = logging.getLogger("risk_router.core_app")

TICK_SECONDS: Final[float] = 15.0
DEFAULT_POLICY: Final[Path] = Path(__file__).resolve().parent.parent / "policy" / "core.toml"


def _plan_secret() -> str:
    secret = os.getenv("CORE_PLAN_SECRET", "").strip()
    if len(secret) < 32:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "CORE_PLAN_SECRET is not configured (32+ chars)")
    return secret


verify_plan_signature = _hmac_dependency(_plan_secret)


def _router(request: Request) -> CoreRouter:
    disabled = getattr(request.app.state, "disabled", None)
    if disabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, {"code": "core_disabled", "detail": disabled})
    return request.app.state.router


def _rejected(exc: PlanRejected) -> HTTPException:
    code = status.HTTP_409_CONFLICT if exc.code in ("plan_mismatch", "no_such_plan") else \
        status.HTTP_422_UNPROCESSABLE_ENTITY
    return HTTPException(code, {"code": exc.code, "detail": exc.detail})


allocator_routes = APIRouter(prefix="/v1/core", tags=["core"], dependencies=[Depends(verify_plan_signature)])
control_routes = APIRouter(prefix="/v1", tags=["control"], dependencies=[Depends(verify_control_signature)])
ops_routes = APIRouter(tags=["ops"])


@allocator_routes.post("/plans")
async def propose(request: Request) -> dict[str, Any]:
    router = _router(request)
    try:
        return await router.receive_plan(await request.json())
    except PlanRejected as exc:
        raise _rejected(exc) from exc
    except AlpacaError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


@allocator_routes.get("/state")
async def core_state(request: Request) -> dict[str, Any]:
    router = _router(request)
    data = router.store.data
    return {"forced": data.forced, "targets": data.targets, "plan": data.plan and
            {k: data.plan[k] for k in ("id", "kind", "status", "execute_on")}}


@control_routes.post("/core/plans/{pid}/approve")
async def approve(pid: str, request: Request) -> dict[str, Any]:
    router = _router(request)
    body = await request.json() if await request.body() else {}
    try:
        return await router.approve(pid, fund=bool(body.get("fund", False)))
    except PlanRejected as exc:
        raise _rejected(exc) from exc


@control_routes.post("/core/raise-cash")
async def raise_cash(request: Request) -> dict[str, Any]:
    router = _router(request)
    body = await request.json()
    try:
        amount, need_by = Decimal(str(body["amount"])), date.fromisoformat(str(body["need_by"]))
    except (KeyError, ValueError, InvalidOperation) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "amount and need_by (YYYY-MM-DD) required") from exc
    if amount <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "amount must be positive")
    try:
        return await router.raise_cash(amount, need_by)
    except PlanRejected as exc:
        raise _rejected(exc) from exc


@control_routes.post("/control/halt")
async def halt(request: Request) -> dict[str, Any]:
    """Works even when core trading is disabled: a halt must always be possible."""
    guard: ExecutionGuard = request.app.state.guard
    guard.halt("Emergency halt via POST /v1/control/halt (core)")
    try:
        canceled = await request.app.state.alpaca.cancel_all_orders()
    except Exception as exc:
        logger.exception("Core halt: cancelling open orders failed")
        return {"halted": True, "orders_canceled": 0, "cancel_error": str(exc)}
    return {"halted": True, "orders_canceled": canceled}


@ops_routes.get("/health")
async def health(request: Request) -> dict[str, Any]:
    policy = request.app.state.policy
    guard: ExecutionGuard = request.app.state.guard
    plan = request.app.state.router.store.data.plan if getattr(request.app.state, "router", None) else None
    return {"status": "ok", "environment": "paper", "mode": "core",
            "trading_enabled": not request.app.state.disabled, "disabled_reason": request.app.state.disabled,
            "halted": guard.state.halted, "halt_reason": guard.state.halt_reason,
            "policy_sha256": policy.sha256 if policy else None,
            "plan": plan and {k: plan[k] for k in ("id", "kind", "status", "execute_on")},
            "limits": {"max_turnover_pct_per_rebalance": str(tier0_core.MAX_TURNOVER_PER_REBALANCE_PCT),
                       "max_order_notional_usd": str(tier0_core.MAX_ORDER_NOTIONAL_USD),
                       "max_rebalance_plans_per_month": tier0_core.MAX_REBALANCE_PLANS_PER_MONTH,
                       "limit_collar_pct": str(tier0_core.LIMIT_COLLAR_PCT), "exits": "none"}}


async def _loop(router: CoreRouter) -> None:
    while True:
        try:
            await router.tick()
        except Exception:
            logger.exception("core tick failed")  # the next tick retries; orders are at-most-once by id
        await asyncio.sleep(TICK_SECONDS)


def create_core_app(*, alpaca: Any = None, background: bool = True, state_dir: Path | None = None,
                    policy_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        root = state_dir or Path(os.getenv("CORE_STATE_DIR", "core-state"))
        app.state.disabled = None
        app.state.policy = None
        app.state.alpaca = alpaca or AsyncAlpaca(os.environ["CORE_ALPACA_KEY_ID"], os.environ["CORE_ALPACA_SECRET_KEY"])
        state = StateStore(root / "core-router-state.json")
        app.state.guard = ExecutionGuard(state, CircuitBreaker(app.state.alpaca, state))
        task = None
        try:
            policy = tier0_core.load_policy(policy_path or Path(os.getenv("CORE_POLICY_PATH", str(DEFAULT_POLICY))))
            app.state.policy = policy
            store = CoreStore(root / "core-plan-state.json")
            router = CoreRouter(app.state.alpaca, app.state.guard, policy, store,
                                Journal(root / "core-journal.jsonl", store, policy.sha256))
            app.state.router = router
            await router.check_account()
            router.journal.write("started", account=policy.account)
            if background:
                task = asyncio.create_task(_loop(router))
        except (tier0_core.PolicyError, PlanRejected, AlpacaError, OSError) as exc:
            app.state.disabled = str(exc)
            logger.critical("CORE TRADING DISABLED: %s", exc)
        yield
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if alpaca is None:
            await app.state.alpaca.aclose()

    app = FastAPI(title="Long-term core router", version="0.1.0", lifespan=lifespan)
    for group in (allocator_routes, control_routes, ops_routes):
        app.include_router(group)
    return app
