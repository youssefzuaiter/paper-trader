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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status

try:
    import fcntl
except ImportError:      # not POSIX: no single-instance lock; the deployment targets are macOS and Linux
    fcntl = None  # type: ignore[assignment]

import tier0_core
from risk_router.alpaca_async import AlpacaError, AsyncAlpaca
from risk_router.app import _hmac_dependency, verify_control_signature
from risk_router.core_alerts import Alerter
from risk_router.core_gatekeeper import CoreRouter, CoreStore, Journal, PlanRejected
from risk_router.guards import CircuitBreaker, ExecutionGuard
from risk_router.state import StateStore

logger = logging.getLogger("risk_router.core_app")

#: HTTP 422. Starlette renamed its constant for this (and deprecated the old name); the number is stable.
UNPROCESSABLE: Final[int] = 422
TICK_SECONDS: Final[float] = 15.0
#: Four missed ticks: the execution loop is not running, or is failing every time.
STALE_TICK_SECONDS: Final[float] = 4 * TICK_SECONDS
#: A fresh process has not ticked yet; give the first tick this long before calling the loop stale.
STARTUP_GRACE_SECONDS: Final[float] = 60.0
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
    code = status.HTTP_409_CONFLICT if exc.code in ("plan_mismatch", "no_such_plan", "plan_in_flight") else \
        UNPROCESSABLE
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
    try:
        # A plan the process could not finish is closed (and ``forced`` set) before the Allocator reads the
        # facts it decides from: otherwise it would decide as though that plan were still going to complete.
        await router.reconcile()
    except AlpacaError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    data = router.store.data
    return {"forced": data.forced, "targets": data.targets, "plan": data.plan and
            {k: data.plan[k] for k in ("id", "kind", "status", "execute_on")}}


@control_routes.get("/core/plan")
async def current_plan(request: Request) -> dict[str, Any]:
    """What the owner is being asked to approve (or what is trading now): the orders, in plain terms."""
    router = _router(request)
    try:
        await router.reconcile()
    except AlpacaError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    record = router.store.data.plan
    if record is None:
        return {"plan": None, "forced": router.store.data.forced}
    return {"plan": {**{k: record[k] for k in ("id", "kind", "status", "decided_on", "execute_on", "redecision")},
                     "sells": record["plan"]["sells"], "buys": record["plan"]["buys"],
                     "orders": {cid: {k: o.get(k) for k in ("symbol", "side", "status", "filled_qty", "filled_avg_price")}
                                for cid, o in record["orders"].items()}},
            "forced": router.store.data.forced}


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
        raise HTTPException(UNPROCESSABLE, "amount and need_by (YYYY-MM-DD) required") from exc
    if amount <= 0:
        raise HTTPException(UNPROCESSABLE, "amount must be positive")
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


def _attention(request: Request) -> list[str]:
    """Everything about this process that a person should look at; empty means all is well."""
    state = request.app.state
    guard: ExecutionGuard = state.guard
    out: list[str] = []
    if state.disabled:
        out.append(f"trading is disabled: {state.disabled}")
    if guard.state.halted:
        out.append(f"the kill switch is on: {guard.state.halt_reason}")
    router: CoreRouter | None = getattr(state, "router", None)
    if router is None:
        return out
    detail = router.health()
    if not detail["journal"]["ok"]:
        out.append(f"the journal failed its integrity check: {detail['journal']['reason']}")
    if state.background and not state.disabled:
        age = detail["last_tick_age_seconds"]
        uptime = (state.clock() - state.started_at).total_seconds()
        if (age is None and uptime > STARTUP_GRACE_SECONDS) or (age is not None and age > STALE_TICK_SECONDS):
            out.append("the execution loop has stopped ticking" if age is not None
                       else "the execution loop has not completed a tick since start-up")
    if router.store.data.plan and router.store.data.plan["status"] == "awaiting_approval":
        out.append(f"plan {router.store.data.plan['id']} is waiting for your signed approval")
    if router.store.data.initial_incomplete:
        out.append("the first build did not finish; the next close will propose the rest")
    return out


@ops_routes.get("/health")
async def health(request: Request) -> dict[str, Any]:
    policy = request.app.state.policy
    guard: ExecutionGuard = request.app.state.guard
    router: CoreRouter | None = getattr(request.app.state, "router", None)
    plan = router.store.data.plan if router else None
    detail = router.health() if router else None
    attention = _attention(request)
    return {"status": "ok" if not attention else "attention", "environment": "paper", "mode": "core",
            "trading_enabled": not request.app.state.disabled, "disabled_reason": request.app.state.disabled,
            "halted": guard.state.halted, "halt_reason": guard.state.halt_reason,
            "policy_sha256": policy.sha256 if policy else None,
            "policy_effective_from": policy.effective_from.isoformat() if policy else None,
            "plan": plan and {k: plan[k] for k in ("id", "kind", "status", "execute_on")},
            "journal": detail and detail["journal"],
            "tick": detail and {"last_age_seconds": detail["last_tick_age_seconds"], "failures": detail["tick_failures"]},
            "attention": attention,
            "limits": {"max_turnover_pct_per_rebalance": str(tier0_core.MAX_TURNOVER_PER_REBALANCE_PCT),
                       "max_order_notional_usd": str(tier0_core.MAX_ORDER_NOTIONAL_USD),
                       "max_rebalance_plans_per_month": tier0_core.MAX_REBALANCE_PLANS_PER_MONTH,
                       "limit_collar_pct": str(tier0_core.LIMIT_COLLAR_PCT), "exits": "none"}}


async def _loop(router: CoreRouter) -> None:
    while True:
        try:
            await router.run_tick()     # counts and journals failures itself; the next tick retries
        except Exception:
            logger.exception("core loop: bookkeeping failed")  # e.g. a full disk: still never stop ticking
        await asyncio.sleep(TICK_SECONDS)


def _take_instance_lock(root: Path) -> Any:
    """An exclusive, non-blocking lock on ``<state dir>/.core.lock``, held for the process's life.

    Two routers on one state directory would each read the same approved plan and each send its orders (Alpaca's
    client-order-id dedupe would catch most of that, but the journal and the plan state would be written by both).
    The classic way to get there is a leftover process plus a restarted container. Held by the open file, so a
    crashed process releases it automatically. Two *hosts* are not covered; they would need a shared store."""
    root.mkdir(parents=True, exist_ok=True)
    handle = open(root / ".core.lock", "a+")  # noqa: SIM115 - kept open on purpose: the lock lives as long as it does
    if fcntl is not None:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise PlanRejected("already_running", f"another core router is already running on {root}: two routers on "
                                                  f"one state directory would each place the orders") from exc
    return handle


def create_core_app(*, alpaca: Any = None, background: bool = True, state_dir: Path | None = None,
                    policy_path: Path | None = None, now: Callable[[], datetime] | None = None,
                    alerter: Alerter | None = None) -> FastAPI:
    """``now`` and ``alerter`` exist for tests; production passes neither (real clock, alerts from the env)."""
    clock = now or (lambda: datetime.now(UTC))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        root = state_dir or Path(os.getenv("CORE_STATE_DIR", "core-state"))
        app.state.disabled = None
        app.state.policy = None
        app.state.background = background
        app.state.clock = clock
        app.state.started_at = clock()
        app.state.alerter = alerter or Alerter.from_env()
        app.state.alpaca = alpaca or AsyncAlpaca(os.environ["CORE_ALPACA_KEY_ID"], os.environ["CORE_ALPACA_SECRET_KEY"])
        state = StateStore(root / "core-router-state.json")
        app.state.guard = ExecutionGuard(state, CircuitBreaker(app.state.alpaca, state, now=clock))
        task = None
        instance_lock = None
        try:
            instance_lock = _take_instance_lock(root)    # a second router on this directory must not trade
            policy = tier0_core.load_policy(policy_path or Path(os.getenv("CORE_POLICY_PATH", str(DEFAULT_POLICY))))
            app.state.policy = policy
            store = CoreStore(root / "core-plan-state.json")
            router = CoreRouter(app.state.alpaca, app.state.guard, policy, store,
                                Journal(root / "core-journal.jsonl", store, policy.sha256),
                                now=clock, alerter=app.state.alerter)
            app.state.router = router
            journal = router.journal.status()
            if not journal["ok"]:
                # Not written to the journal: appending to a tampered chain would only hide the damage.
                app.state.alerter.notify("Core: the journal failed its integrity check", str(journal["reason"]))
                raise PlanRejected("journal_invalid", f"the journal failed its integrity check: {journal['reason']}")
            await router.check_account()
            router.journal.write("started", account=policy.account)
            if background:
                task = asyncio.create_task(_loop(router))
        except (tier0_core.PolicyError, PlanRejected, AlpacaError, OSError) as exc:
            app.state.disabled = str(exc).strip()      # Alpaca's error bodies end in a newline
            logger.critical("CORE TRADING DISABLED: %s", exc)
        yield
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if instance_lock is not None:
            instance_lock.close()                        # closing the file releases the lock
        if alpaca is None:
            await app.state.alpaca.aclose()

    app = FastAPI(title="Long-term core router", version="0.1.0", lifespan=lifespan)
    for group in (allocator_routes, control_routes, ops_routes):
        app.include_router(group)
    return app
