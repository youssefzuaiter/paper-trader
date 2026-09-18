"""FastAPI entry point for the Tier-0 Alpaca paper trading agent.

Run with::

    uvicorn main:app --reload --port 8000

Pipeline::

    ingest (simulated)  ->  inference.predict_move  ->  execution.validate_and_plan
                        ->  Alpaca paper submit_order  ->  signed receipt -> Next.js

Every order this service can produce is a sandbox order. See ``broker.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

import broker
import execution
import inference
import webhook
from outbox import default_outbox
from reconcile import reconcile_loop, reconcile_recent_fills
from config import (
    ConfigError,
    broker_is_configured,
    get_broker_settings,
    get_webhook_settings,
    is_autonomous_mode_enabled,
)
from models import autoencoder
from scheduler import (
    _log_event,
    agent_telemetry,
    run_signal_cycle,
    trading_loop,
    trigger_emergency_halt,
    websocket_settlement_stream,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("tier0")


class SignalRequest(BaseModel):
    """Evaluate or execute a single ticker.

    ``ticker`` is optional: omit it (leave both fields out entirely) to
    have the server pick a random scenario from
    ``inference.MARKET_SCENARIOS`` instead — a quick way to stress-test
    the Tier-0 gates across varied tickers/sentiment without hardcoding
    one headline in every call.
    """

    ticker: str | None = Field(
        default=None, min_length=1, max_length=8, examples=["NVDA"]
    )
    headline: str | None = Field(
        default=None,
        description="Headline to score. Omit to pull one from the simulated feed.",
        examples=["NVDA beats quarterly estimates and raises full-year guidance"],
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the model and fail fast on misconfiguration."""
    logger.info("Tier-0 paper trader starting (PAPER_ONLY=%s)", broker.PAPER_ONLY)

    # The two settings models resolve independently, so a missing Alpaca key
    # degrades exactly one capability instead of the whole service: the
    # receipt path signs and delivers with no broker credentials at all.
    # Neither failure is fatal — /health must stay reachable so an operator
    # can see *why* the service is degraded.
    try:
        logger.info("Webhook config: %s", get_webhook_settings().redacted())
    except ConfigError as exc:
        logger.error("Receipt delivery UNAVAILABLE: %s", exc)

    if broker_is_configured():
        logger.info("Broker config: %s", get_broker_settings().redacted())
    else:
        logger.warning(
            "Broker credentials absent — order placement UNAVAILABLE "
            "(/signals/execute returns 503). Receipt signing and delivery are "
            "unaffected."
        )

    # Pay the model load cost at boot, not on the first order.
    await inference.predict_move("WARMUP", "service startup warmup headline")
    logger.info("Sentiment model warm (%s)", inference.MODEL_NAME)

    logger.info(
        "Tier-0 limits: min gain %s%%, max notional $%s, stop %s%%",
        execution.MIN_PREDICTED_GAIN_PCT,
        execution.MAX_NOTIONAL_USD,
        execution.STOP_LOSS_PCT,
    )

    # AUTONOMOUS_MODE gates the background runner entirely at boot — off
    # by default (config.is_autonomous_mode_enabled() returns False for
    # any unset/non-truthy value), so a fresh checkout never starts
    # submitting trades on its own.
    autonomous_task: asyncio.Task[None] | None = None
    if is_autonomous_mode_enabled():
        logger.info("AUTONOMOUS_MODE enabled — starting background trading_loop()")
        autonomous_task = asyncio.create_task(trading_loop())
    else:
        logger.info("AUTONOMOUS_MODE disabled — no background trading loop started")

    # websocket_settlement_stream runs UNCONDITIONALLY — not gated on
    # AUTONOMOUS_MODE — so it also catches fills for orders submitted
    # manually (e.g. a direct /signals/execute call), not only ones the
    # autonomous loop itself created. See its own docstring in
    # scheduler.py. Replaces the earlier 30s REST-polling settlement_loop
    # with Alpaca's real-time trade_updates WebSocket.
    logger.info("Starting background websocket_settlement_stream()")
    settlement_task = asyncio.create_task(websocket_settlement_stream())

    # Durable outbox (see outbox.py): re-signs and replays any receipt
    # PFW never acknowledged. Runs unconditionally, like the settlement
    # stream — a queued receipt from a manual /signals/execute needs
    # delivering just as much as one from the autonomous loop.
    logger.info("Starting background outbox.replay_loop() (%d pending)", default_outbox.pending_count())
    outbox_task = asyncio.create_task(default_outbox.replay_loop(webhook.send_once))

    # Reconciliation against Alpaca's order history (see reconcile.py):
    # one pass shortly after startup, then hourly. Closes the gap the
    # outbox can't — a settlement this process THOUGHT was delivered, or
    # never generated at all.
    logger.info("Starting background reconcile_loop()")
    reconcile_task = asyncio.create_task(reconcile_loop(on_recovered=_telemetry_for_recovered_fill))

    yield

    for task in (reconcile_task, outbox_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if autonomous_task is not None:
        autonomous_task.cancel()
        try:
            await autonomous_task
        except asyncio.CancelledError:
            pass
        logger.info("Background trading_loop cancelled cleanly")

    settlement_task.cancel()
    try:
        await settlement_task
    except asyncio.CancelledError:
        pass
    logger.info("Background websocket_settlement_stream cancelled cleanly")

    logger.info("Tier-0 paper trader shutting down")


app = FastAPI(
    title="Tier-0 Paper Trading Agent",
    version="1.0.0",
    description=(
        "Sandbox-only equities agent. Sentiment inference gated by a "
        "deterministic execution layer, with HMAC-signed receipts."
    ),
    lifespan=lifespan,
)

# CORS: /telemetry is read from a browser running PFW's Next.js dev server
# on a different origin (localhost:3000). Scoped to that one local origin
# and GET only — this service has no cookies/auth to leak, but there is
# no reason to open it wider than the one real caller.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _webhook_is_configured() -> bool:
    """Mirror of `broker_is_configured` for the delivery path."""
    try:
        get_webhook_settings()
    except ConfigError:
        return False
    return True


@app.get("/telemetry", tags=["ops"])
async def telemetry() -> list[dict[str, Any]]:
    """The last (up to) 50 autonomous-agent events, oldest first.

    In-memory only (scheduler.agent_telemetry, a bounded deque) — resets
    on process restart. Populated only while AUTONOMOUS_MODE is enabled;
    an empty list is the correct, expected response otherwise, not an
    error.
    """
    return list(agent_telemetry)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    """Liveness probe. Never touches the broker, so it works without credentials."""
    return {
        "status": "ok",
        "environment": "paper",
        "paper_only": broker.PAPER_ONLY,
        # Makes the degraded state observable rather than something you only
        # discover by getting a 503 from /signals/execute.
        "broker_configured": broker_is_configured(),
        "webhook_configured": _webhook_is_configured(),
        # Receipts queued for replay because PFW never acknowledged them
        # (outbox.py). PFW's dashboard badge surfaces a non-zero count.
        "outbox_pending": default_outbox.pending_count(),
        "model": inference.MODEL_NAME,
        "tier0_limits": {
            "min_predicted_gain_pct": str(execution.MIN_PREDICTED_GAIN_PCT),
            "max_notional_usd": str(execution.MAX_NOTIONAL_USD),
            "stop_loss_pct": str(execution.STOP_LOSS_PCT),
        },
    }


@app.get("/account", tags=["ops"])
async def account() -> dict[str, Any]:
    """Paper account snapshot. Confirms credentials reach the sandbox."""
    try:
        acct = broker.get_account()
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch paper account")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return {
        "account_number": acct.account_number,
        "status": str(acct.status),
        "buying_power": str(acct.buying_power),
        "cash": str(acct.cash),
        "equity": str(acct.equity),
    }


@app.get("/positions", tags=["ops"])
async def positions() -> list[dict[str, str]]:
    """Real, live Alpaca positions for the paper account.

    Read-only — used by PFW's ``scripts/reconcile-ledger.ts`` nightly
    audit to compare against its own `PortfolioHolding` rows, which is
    exactly why this stays a thin, unopinionated {symbol, qty} list: any
    comparison logic (tolerances, which symbols even apply) belongs in
    the auditing script, not baked into this service's own response
    shape.
    """
    try:
        client = broker.get_trading_client()
        raw_positions = await asyncio.to_thread(client.get_all_positions)
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch paper positions")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return [{"symbol": p.symbol, "qty": str(p.qty)} for p in raw_positions]


#: How far an inbound /control/halt request's timestamp may be from this
#: server's clock, in seconds — same window and same reasoning as PFW's
#: own inbound verification (src/lib/webhook-signature.ts's
#: REPLAY_WINDOW_SECONDS): binding the timestamp into the MAC only makes
#: a captured request un-replayable if a stale one is actually rejected.
_CONTROL_REPLAY_WINDOW_SECONDS = 300


def _telemetry_for_recovered_fill(order: Any) -> None:
    """Surface a reconciliation recovery on the Agent Activity page like a live settlement."""
    _log_event(
        "settle",
        ticker=str(order.symbol),
        status="settled (reconciled)",
        details=f"Recovered fill {order.filled_qty} @ {order.filled_avg_price} (order {order.id}, via reconciliation)",
    )


async def _verify_control_request(request: Request) -> bytes:
    """The shared trust boundary for every ``/control/*`` endpoint: an
    HMAC-SHA256 signature over the raw request body, the exact same
    scheme (and the exact same shared ``WEBHOOK_SECRET``) this service
    already uses to SIGN its own outbound receipts to PFW — verified here
    in the reverse direction. The caller is PFW's own server, which signs
    after confirming the browser holds an authenticated PFW session; the
    browser itself never sees ``WEBHOOK_SECRET``, since embedding a shared
    HMAC secret in client-side JavaScript would hand anyone who opens dev
    tools the ability to forge trade receipts and settlements too. Raises
    the appropriate ``HTTPException``; returns the verified raw body.
    """
    raw_body = await request.body()
    # Starlette's Headers is case-insensitive by construction, matching
    # HTTP's own header-name semantics — no need to normalize case here.
    timestamp = request.headers.get(webhook.TIMESTAMP_HEADER)
    signature = request.headers.get(webhook.SIGNATURE_HEADER)

    try:
        settings = get_webhook_settings()
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    if not timestamp or not signature:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing signature headers")
    try:
        timestamp_seconds = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid timestamp") from exc
    if abs(time.time() - timestamp_seconds) > _CONTROL_REPLAY_WINDOW_SECONDS:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Stale signature")

    if not webhook.verify(raw_body, timestamp, signature, settings.secret):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid signature")
    return raw_body


class ReconcileRequest(BaseModel):
    lookback_hours: int = Field(default=24, ge=1, le=24 * 30)


@app.post("/control/reconcile", tags=["control"])
async def reconcile(request: Request) -> dict[str, Any]:
    """Re-send the settlement receipt for every fill Alpaca reports in the
    lookback window (see ``reconcile.py``). Idempotent on the PFW side, so
    safe to call at any time; same HMAC trust boundary as ``/control/halt``.
    """
    raw_body = await _verify_control_request(request)
    try:
        params = ReconcileRequest.model_validate_json(raw_body or b"{}")
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors()) from exc
    stats = await reconcile_recent_fills(
        lookback_hours=params.lookback_hours, on_recovered=_telemetry_for_recovered_fill,
    )
    return {"ok": stats.error is None, **stats.__dict__}


class QuotesRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=200)


@app.post("/control/quotes", tags=["control"])
async def latest_quotes(request: Request) -> dict[str, Any]:
    """Alpaca's latest IEX trade price for each requested symbol — the
    mark PFW values a holding at when the ticker is one this agent booked
    from a real fill rather than one of PFW's own seeded instruments (its
    mock feed throws on anything else, which took its dashboard down for
    the trading account once). PFW's server calls this from its daily
    sync, never the browser; same HMAC trust boundary as ``/control/halt``
    because the Alpaca credentials live here and only here. Symbols
    Alpaca has no trade for come back under ``missing`` rather than
    failing the batch.
    """
    raw_body = await _verify_control_request(request)
    try:
        params = QuotesRequest.model_validate_json(raw_body or b"{}")
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors()) from exc
    try:
        prices = await asyncio.to_thread(broker.get_latest_prices, params.symbols)
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — a market-data outage is a 502 for the caller, never a crash here
        logger.warning("quotes: Alpaca market-data request failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Market data unavailable: {type(exc).__name__}") from exc
    wanted = sorted({symbol.strip().upper() for symbol in params.symbols if symbol and symbol.strip()})
    return {
        "quotes": {
            symbol: {"price": str(price.price), "timestamp": price.timestamp.isoformat()}
            for symbol, price in prices.items()
        },
        "missing": [symbol for symbol in wanted if symbol not in prices],
        "feed": "iex",
    }


@app.post("/control/halt", tags=["control"])
async def halt(request: Request) -> dict[str, Any]:
    """Emergency kill switch.

    Trust boundary is an HMAC-SHA256 signature over the raw request body
    (the exact same scheme, and the exact same shared ``WEBHOOK_SECRET``,
    this service already uses to SIGN its own outbound receipts to PFW —
    verified here in the reverse direction instead). The caller is
    PFW's own server (``POST /api/agent/halt``), which signs the request
    itself after confirming the browser holds an authenticated PFW
    session — the browser itself never sees ``WEBHOOK_SECRET`` at any
    point, since embedding a shared HMAC secret in client-side JavaScript
    would hand anyone who opens dev tools the ability to forge trade
    receipts and settlements too, not just halt requests.

    Sets ``scheduler.IS_HALTED`` and cancels every open Alpaca order.
    Deliberately no "resume" counterpart: an emergency halt should
    require a deliberate restart of this process to clear, not an easy
    accidental toggle back on.
    """
    await _verify_control_request(request)

    trigger_emergency_halt("Emergency halt triggered via POST /control/halt")

    try:
        client = broker.get_trading_client()
        cancel_statuses = await asyncio.to_thread(client.cancel_orders)
        canceled_count = len(cancel_statuses) if cancel_statuses else 0
    except Exception as exc:
        logger.exception("Emergency halt: failed to cancel open orders")
        return {"halted": True, "orders_canceled": 0, "cancel_error": str(exc)}

    logger.critical("Emergency halt: %d open order(s) canceled", canceled_count)
    return {"halted": True, "orders_canceled": canceled_count}


class TransactionAnalysisRequest(BaseModel):
    """One transaction to score for cash-flow anomaly (Phase 3, ad hoc).

    ``occurred_at`` should be timezone-aware; the hour fed into the
    autoencoder is derived from it directly rather than accepted as a
    separate raw-integer field, since a caller-supplied bare hour with no
    timezone context is genuinely ambiguous in a way a full timestamp
    isn't. ``amount`` is the transaction's absolute value in the
    currency's MAJOR unit (dollars/shekels, not cents/agorot) — this
    service has no currency-conversion concerns of its own here, unlike
    its trading side; the caller's own currency is whatever it already
    reports amounts in.
    """

    transaction_id: str = Field(
        ..., min_length=1, description="Caller's own id, echoed back for correlation only — never looked up here."
    )
    amount: float
    category: str = Field(..., min_length=1, max_length=80)
    occurred_at: datetime


@app.post("/analyze/transaction", tags=["analytics"])
async def analyze_transaction(request: Request) -> dict[str, Any]:
    """Cash-flow anomaly check for one transaction (Phase 3, ad hoc).

    Same inbound HMAC trust boundary as ``/control/halt`` above —
    identical shared ``WEBHOOK_SECRET``, identical replay window, no new
    secret needed. The expected caller is PFW's own server, signing this
    request the same way it signs the halt request, after it has already
    resolved which user's transaction this is; this endpoint never sees a
    PFW user id or session, only the transaction fields it needs to score
    (see ``TransactionAnalysisRequest``) — it has no way to look up
    anything about the caller's account even if it wanted to.

    Runs the transaction through ``models.autoencoder``'s trained
    checkpoint (see that module and ``train_autoencoder.py`` for the
    model itself and how its Z-score threshold was derived) and returns
    whether its reconstruction error clears that threshold.
    """
    raw_body = await request.body()
    timestamp = request.headers.get(webhook.TIMESTAMP_HEADER)
    signature = request.headers.get(webhook.SIGNATURE_HEADER)

    try:
        settings = get_webhook_settings()
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    if not timestamp or not signature:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing signature headers")

    try:
        timestamp_seconds = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid timestamp") from exc

    if abs(time.time() - timestamp_seconds) > _CONTROL_REPLAY_WINDOW_SECONDS:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Stale signature")

    if not webhook.verify(raw_body, timestamp, signature, settings.secret):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid signature")

    try:
        payload = TransactionAnalysisRequest.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc.errors())) from exc

    try:
        result = autoencoder.score_transaction(payload.amount, payload.category, payload.occurred_at.hour)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return {
        "transaction_id": payload.transaction_id,
        "is_anomaly": result.is_anomaly,
        "reconstruction_error": result.reconstruction_error,
        "z_score": result.z_score,
        "threshold": result.threshold,
    }


async def _resolve_ticker_and_headline(request: SignalRequest) -> tuple[str, str]:
    """Resolve a request's (ticker, headline), falling back to a random
    ``inference.MARKET_SCENARIOS`` pick when no ticker was supplied at all.
    """
    if request.ticker is None:
        return inference.pick_random_scenario()
    headline = request.headline or await inference.fetch_latest_headline(request.ticker)
    return request.ticker, headline


@app.post("/signals/evaluate", tags=["trading"])
async def evaluate(request: SignalRequest) -> dict[str, Any]:
    """Dry run: ingest, infer and apply the Tier-0 gates. Submits nothing.

    Use this to inspect what the engine *would* do. It needs no credentials.
    Runs through the SAME ``scheduler.run_signal_cycle`` pipeline as the
    autonomous loop (``submit=False``), so the evaluation shows up in the
    telemetry feed and in PFW's ``ScenarioMetrics`` like any other —
    previously a manual call was invisible to both.
    """
    ticker, headline = await _resolve_ticker_and_headline(request)
    return await run_signal_cycle(ticker, headline, submit=False)


@app.post("/signals/execute", tags=["trading"])
async def execute(request: SignalRequest) -> dict[str, Any]:
    """Full pipeline: ingest -> infer -> Tier-0 gate -> paper order -> receipt.

    A rejected signal is a normal 200 response with ``executed: false`` — the
    engine declining to trade is an expected outcome, not an error. Same
    ``scheduler.run_signal_cycle`` pipeline as the autonomous loop, so the
    telemetry/metrics side effects are identical to an autonomous cycle.
    """
    ticker, headline = await _resolve_ticker_and_headline(request)
    try:
        return await run_signal_cycle(ticker, headline, submit=True)
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        logger.exception("Order submission failed for %s", ticker)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
