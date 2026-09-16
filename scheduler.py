"""Autonomous background trading loop, and the settlement loop that
confirms what it actually did.

``trading_loop`` runs the exact same pipeline ``/signals/execute`` runs —
ingest (a random scenario) -> ``inference.predict_move`` ->
``execution.validate_and_plan`` -> Alpaca paper submit -> "pending"
receipt -> PFW — on an infinite loop, with a randomized 60-300s jitter
between iterations rather than a fixed interval. A uniform cron-like
cadence is exactly the kind of synthetic signal a "natural trading
activity" simulation is meant to avoid. Gated on ``AUTONOMOUS_MODE``
(``config.is_autonomous_mode_enabled()``).

``websocket_settlement_stream`` runs independently, UNCONDITIONALLY (not
gated on AUTONOMOUS_MODE — see its own docstring), subscribing to
Alpaca's real-time ``trade_updates`` WebSocket channel and forwarding a
"settled" receipt the moment a real fill event arrives, carrying the
real fill price. Two-phase settlement (ad hoc, extends the original
single-shot design) exists because a submitted order and an actual fill
are genuinely different events — verified live: every paper order this
agent submitted before this fix was priced against a synthetic quote
wildly divorced from the real market and never filled at all, while
PFW's ledger had already booked each one as a completed trade the
moment Alpaca merely ACCEPTED it.

Replaces an earlier 30s REST-polling ``settlement_loop`` — push-based
streaming means a fill is settled within moments of actually happening,
not up to 30s later, and drops the periodic ``GetOrdersRequest`` call
entirely. The trade-off, stated plainly rather than glossed over: the
old poller was self-healing by construction (every cycle re-scanned all
recently-closed orders, so a transient PFW outage just meant "try again
next cycle" for free) — a push stream has no such natural retry. A fill
event that arrives while PFW is down beyond ``webhook.py``'s own
in-request retry budget (3 attempts, ~3.5s total) is not retried again
by anything in this file. Acceptable for this project's own paper-
trading scale, but worth knowing before relying on this for anything
where a missed settlement would be costly.

Both loops are started from ``main.py``'s ``lifespan`` — this module
never starts itself and has no ``if __name__ == "__main__"`` entry point.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final, Literal

from alpaca.trading.enums import TradeEvent
from alpaca.trading.models import TradeUpdate

import broker
import execution
import inference
import webhook

logger = logging.getLogger("tier0.scheduler")

#: Spacing between consecutive autonomous executions, in seconds.
_MIN_INTERVAL_SECONDS: Final[int] = 60
_MAX_INTERVAL_SECONDS: Final[int] = 300

#: Sleep duration once the circuit breaker trips, in seconds — long
#: enough to stop hammering the account-equity endpoint and the
#: telemetry feed with a repeat alert every 1-5 minutes, short enough
#: that the loop still naturally re-checks well within the same trading
#: day rather than needing a restart to recover.
_CIRCUIT_BREAKER_COOLDOWN_SECONDS: Final[int] = 3600

TelemetryAction = Literal[
    "wake", "evaluate", "reject", "execute", "settle", "sleep", "error", "critical_alert",
]

#: Bounded in-memory ring buffer of the last 50 agent events, read by
#: `GET /telemetry` in main.py. Deliberately module-level, process-local
#: state — no DB, no queue: this is a live "what is the agent doing right
#: now" activity feed for one running process, not a durable record.
#: PFW's own signed receipts and its LedgerCommit hash chain (AGENTS.md
#: §3mm) already are the durable, tamper-evident record of what actually
#: executed; this resets to empty on every process restart, on purpose.
agent_telemetry: Final[deque[dict[str, Any]]] = deque(maxlen=50)


def _log_event(action: TelemetryAction, *, ticker: str | None, status: str, details: str) -> None:
    """Append one telemetry event. Timestamped in UTC ISO-8601 so the
    Next.js dashboard doesn't have to guess this process's local timezone.
    """
    agent_telemetry.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "ticker": ticker,
            "status": status,
            "details": details,
        }
    )


async def run_signal_cycle(ticker: str, headline: str, *, submit: bool) -> dict[str, Any]:
    """One full ingest -> infer -> Tier-0 gate -> (paper order ->) receipt
    pass for ONE scenario, with telemetry and scenario-metrics emitted
    along the way.

    The single pipeline behind three callers (trader integration
    hardening, ad hoc): the autonomous loop (``_run_one_cycle``, a random
    scenario, ``submit=True``), ``POST /signals/execute`` (a caller-supplied
    or random scenario, ``submit=True``) and ``POST /signals/evaluate``
    (``submit=False`` — the Tier-0 gate runs, nothing is sent to the
    broker). Before this, ``main.py``'s two endpoints duplicated the
    body but skipped ``_log_event``/``send_scenario_metrics`` entirely,
    so a manual call was invisible on PFW's Agent Activity page and never
    reached its ``ScenarioMetrics`` table — with ``AUTONOMOUS_MODE`` off
    the page stayed empty forever, however much the trader was used.

    Returns the same dict ``/signals/execute`` has always returned
    (``executed``/``plan``/``order``/``signal``/``receipt_delivery``, or
    ``executed: False`` + ``rejection``); a dry run returns
    ``approved``/``plan`` (or ``rejection``) plus ``quote`` instead.

    Shadow A/B pipeline (Phase 3, ad hoc): ``inference._shadow_evaluate``
    runs CONCURRENTLY with the primary ``inference.predict_move`` (and
    ``inference.fetch_quote``, equally independent) via ``asyncio.gather``
    — genuinely at the same time, not sequentially awaited one after the
    other. The shadow verdict is persisted alongside the primary one in
    the SAME ``send_scenario_metrics`` call below, for offline A/B
    backtesting only. It is deliberately NEVER passed to
    ``execution.execute_signal`` below, and never referenced by any
    ``_log_event``/telemetry call either — only the primary Tier-0
    model's own ``signal`` decides what happens at the broker; the
    shadow model has no live effect of any kind.
    """
    quote, signal, shadow_signal = await asyncio.gather(
        inference.fetch_quote(ticker),
        inference.predict_move(ticker, headline),
        inference._shadow_evaluate(ticker, headline),
    )

    _log_event(
        "evaluate",
        ticker=ticker,
        status="evaluating" if submit else "evaluating (dry run)",
        details=f"Scored {ticker} at {signal.predicted_move_pct:+.2f}% ({headline!r})",
    )

    result: dict[str, Any]
    if submit:
        result = await execution.execute_signal(signal, quote)
    else:
        result = _dry_run(signal, quote)

    if result.get("executed"):
        plan = result["plan"]
        order = result["order"]
        decision = "executed"
        logger.info(
            "Cycle EXECUTED %s qty=%s @ %s (order %s, status=%s)",
            plan["symbol"], plan["quantity"], plan["limit_price"],
            order["id"], order["status"],
        )
        _log_event(
            "execute",
            ticker=plan["symbol"],
            status="executed",
            details=(
                f"qty={plan['quantity']} @ {plan['limit_price']} "
                f"(order {order['id']}, {order['status']})"
            ),
        )
    elif result.get("approved"):
        decision = "approved_dry_run"
        logger.info("Dry run APPROVED %s: %s", ticker, result["plan"])
        _log_event(
            "evaluate",
            ticker=ticker,
            status="approved (dry run)",
            details=f"Tier-0 gate cleared for {ticker}; nothing submitted (dry run)",
        )
    else:
        rejection = result["rejection"]
        decision = rejection["code"]
        logger.info(
            "Cycle REJECTED %s: %s (%s)",
            ticker, rejection["code"], rejection["detail"],
        )
        _log_event(
            "reject",
            ticker=ticker,
            status="rejected" if submit else "rejected (dry run)",
            details=f"{rejection['code']}: {rejection['detail']}",
        )

    metrics_delivery = await webhook.send_scenario_metrics(
        ticker=ticker,
        headline=headline,
        predicted_move_pct=signal.predicted_move_pct,
        decision=decision,
        shadow_predicted_move_pct=shadow_signal.predicted_move_pct,
        shadow_decision=shadow_signal.decision,
    )
    if not metrics_delivery.get("delivered"):
        logger.warning(
            "Scenario metrics for %s not delivered: %s%s", ticker, metrics_delivery.get("error"),
            " (queued for replay)" if metrics_delivery.get("queued") else "",
        )
    return result


def _dry_run(signal: Any, quote: Any) -> dict[str, Any]:
    """The Tier-0 gate without the broker — ``/signals/evaluate``'s original body."""
    payload: dict[str, Any] = {
        "executed": False,
        "signal": signal.model_dump(mode="json"),
        "quote": {
            "bid": str(quote.bid),
            "ask": str(quote.ask),
            "as_of": quote.as_of.isoformat(),
        },
    }
    try:
        plan = execution.validate_and_plan(signal, quote)
    except execution.Tier0Rejection as rejection:
        return payload | {
            "approved": False,
            "rejection": {"code": rejection.code.value, "detail": rejection.detail},
        }
    return payload | {"approved": True, "plan": plan.as_dict()}


async def _run_one_cycle() -> None:
    """One autonomous pass: a randomly picked market scenario through
    ``run_signal_cycle`` — the same ``inference.pick_random_scenario()``
    the ``/signals/*`` endpoints fall back to when a request omits
    ``ticker``.
    """
    ticker, headline = inference.pick_random_scenario()
    await run_signal_cycle(ticker, headline, submit=True)


IS_HALTED: bool = False


def trigger_emergency_halt(reason: str) -> None:
    """Set the kill switch and record a critical_alert telemetry event.

    Called ONLY by main.py's POST /control/halt handler, and ONLY after
    that handler has independently verified the inbound request's HMAC
    signature — this function itself trusts its caller completely, which
    is exactly why it is never itself exposed as an HTTP-reachable
    action. Order cancellation is the caller's responsibility (it needs
    the broker client and its own error handling for a partial-cancel
    outcome), not this function's — this one only ever flips the switch
    and records that it happened.
    """
    global IS_HALTED
    IS_HALTED = True
    logger.critical("EMERGENCY HALT: %s", reason)
    _log_event("critical_alert", ticker=None, status="halted", details=reason)


async def _fetch_daily_pnl_pct() -> Decimal | None:
    """Fetch today's account P&L%, for the circuit breaker check below.

    Returns ``None`` (rather than raising) on any failure — broker not
    configured, a transient Alpaca API error, ``get_daily_pnl_pct``'s own
    ``ValueError`` guards. Deliberately does NOT itself trip the circuit
    breaker: conflating "couldn't check" with "breached" would be a
    different, unrequested design decision, and a genuinely unconfigured
    or unreachable broker is already handled by ``_run_one_cycle``'s own
    existing error path, which the caller falls through to when this
    returns ``None``.
    """
    try:
        return await asyncio.to_thread(broker.get_daily_pnl_pct)
    except Exception as exc:
        logger.warning("Could not fetch daily P&L for circuit breaker check: %s", exc)
        return None


async def trading_loop() -> None:
    """Runs forever until cancelled, executing one cycle per iteration.

    Checks the hard portfolio circuit breaker
    (``execution.CIRCUIT_BREAKER_DAILY_PNL_PCT``) at the very top of
    every iteration, BEFORE inference runs at all. Once today's account
    P&L is at or below that threshold, the loop logs a critical alert,
    skips inference entirely for this iteration, and sleeps for
    ``_CIRCUIT_BREAKER_COOLDOWN_SECONDS`` (a prolonged cooldown, not the
    usual 1-5 minute jitter) before re-checking — it keeps re-tripping
    and re-alerting on every subsequent hourly check for as long as the
    breach persists, which is deliberate: a standing risk breach
    genuinely warrants a standing alert, not a one-time notice.

    Checked BEFORE even that: the emergency kill switch (``IS_HALTED``,
    set only via ``trigger_emergency_halt`` from main.py's
    ``POST /control/halt``, itself HMAC-verified). Once halted, this loop
    polls every second — no jitter, no telemetry spam per poll (the
    single ``critical_alert`` was already recorded by
    ``trigger_emergency_halt`` at the moment of the halt itself) — and
    submits nothing further until the process is restarted. Deliberately
    checked ahead of the circuit breaker too: a halt is a manual,
    higher-priority override that shouldn't wait on an extra Alpaca
    account-equity fetch to take effect.

    A single cycle's failure (a broker outage, a misconfigured
    credential, a transient network error) is logged — and recorded to
    telemetry as an "error" event, so a failure is visible on the
    dashboard instead of just leaving an unexplained gap — and the loop
    continues; one bad tick must not silently kill the whole autonomous
    runner. ``asyncio.CancelledError`` is deliberately NOT swallowed: it
    has to propagate so the ``lifespan`` shutdown handler in ``main.py``
    can observe the task actually stopped, rather than the loop silently
    ignoring its own cancellation and outliving the server.
    """
    logger.info(
        "Autonomous trading_loop starting (jitter %s-%ss between cycles)",
        _MIN_INTERVAL_SECONDS, _MAX_INTERVAL_SECONDS,
    )
    while True:
        if IS_HALTED:
            await asyncio.sleep(1)
            continue

        _log_event("wake", ticker=None, status="waking", details="Autonomous cycle waking up")

        pnl_pct = await _fetch_daily_pnl_pct()
        if pnl_pct is not None and pnl_pct <= execution.CIRCUIT_BREAKER_DAILY_PNL_PCT:
            logger.critical(
                "CIRCUIT BREAKER TRIPPED: daily P&L %s%% is at/below the hard %s%% limit "
                "— skipping inference, cooling down for %ss",
                pnl_pct, execution.CIRCUIT_BREAKER_DAILY_PNL_PCT, _CIRCUIT_BREAKER_COOLDOWN_SECONDS,
            )
            _log_event(
                "critical_alert",
                ticker=None,
                status="circuit_breaker",
                details=(
                    f"Daily P&L {pnl_pct:.2f}% breached the "
                    f"{execution.CIRCUIT_BREAKER_DAILY_PNL_PCT}% circuit breaker — "
                    f"trading paused for {_CIRCUIT_BREAKER_COOLDOWN_SECONDS}s"
                ),
            )
            await asyncio.sleep(_CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            continue

        try:
            await _run_one_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Autonomous cycle failed; continuing after the usual jitter")
            _log_event("error", ticker=None, status="error", details=str(exc))

        interval = random.randint(_MIN_INTERVAL_SECONDS, _MAX_INTERVAL_SECONDS)
        _log_event(
            "sleep",
            ticker=None,
            status="sleeping",
            details=f"Sleeping {interval}s before the next cycle",
        )
        await asyncio.sleep(interval)


#: In-memory "already forwarded/resolved" ring buffer, bounded so this
#: process's memory footprint can't grow without limit over a long
#: uptime. A courtesy against a duplicate WebSocket delivery — Alpaca's
#: own docs don't guarantee exactly-once delivery on reconnect — NOT the
#: actual correctness backstop. PFW's own idempotency dedup on
#: Trade.idempotencyKey (webhook.py's own doc comment: "retries are
#: safe... idempotency_key lets the receiver dedupe") is what makes a
#: redundant resend harmless even if this buffer were empty (it always
#: is on every process restart), or if an old order id ever aged out
#: past the 1000-entry cap.
_synced_order_ids: Final[deque[str]] = deque(maxlen=1000)


async def _handle_trade_update(data: TradeUpdate) -> None:
    """Handle one ``trade_updates`` WebSocket event.

    Alpaca's own event taxonomy (``alpaca.trading.enums.TradeEvent`` —
    new/accepted/fill/partial_fill/canceled/rejected/expired/...) — only
    ``FILL`` is a complete, final execution worth settling. Every other
    event, ``PARTIAL_FILL`` included, is silently ignored here ON
    PURPOSE, not mishandled: this project's own orders are always DAY
    orders sized to fill in one shot at the Tier-0 notional cap, so a
    partial fill simply hasn't reached ``FILL`` yet — it will, as its own
    later event, or the order will otherwise resolve (cancel/expire)
    with nothing here needing to react to the partial event itself.
    """
    if data.event != TradeEvent.FILL:
        return

    order = data.order
    order_id = str(order.id)
    if order_id in _synced_order_ids:
        return
    if not order.filled_avg_price or not order.filled_qty:
        return

    delivery = await webhook.send_settlement_receipt(order)
    if delivery.get("delivered"):
        _synced_order_ids.append(order_id)
        logger.info(
            "Settled %s qty=%s @ %s (order %s) via WebSocket",
            order.symbol, order.filled_qty, order.filled_avg_price, order_id,
        )
        _log_event(
            "settle",
            ticker=order.symbol,
            status="settled",
            details=f"Filled {order.filled_qty} @ {order.filled_avg_price} (order {order_id}, via WebSocket)",
        )
    elif str(delivery.get("error", "")).startswith("HTTP 4"):
        # Poison pill: PFW rejected this receipt outright (a contract
        # problem — e.g. insufficient_shares on a hypothetical SELL —
        # not a transient outage), the same "4xx will not fix itself"
        # rule webhook._deliver's own retry loop already applies within one
        # delivery attempt. Without this, a permanently-rejected
        # settlement would otherwise just sit undelivered forever, since
        # a push stream never redelivers a past event on its own.
        _synced_order_ids.append(order_id)
        logger.error(
            "Settlement for order %s permanently rejected by PFW, giving up: %s",
            order_id, delivery.get("error"),
        )
        _log_event(
            "error",
            ticker=order.symbol,
            status="error",
            details=f"Settlement permanently rejected (order {order_id}): {delivery.get('error')}",
        )
    else:
        # A transient failure. webhook._deliver already retried 3x inline
        # and has now handed the receipt to the durable outbox, whose
        # replay loop re-signs and re-POSTs it with backoff until PFW
        # acknowledges it — so, unlike before, this fill is NOT lost just
        # because PFW was down for the few seconds around it. Marked
        # synced here because the outbox now owns delivery; the push
        # stream itself never redelivers a past event.
        _synced_order_ids.append(order_id)
        logger.warning(
            "Settlement receipt for order %s not delivered inline; queued in the outbox for replay: %s",
            order_id, delivery.get("error"),
        )
        _log_event(
            "error",
            ticker=order.symbol,
            status="queued",
            details=f"Settlement receipt queued for replay (order {order_id}): {delivery.get('error')}",
        )


async def websocket_settlement_stream() -> None:
    """Runs forever until cancelled, streaming Alpaca's real-time
    ``trade_updates`` WebSocket and settling each genuine fill as it
    arrives.

    Runs UNCONDITIONALLY — unlike ``trading_loop``, this is NOT gated on
    ``AUTONOMOUS_MODE``. A fill can happen for an order this agent
    submitted manually (a direct ``/signals/execute`` call, with the
    autonomous loop otherwise off) just as easily as for one the
    autonomous loop itself created, and both need their pending Trade
    settled the same way — there is nothing autonomy-specific about
    noticing that a real fill occurred.

    Deliberately awaits alpaca-py's own ``TradingStream._run_forever()``
    directly, rather than its public ``run()`` — confirmed by reading
    alpaca-py 0.44.0's actual source before writing this, not assumed:
    ``run()`` wraps itself in a brand-new ``asyncio.run(...)`` call,
    which is right for a standalone script but cannot be awaited from
    inside an already-running event loop (this service's own). Awaiting
    ``_run_forever()`` directly runs the stream in THIS process's
    existing loop instead, which is also what makes a plain
    ``task.cancel()`` — the EXACT same shutdown mechanism ``trading_loop``
    already uses, no new stop-signaling primitive needed — fully
    sufficient here: ``asyncio.CancelledError`` (a ``BaseException``, not
    an ``Exception`` — verified, not assumed) raised into the suspended
    ``await`` below is re-raised immediately by the ``except
    asyncio.CancelledError: raise`` branch, past the reconnect logic
    entirely, exactly mirroring how ``trading_loop`` itself shuts down.
    This is a deliberate, documented reliance on a private method, the
    same trade-off ``broker.get_trading_stream()`` already accepts for
    reading ``_endpoint`` — alpaca-py exposes no public alternative for
    embedding this class in an existing loop.

    Heartbeat and reconnect are NOT hand-rolled here — verified by
    reading the source: ``TradingStream.__init__`` already sets
    ``ping_interval=10``/``ping_timeout=180`` on the underlying
    ``websockets`` connection (protocol-level heartbeat), and
    ``_run_forever`` already retries a dropped connection with genuine
    exponential backoff (1s-30s, via its own ``_wait_before_reconnect``)
    on any ``websockets.WebSocketException``. Re-implementing either
    would only duplicate logic the SDK already gets right. The outer
    ``while`` loop below is a SEPARATE, outer layer of backoff — its own
    exponential delay, capped the same way — for the one case the SDK's
    internal loop does NOT cover: ``_run_forever()`` itself returning or
    raising past its own retry logic entirely (e.g. a bug, or every
    retry exhausted in some way its own code doesn't loop past).
    """
    stream = broker.get_trading_stream()
    stream.subscribe_trade_updates(_handle_trade_update)

    retries = 0
    while True:
        logger.info("Starting Alpaca trade_updates WebSocket stream")
        try:
            await stream._run_forever()  # noqa: SLF001 — see docstring
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("WebSocket settlement stream crashed")
            _log_event("error", ticker=None, status="error", details=str(exc))
        finally:
            await stream.close()

        retries += 1
        delay = min(2**retries, 30)
        logger.warning(
            "trade_updates WebSocket stream exited unexpectedly; reconnecting in %ss (attempt %d)",
            delay, retries,
        )
        await asyncio.sleep(delay)
