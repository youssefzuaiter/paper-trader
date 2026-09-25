"""Phase 3 — Tier-0 deterministic execution engine (the monolith's order path).

This module sits between the model and the broker, and it is deliberately
boring: fixed thresholds, integer and ``Decimal`` arithmetic, no model input
reaching the sizing math except as a pass/fail on a single scalar. The Risk &
Routing agent (``risk_router/``) is the other, and eventually the only, path
that can create an order; both size through ``tier0.plan_buy``.

Every risk constant is a hardcoded ``Final`` in ``tier0.py``. They are *not*
read from ``.env``, *not* function parameters, and *not* reachable from any
request body. A limit that a caller can widen is not a limit.

Money never touches ``float``. Sizing is ``Decimal`` with explicit rounding
modes; the cast to ``float`` happens once, at the Alpaca API boundary.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest

import webhook
from broker import get_trading_client
from inference import InferenceSignal, Quote

# The limits and the sizing arithmetic live in tier0.py, shared with the
# Risk & Routing agent. Re-exported here so every existing
# ``execution.<NAME>`` reference keeps resolving to the same object.
from tier0 import (  # noqa: F401 — re-exports
    CIRCUIT_BREAKER_DAILY_PNL_PCT,
    LIMIT_BUFFER_PCT,
    MAX_NOTIONAL_USD,
    MIN_PREDICTED_GAIN_PCT,
    RISK_BUDGET_USD,
    STOP_LOSS_PCT,
    ExecutionPlan,
    RejectionCode,
    StopLossKind,
    Tier0Rejection,
    plan_buy,
)

logger = logging.getLogger(__name__)


def validate_and_plan(signal: InferenceSignal, quote: Quote) -> ExecutionPlan:
    """Apply the Tier-0 gates and size the order.

    Gates 0-1 (finite signal, minimum predicted gain) are this module's
    entry rule; pricing, sizing and the cap post-condition are
    ``tier0.plan_buy``, shared with the Risk & Routing agent.

    Args:
        signal: Model output. Only ``predicted_move_pct`` influences the gate.
        quote: Quote used for pricing.

    Returns:
        A priced :class:`ExecutionPlan`.

    Raises:
        Tier0Rejection: If any hardcoded gate fails.
    """
    # --- Gate 0: the signal must be a finite number ------------------------
    predicted = signal.predicted_move_pct
    if predicted != predicted or predicted in (float("inf"), float("-inf")):
        raise Tier0Rejection(
            RejectionCode.NON_FINITE_SIGNAL,
            f"predicted_move_pct is not finite: {predicted!r}",
        )
    predicted_dec = Decimal(str(predicted))

    # --- Gate 1: minimum predicted gain ------------------------------------
    if predicted_dec < MIN_PREDICTED_GAIN_PCT:
        raise Tier0Rejection(
            RejectionCode.BELOW_MIN_GAIN,
            f"predicted gain {predicted_dec}% is below the "
            f"{MIN_PREDICTED_GAIN_PCT}% minimum",
        )

    # --- Gates 2-4: quote sanity, sizing, cap -------------------------------
    return plan_buy(symbol=quote.ticker, bid=quote.bid, ask=quote.ask, atr=quote.atr)


def build_order_request(plan: ExecutionPlan) -> LimitOrderRequest:
    """Translate a validated plan into an Alpaca ``LimitOrderRequest``.

    ``TimeInForce.DAY`` is mandatory: Alpaca supports no other TIF on
    fractional quantities.
    """
    kwargs: dict[str, object] = {
        "symbol": plan.symbol,
        "qty": float(plan.quantity),  # single float cast, at the API boundary
        "side": OrderSide(plan.side.value),  # tier0's enum → alpaca-py's
        "time_in_force": TimeInForce.DAY,
        "limit_price": float(plan.limit_price),
        "client_order_id": plan.client_order_id,
    }

    if plan.stop_loss_kind is StopLossKind.NATIVE_OTO:
        kwargs["order_class"] = OrderClass.OTO
        kwargs["stop_loss"] = StopLossRequest(stop_price=float(plan.stop_price))

    return LimitOrderRequest(**kwargs)  # type: ignore[arg-type]


async def execute_signal(signal: InferenceSignal, quote: Quote) -> dict:
    """Validate, submit to the Alpaca paper API, and emit a signed receipt.

    Returns:
        A JSON-serialisable outcome dict. On rejection, ``executed`` is
        ``False`` and ``rejection`` explains which gate failed.

    Raises:
        Exception: Broker transport errors propagate — a failed submit must be
            visible, not swallowed. Webhook failures do *not* propagate: the
            trade already happened, so delivery is reported, not raised.
    """
    try:
        plan = validate_and_plan(signal, quote)
    except Tier0Rejection as rejection:
        logger.info(
            "Tier-0 REJECT %s: %s", rejection.code.value, rejection.detail
        )
        return {
            "executed": False,
            "rejection": {"code": rejection.code.value, "detail": rejection.detail},
            "signal": signal.model_dump(mode="json"),
        }

    logger.info(
        "Tier-0 ACCEPT %s qty=%s @ %s (notional $%s, stop %s / %s)",
        plan.symbol, plan.quantity, plan.limit_price,
        plan.notional_usd, plan.stop_price, plan.stop_loss_kind.value,
    )
    if plan.stop_loss_kind is StopLossKind.ENGINE_TRACKED:
        logger.warning(
            "%s sized fractionally (%s); Alpaca will not hold a native stop. "
            "Stop %s is engine-tracked and carried on the receipt.",
            plan.symbol, plan.quantity, plan.stop_price,
        )

    # alpaca-py's REST client is synchronous (requests); awaiting it inline
    # would stall every other request and background task for the whole
    # round-trip to Alpaca. A worker thread keeps the event loop free.
    order = await asyncio.to_thread(get_trading_client().submit_order, build_order_request(plan))
    submitted_at = datetime.now(UTC)
    logger.info("Submitted paper order %s (status=%s)", order.id, order.status)

    delivery = await webhook.send_trade_receipt(
        plan=plan, signal=signal, order=order, executed_at=submitted_at
    )

    return {
        "executed": True,
        "plan": plan.as_dict(),
        "order": {
            "id": str(order.id),
            "client_order_id": order.client_order_id,
            "status": str(order.status),
            "submitted_at": submitted_at.isoformat(),
        },
        "signal": signal.model_dump(mode="json"),
        "receipt_delivery": delivery,
    }
