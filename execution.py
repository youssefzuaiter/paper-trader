"""Phase 3 — Tier-0 deterministic execution engine.

This module is the only place that can create an order. It sits between the
model and the broker, and it is deliberately boring: fixed thresholds, integer
and ``Decimal`` arithmetic, no model input reaching the sizing math except as a
pass/fail on a single scalar.

Every risk constant below is a hardcoded module ``Final``. They are *not* read
from ``.env``, *not* function parameters, and *not* reachable from any request
body. A limit that a caller can widen is not a limit.

Money never touches ``float``. Sizing is ``Decimal`` with explicit rounding
modes; the cast to ``float`` happens once, at the Alpaca API boundary.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from enum import StrEnum
from typing import Final

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest

import webhook
from broker import get_trading_client
from inference import InferenceSignal, Quote

logger = logging.getLogger(__name__)

# ===========================================================================
# TIER-0 HARDCODED LIMITS — do not parameterise, do not move to .env
# ===========================================================================

#: Reject any trade whose predicted gain is below this, in percent.
MIN_PREDICTED_GAIN_PCT: Final[Decimal] = Decimal("10")

#: Hard cap on capital exposure per order, in USD. Fractional sizing exists
#: precisely so this cap binds on expensive symbols instead of rejecting them.
#: NEVER WIDENED by the volatility-parity sizing below — quantity is
#: always the SMALLER of this notional-based figure and the ATR-based
#: one, by construction (``min(...)``), so this remains the absolute
#: ceiling on capital exposure regardless of how volatility sizing
#: changes in the future. "A limit that a .env file can widen is not a
#: limit" (this module's own docstring) applies just as much to a future
#: sizing formula as it does to `.env`.
MAX_NOTIONAL_USD: Final[Decimal] = Decimal("10")

#: Volatility-parity risk budget, in USD (ad hoc, Phase 4) — the
#: DOLLAR AMOUNT of price movement (one day's worth, per the 14-day ATR)
#: this Tier-0 engine is willing to accept per position. Quantity is
#: ``RISK_BUDGET_USD / atr``, which is SMALLER for a more volatile
#: symbol (a bigger ATR) — the mechanism that "naturally sizes down
#: highly volatile assets like TSLA," per this feature's own stated
#: goal. This does NOT replace MAX_NOTIONAL_USD; both apply, and the
#: SMALLER of the two quantities always wins (see validate_and_plan).
#:
#: Calibrated against REAL Alpaca IEX daily bars, not guessed (checked
#: 2026-09-06): TSLA/AAPL/MSFT/GOOGL/AMZN's real 14-day ATRs were
#: $15.20/$7.38/$9.91/$6.50/$5.82 against prices of roughly
#: $354/$320/$500/$338/$258 — TSLA's ATR-to-price ratio (4.30%) was
#: genuinely the highest of the five, confirming it as the most volatile
#: by this measure, matching the task's own premise. $0.30 was chosen
#: specifically so that, against THAT real snapshot, TSLA's
#: volatility-based quantity (0.30/15.20 ≈ 0.0197 shares) is smaller
#: than its notional-based quantity (10/354 ≈ 0.0283 shares) and
#: therefore binds — while the other four names' volatility-based
#: quantities stay LARGER than their notional-based ones, so
#: MAX_NOTIONAL_USD keeps governing them exactly as before this feature
#: shipped. A real, honest limitation: ATR moves with the market, so
#: this constant may need periodic recalibration to keep having the
#: intended effect rather than silently becoming a no-op (if every
#: symbol's ATR falls) or binding on everything (if every symbol's ATR
#: rises) — it is not a law of physics the way the money-integer rules
#: are, just a calibrated snapshot.
RISK_BUDGET_USD: Final[Decimal] = Decimal("0.30")

#: Stop-loss distance below the entry limit price, in percent.
STOP_LOSS_PCT: Final[Decimal] = Decimal("5")

#: Hard portfolio circuit breaker. Checked in scheduler.trading_loop,
#: BEFORE inference even runs, against broker.get_daily_pnl_pct() — once
#: today's account P&L is at or below this, the loop skips straight to a
#: prolonged cooldown sleep instead of submitting anything. A risk limit
#: like every other constant in this block: hardcoded, not read from
#: .env, not reachable from any request payload.
CIRCUIT_BREAKER_DAILY_PNL_PCT: Final[Decimal] = Decimal("-2.5")

#: Marketable-limit buffer above the ask, in percent. Keeps the order
#: crossable without becoming an unbounded market order.
LIMIT_BUFFER_PCT: Final[Decimal] = Decimal("0.25")

#: Alpaca accepts up to 9 decimal places on fractional quantities.
_QTY_PRECISION: Final[Decimal] = Decimal("0.000000001")

#: US equities quote in pennies at/above $1.00.
_PRICE_PRECISION: Final[Decimal] = Decimal("0.01")

#: Alpaca rejects an order whose fractional qty rounds below this.
_MIN_FRACTIONAL_QTY: Final[Decimal] = Decimal("0.000000001")

# ===========================================================================


class RejectionCode(StrEnum):
    """Why the Tier-0 layer refused to forward an order."""

    BELOW_MIN_GAIN = "below_min_gain"
    NON_FINITE_SIGNAL = "non_finite_signal"
    INVALID_QUOTE = "invalid_quote"
    NOTIONAL_TOO_SMALL = "notional_too_small"
    CAP_BREACH = "cap_breach"


class StopLossKind(StrEnum):
    """How the stop-loss was attached.

    Alpaca's server rejects ``bracket``/``OCO``/``OTO`` order classes on
    fractional quantities. At a $10 cap almost every symbol sizes fractionally,
    so both paths are live in practice — see ``_choose_stop_strategy``.
    """

    #: Stop is resting at the broker as a one-triggers-other child order.
    NATIVE_OTO = "native_oto"
    #: Broker cannot hold the stop; the engine owns it and it is on the receipt.
    ENGINE_TRACKED = "engine_tracked"


class Tier0Rejection(Exception):
    """Raised when a signal fails validation. Carries a machine-readable code."""

    def __init__(self, code: RejectionCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class ExecutionPlan:
    """A fully-priced, validated order, not yet submitted."""

    __slots__ = (
        "symbol", "side", "quantity", "limit_price", "stop_price",
        "notional_usd", "stop_loss_kind", "order_class", "client_order_id",
    )

    def __init__(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        limit_price: Decimal,
        stop_price: Decimal,
        notional_usd: Decimal,
        stop_loss_kind: StopLossKind,
        order_class: OrderClass | None,
        client_order_id: str,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.notional_usd = notional_usd
        self.stop_loss_kind = stop_loss_kind
        self.order_class = order_class
        self.client_order_id = client_order_id

    def as_dict(self) -> dict[str, str | None]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "limit_price": str(self.limit_price),
            "stop_price": str(self.stop_price),
            "notional_usd": str(self.notional_usd),
            "stop_loss_kind": self.stop_loss_kind.value,
            "order_class": self.order_class.value if self.order_class else None,
            "client_order_id": self.client_order_id,
        }


def _choose_stop_strategy(quantity: Decimal) -> StopLossKind:
    """Pick the strongest stop the broker will actually accept.

    Whole-share orders get a real resting stop at Alpaca. Fractional orders
    cannot: the server rejects advanced order classes on fractional qty, so the
    stop is carried on the receipt and owned by the engine instead. The stop is
    never silently dropped.
    """
    is_whole = quantity == quantity.to_integral_value() and quantity >= Decimal(1)
    return StopLossKind.NATIVE_OTO if is_whole else StopLossKind.ENGINE_TRACKED


def validate_and_plan(signal: InferenceSignal, quote: Quote) -> ExecutionPlan:
    """Apply the Tier-0 gates and size the order.

    Args:
        signal: Model output. Only ``predicted_move_pct`` influences the gate.
        quote: Simulated NBBO snapshot used for pricing.

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

    # --- Gate 2: the quote must be sane ------------------------------------
    if quote.ask <= 0 or quote.bid <= 0 or quote.ask < quote.bid:
        raise Tier0Rejection(
            RejectionCode.INVALID_QUOTE,
            f"unusable quote bid={quote.bid} ask={quote.ask}",
        )

    # --- Pricing ------------------------------------------------------------
    # Round the entry UP and the stop DOWN: both directions are conservative,
    # so rounding can never widen risk beyond the configured band.
    limit_price = (quote.ask * (Decimal(1) + LIMIT_BUFFER_PCT / Decimal(100))
                   ).quantize(_PRICE_PRECISION, rounding=ROUND_UP)
    stop_price = (limit_price * (Decimal(1) - STOP_LOSS_PCT / Decimal(100))
                  ).quantize(_PRICE_PRECISION, rounding=ROUND_DOWN)

    # --- Gate 3: hard notional cap, tightened by volatility parity --------
    # ROUND_DOWN so each candidate can only ever bind tighter, never looser.
    notional_based_quantity = (MAX_NOTIONAL_USD / limit_price).quantize(
        _QTY_PRECISION, rounding=ROUND_DOWN
    )

    # Volatility-parity sizing (ad hoc, Phase 4) — ADDITIVE to the notional
    # cap above, never a replacement for it: quantity is always the
    # SMALLER of the two, so MAX_NOTIONAL_USD remains an absolute ceiling
    # no matter what the ATR-based figure says (see RISK_BUDGET_USD's own
    # doc comment for the calibration this relies on). `quote.atr` is
    # `None` on the synthetic-quote fallback path (no real market data to
    # compute a real ATR from) — sizing degrades to the notional-only
    # figure in that case, the same graceful-degradation posture every
    # other real-data dependency in this app already has.
    quantity = notional_based_quantity
    if quote.atr is not None and quote.atr > 0:
        volatility_based_quantity = (RISK_BUDGET_USD / quote.atr).quantize(
            _QTY_PRECISION, rounding=ROUND_DOWN
        )
        quantity = min(notional_based_quantity, volatility_based_quantity)

    if quantity < _MIN_FRACTIONAL_QTY:
        raise Tier0Rejection(
            RejectionCode.NOTIONAL_TOO_SMALL,
            f"${MAX_NOTIONAL_USD} buys {quantity} of {quote.ticker} at "
            f"{limit_price}, below Alpaca's minimum fractional quantity",
        )

    notional_usd = (quantity * limit_price).quantize(
        _PRICE_PRECISION, rounding=ROUND_HALF_UP
    )

    # --- Post-condition: the cap held --------------------------------------
    # Belt-and-braces. If sizing arithmetic is ever changed and this trips,
    # the order is dropped rather than sent oversized.
    if notional_usd > MAX_NOTIONAL_USD:
        raise Tier0Rejection(
            RejectionCode.CAP_BREACH,
            f"computed notional ${notional_usd} exceeds the hard cap "
            f"${MAX_NOTIONAL_USD} — order suppressed",
        )

    stop_loss_kind = _choose_stop_strategy(quantity)
    return ExecutionPlan(
        symbol=quote.ticker,
        side=OrderSide.BUY,
        quantity=quantity,
        limit_price=limit_price,
        stop_price=stop_price,
        notional_usd=notional_usd,
        stop_loss_kind=stop_loss_kind,
        order_class=(
            OrderClass.OTO if stop_loss_kind is StopLossKind.NATIVE_OTO else None
        ),
        client_order_id=uuid.uuid4().hex,
    )


def build_order_request(plan: ExecutionPlan) -> LimitOrderRequest:
    """Translate a validated plan into an Alpaca ``LimitOrderRequest``.

    ``TimeInForce.DAY`` is mandatory: Alpaca supports no other TIF on
    fractional quantities.
    """
    kwargs: dict[str, object] = {
        "symbol": plan.symbol,
        "qty": float(plan.quantity),  # single float cast, at the API boundary
        "side": plan.side,
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

    order = get_trading_client().submit_order(build_order_request(plan))
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
