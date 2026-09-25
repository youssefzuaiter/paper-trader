"""Tier-0 risk limits and order sizing — the one copy every service shares.

Split out of ``execution.py`` so the Risk & Routing agent
(``risk_router/``) enforces exactly the same limits and the same
``Decimal`` sizing as the monolith, without importing ``execution.py`` —
which pulls in ``inference.py`` and therefore torch, ~700 MB of image the
gatekeeper has no use for. Pure standard library, no I/O — not even
alpaca-py, whose enums import pandas; ``OrderSide``/``OrderClass`` below
carry the same wire values and ``execution.py`` converts at its boundary.

Every risk constant below is a hardcoded module ``Final``. They are *not*
read from ``.env``, *not* function parameters, and *not* reachable from
any request body. A limit that a caller can widen is not a limit.

Money never touches ``float``. Sizing is ``Decimal`` with explicit rounding
modes, and every rounding goes in the direction that tightens risk.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from enum import StrEnum
from typing import Final

# ===========================================================================
# TIER-0 HARDCODED LIMITS — do not parameterise, do not move to .env
# ===========================================================================

#: Reject any trade whose predicted gain is below this, in percent.
#: The monolith's entry gate (``execution.validate_and_plan``). The Risk &
#: Routing agent gates on ``ROUTER_MIN_PROB_UP`` instead — its signals are
#: calibrated probabilities, not a rescaled sentiment score.
MIN_PREDICTED_GAIN_PCT: Final[Decimal] = Decimal("10")

#: Hard cap on capital exposure per order, in USD. Fractional sizing exists
#: precisely so this cap binds on expensive symbols instead of rejecting them.
#: NEVER WIDENED by the volatility-parity sizing below — quantity is
#: always the SMALLER of this notional-based figure and the ATR-based
#: one, by construction (``min(...)``), so this remains the absolute
#: ceiling on capital exposure regardless of how volatility sizing
#: changes in the future.
MAX_NOTIONAL_USD: Final[Decimal] = Decimal("10")

#: Volatility-parity risk budget, in USD — the DOLLAR AMOUNT of price
#: movement (one day's worth, per the 14-day ATR) this engine is willing
#: to accept per position. Quantity is ``RISK_BUDGET_USD / atr``, which is
#: SMALLER for a more volatile symbol. This does NOT replace
#: MAX_NOTIONAL_USD; both apply, and the SMALLER quantity always wins.
#:
#: Calibrated against REAL Alpaca IEX daily bars (2026-09-06):
#: TSLA/AAPL/MSFT/GOOGL/AMZN's 14-day ATRs were $15.20/$7.38/$9.91/$6.50/
#: $5.82 against prices of roughly $354/$320/$500/$338/$258. $0.30 makes
#: TSLA's volatility-based quantity (0.30/15.20 ≈ 0.0197) bind below its
#: notional-based one (10/354 ≈ 0.0283) while the other four stay governed
#: by MAX_NOTIONAL_USD. ATR moves with the market, so this is a calibrated
#: snapshot that may need periodic recalibration, not a law of physics.
RISK_BUDGET_USD: Final[Decimal] = Decimal("0.30")

#: Stop-loss distance below the entry price, in percent. Enforced at the
#: broker for whole-share orders (OTO child) and by the Risk & Routing
#: agent's exit monitor for everything else.
STOP_LOSS_PCT: Final[Decimal] = Decimal("5")

#: Take-profit distance above the average entry price, in percent. Enforced
#: by the Risk & Routing agent's exit monitor.
TAKE_PROFIT_PCT: Final[Decimal] = Decimal("10")

#: Hard portfolio circuit breaker: once today's account P&L is at or below
#: this, no new position may be opened for the rest of the trading day.
#: Risk-reducing orders (exits) are still allowed — a breaker that blocked
#: stop-losses would trap the account in the losses it exists to limit.
CIRCUIT_BREAKER_DAILY_PNL_PCT: Final[Decimal] = Decimal("-2.5")

#: Marketable-limit buffer above the ask, in percent. Keeps the order
#: crossable without becoming an unbounded market order.
LIMIT_BUFFER_PCT: Final[Decimal] = Decimal("0.25")

# --- Portfolio limits, enforced by the Risk & Routing agent ----------------
# These close the "exposure loophole": MAX_NOTIONAL_USD bounds ONE order,
# and nothing used to bound how many of them accumulated. Dollar caps, not
# percentage-of-equity weights: at $10 an order against a $100k paper
# account every weight would round to ~0% and bind on nothing.

#: No pyramiding: a symbol with an open position OR a working buy order
#: cannot be bought again until it is fully exited.
ALLOW_PYRAMIDING: Final[bool] = False

#: Ceiling on the summed market value of all positions plus the notional
#: of every working buy order, in USD.
MAX_GROSS_EXPOSURE_USD: Final[Decimal] = Decimal("50")

#: Ceiling on distinct symbols held (or being bought) at once.
MAX_OPEN_POSITIONS: Final[int] = 5

#: Ceiling on buy orders submitted per trading day, across every service
#: that trades this account — counted from Alpaca's own order history, so
#: it survives restarts and cannot be reset by redeploying.
MAX_NEW_BUYS_PER_DAY: Final[int] = 10

#: After a symbol is sold, how long before it may be bought again.
REENTRY_COOLDOWN_SECONDS: Final[int] = 4 * 3600

#: The swarm's holding period is intraday, because that is the horizon the
#: return model was trained on (swarm/train_return_model.py): the router
#: opens no position in the last LAST_ENTRY_BEFORE_CLOSE of a session and
#: sells every position SESSION_EXIT_BEFORE_CLOSE before the bell, early
#: enough for a market order to fill in the regular session. Measured from
#: Alpaca's clock (next_close), so a 13:00 half-day is handled too.
LAST_ENTRY_BEFORE_CLOSE: Final[timedelta] = timedelta(minutes=30)
SESSION_EXIT_BEFORE_CLOSE: Final[timedelta] = timedelta(minutes=10)

#: Minimum calibrated probability of an up-move (from the Inference
#: Agents' scikit-learn pipeline) before the router will open a position.
#: A placeholder until that model has been backtested — revisit with its
#: calibration curve, not by feel.
ROUTER_MIN_PROB_UP: Final[Decimal] = Decimal("0.60")

#: Alpaca accepts up to 9 decimal places on fractional quantities.
QTY_PRECISION: Final[Decimal] = Decimal("0.000000001")

#: US equities quote in pennies at/above $1.00.
PRICE_PRECISION: Final[Decimal] = Decimal("0.01")

#: Alpaca rejects an order whose fractional qty rounds below this.
MIN_FRACTIONAL_QTY: Final[Decimal] = Decimal("0.000000001")

# ===========================================================================


class OrderSide(StrEnum):
    """Same values as ``alpaca.trading.enums.OrderSide``."""

    BUY = "buy"
    SELL = "sell"


class OrderClass(StrEnum):
    """The one order class this engine uses; value as in alpaca-py."""

    OTO = "oto"


class RejectionCode(StrEnum):
    """Why a Tier-0 layer refused to forward an order."""

    BELOW_MIN_GAIN = "below_min_gain"
    NON_FINITE_SIGNAL = "non_finite_signal"
    INVALID_QUOTE = "invalid_quote"
    NOTIONAL_TOO_SMALL = "notional_too_small"
    CAP_BREACH = "cap_breach"


class StopLossKind(StrEnum):
    """How the stop-loss was attached.

    Alpaca's server rejects ``bracket``/``OCO``/``OTO`` order classes on
    fractional quantities. At a $10 cap almost every symbol sizes
    fractionally, so both paths are live in practice.
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


def choose_stop_strategy(quantity: Decimal) -> StopLossKind:
    """Pick the strongest stop the broker will actually accept.

    Whole-share orders get a real resting stop at Alpaca. Fractional orders
    cannot: the server rejects advanced order classes on fractional qty, so
    the stop is carried on the receipt and owned by the engine instead.
    """
    is_whole = quantity == quantity.to_integral_value() and quantity >= Decimal(1)
    return StopLossKind.NATIVE_OTO if is_whole else StopLossKind.ENGINE_TRACKED


def plan_buy(
    *,
    symbol: str,
    bid: Decimal,
    ask: Decimal,
    atr: Decimal | None,
    client_order_id: str | None = None,
) -> ExecutionPlan:
    """Price and size a BUY under the Tier-0 limits.

    The part of the Tier-0 gate that does not depend on how the signal was
    produced: quote sanity, marketable-limit pricing, notional cap tightened
    by volatility parity, and the cap post-condition. Callers apply their
    own entry gate first (``execution.validate_and_plan``: +10% predicted
    move; the Risk & Routing agent: calibrated up-probability).

    Raises:
        Tier0Rejection: ``invalid_quote``, ``notional_too_small`` or
            ``cap_breach``.
    """
    if ask <= 0 or bid <= 0 or ask < bid:
        raise Tier0Rejection(
            RejectionCode.INVALID_QUOTE,
            f"unusable quote bid={bid} ask={ask}",
        )

    # Round the entry UP and the stop DOWN: both directions are conservative,
    # so rounding can never widen risk beyond the configured band.
    limit_price = (ask * (Decimal(1) + LIMIT_BUFFER_PCT / Decimal(100))
                   ).quantize(PRICE_PRECISION, rounding=ROUND_UP)
    stop_price = (limit_price * (Decimal(1) - STOP_LOSS_PCT / Decimal(100))
                  ).quantize(PRICE_PRECISION, rounding=ROUND_DOWN)

    # ROUND_DOWN so each candidate can only ever bind tighter, never looser.
    quantity = (MAX_NOTIONAL_USD / limit_price).quantize(QTY_PRECISION, rounding=ROUND_DOWN)
    # Volatility parity is ADDITIVE to the notional cap, never a
    # replacement: the smaller quantity wins, so MAX_NOTIONAL_USD stays an
    # absolute ceiling. No ATR (synthetic quote, data gap) → notional only.
    if atr is not None and atr > 0:
        volatility_quantity = (RISK_BUDGET_USD / atr).quantize(QTY_PRECISION, rounding=ROUND_DOWN)
        quantity = min(quantity, volatility_quantity)

    if quantity < MIN_FRACTIONAL_QTY:
        raise Tier0Rejection(
            RejectionCode.NOTIONAL_TOO_SMALL,
            f"${MAX_NOTIONAL_USD} buys {quantity} of {symbol} at "
            f"{limit_price}, below Alpaca's minimum fractional quantity",
        )

    notional_usd = (quantity * limit_price).quantize(PRICE_PRECISION, rounding=ROUND_HALF_UP)

    # Belt-and-braces. If sizing arithmetic is ever changed and this trips,
    # the order is dropped rather than sent oversized.
    if notional_usd > MAX_NOTIONAL_USD:
        raise Tier0Rejection(
            RejectionCode.CAP_BREACH,
            f"computed notional ${notional_usd} exceeds the hard cap "
            f"${MAX_NOTIONAL_USD} — order suppressed",
        )

    stop_loss_kind = choose_stop_strategy(quantity)
    return ExecutionPlan(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        limit_price=limit_price,
        stop_price=stop_price,
        notional_usd=notional_usd,
        stop_loss_kind=stop_loss_kind,
        order_class=OrderClass.OTO if stop_loss_kind is StopLossKind.NATIVE_OTO else None,
        client_order_id=client_order_id or uuid.uuid4().hex,
    )
