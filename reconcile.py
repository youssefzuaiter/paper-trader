"""Reconciliation against Alpaca's own order history.

The outbox (``outbox.py``) makes a receipt durable from the moment this
process fails to deliver it. It cannot help with a receipt that was
never generated in the first place, or one this process believed was
delivered when it wasn't — both happened for real before the outbox and
the settlement-race fix existed: fills whose settlement PFW answered
"200 duplicate" to and silently dropped, leaving the trade ``PENDING``
with no ledger row, and one order whose receipts were lost outright
while PFW was down.

Alpaca is the source of truth for what actually filled, so this module
asks it: every CLOSED order in the lookback window that is FILLED gets
its settlement receipt rebuilt from the broker's own fill data
(``webhook.build_settlement_receipt``, the same function the live
WebSocket path uses — never a second receipt shape) and re-sent. PFW
handles every outcome of a re-sent settlement idempotently: an
already-settled trade answers ``200 duplicate``, a stranded ``PENDING``
trade gets settled (``201``), and a fill PFW never saw at all becomes a
settled trade (``201``) — so re-sending is always safe and the only cost
is one round-trip per recent fill.

Runs once at startup and then hourly (``reconcile_loop``), and on demand
via ``POST /control/reconcile`` (HMAC-verified like ``/control/halt``).
Never raises: a broker outage makes a pass a logged no-op, exactly as a
PFW outage does for the outbox.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Final

from alpaca.trading.enums import OrderStatus, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

import broker
import webhook

logger = logging.getLogger(__name__)

RECONCILE_LOOKBACK_HOURS: Final[int] = 24
RECONCILE_INTERVAL_SECONDS: Final[float] = 3600.0
#: How long after startup the first pass runs — lets the WebSocket
#: settlement stream connect first, so a fill landing right at boot is
#: settled by the live path and merely confirmed by this one.
RECONCILE_STARTUP_DELAY_SECONDS: Final[float] = 15.0
_MAX_ORDERS: Final[int] = 500

SendSettlement = Callable[[Any], Awaitable[dict[str, Any]]]
OnRecovered = Callable[[Any], None]


@dataclass(frozen=True)
class ReconcileStats:
    """One pass, itemised. ``recovered`` is the number that matters: fills
    PFW had not booked until this pass re-sent them."""

    orders_seen: int = 0
    fills: int = 0
    recovered: int = 0
    already_booked: int = 0
    queued: int = 0
    rejected: int = 0
    error: str | None = None


async def reconcile_recent_fills(
    *,
    lookback_hours: int = RECONCILE_LOOKBACK_HOURS,
    client: Any | None = None,
    send: SendSettlement = webhook.send_settlement_receipt,
    on_recovered: OnRecovered | None = None,
) -> ReconcileStats:
    """Re-send the settlement receipt for every fill in the window. Never raises."""
    after = datetime.now(UTC) - timedelta(hours=lookback_hours)
    try:
        trading_client = client or broker.get_trading_client()
        orders = await asyncio.to_thread(
            trading_client.get_orders,
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after, limit=_MAX_ORDERS),
        )
    except Exception as exc:  # noqa: BLE001 — a broker outage is a logged no-op, never a crash
        logger.warning("reconcile: could not list orders from the broker: %s: %s", type(exc).__name__, exc)
        return ReconcileStats(error=f"{type(exc).__name__}: {exc}")

    fills = [o for o in orders if _is_settleable_fill(o)]
    recovered = already_booked = queued = rejected = 0
    for order in fills:
        delivery = await send(order)
        if delivery.get("delivered"):
            if delivery.get("status_code") == 201:
                recovered += 1
                logger.warning(
                    "reconcile: RECOVERED fill %s %s qty=%s @ %s (order %s) — PFW had not booked it",
                    order.symbol, str(order.side).rsplit(".", maxsplit=1)[-1].lower(),
                    order.filled_qty, order.filled_avg_price, order.id,
                )
                if on_recovered is not None:
                    on_recovered(order)
            else:
                already_booked += 1
        elif delivery.get("queued"):
            queued += 1
        else:
            rejected += 1
            logger.error("reconcile: PFW rejected settlement for order %s: %s", order.id, delivery.get("error"))

    stats = ReconcileStats(
        orders_seen=len(orders), fills=len(fills), recovered=recovered,
        already_booked=already_booked, queued=queued, rejected=rejected,
    )
    logger.info("reconcile: pass over the last %dh — %s", lookback_hours, stats)
    return stats


def _is_settleable_fill(order: Any) -> bool:
    """The same guards the live WebSocket path applies before settling."""
    status = getattr(order, "status", None)
    if status != OrderStatus.FILLED and str(status).rsplit(".", maxsplit=1)[-1].lower() != "filled":
        return False
    return bool(getattr(order, "filled_avg_price", None)) and bool(getattr(order, "filled_qty", None))


async def reconcile_loop(*, on_recovered: OnRecovered | None = None) -> None:
    """Background task: one pass shortly after startup, then hourly."""
    await asyncio.sleep(RECONCILE_STARTUP_DELAY_SECONDS)
    while True:
        try:
            await reconcile_recent_fills(on_recovered=on_recovered)
        except Exception:  # noqa: BLE001 — belt and braces; reconcile_recent_fills already never raises
            logger.exception("reconcile: pass failed unexpectedly")
        await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)
