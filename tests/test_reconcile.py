"""reconcile.py — re-deriving settlement receipts from Alpaca's order history.

A fake trading client and a scripted sender: what's under test is the
selection (only genuinely FILLED orders with fill data), the outcome
tallies, the recovery callback, and the never-raises contract — not
alpaca-py or HTTP.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from alpaca.trading.enums import OrderStatus

from reconcile import ReconcileStats, reconcile_recent_fills


def _order(client_order_id: str, *, status: OrderStatus = OrderStatus.FILLED, price: str | None = "362.25", qty: str | None = "0.02"):
    return SimpleNamespace(
        id=f"id-{client_order_id}", client_order_id=client_order_id, symbol="TSLA", side="OrderSide.BUY",
        status=status, filled_avg_price=price, filled_qty=qty,
    )


class FakeClient:
    def __init__(self, orders: list[Any]) -> None:
        self.orders = orders
        self.requests: list[Any] = []

    def get_orders(self, request: Any) -> list[Any]:
        self.requests.append(request)
        return self.orders


class ScriptedSender:
    """Maps client_order_id -> delivery dict, records every call."""

    def __init__(self, outcomes: dict[str, dict[str, Any]]) -> None:
        self.outcomes = outcomes
        self.sent: list[str] = []

    async def __call__(self, order: Any) -> dict[str, Any]:
        self.sent.append(order.client_order_id)
        return self.outcomes[order.client_order_id]


RECOVERED = {"delivered": True, "status_code": 201}
ALREADY = {"delivered": True, "status_code": 200}
QUEUED = {"delivered": False, "queued": True, "error": "ConnectError"}
REJECTED = {"delivered": False, "queued": False, "error": "HTTP 400: insufficient_shares"}


@pytest.mark.asyncio
async def test_only_filled_orders_with_fill_data_are_resent() -> None:
    client = FakeClient([
        _order("a"),
        _order("b", status=OrderStatus.CANCELED),
        _order("c", price=None),
        _order("d", qty=None),
        _order("e"),
    ])
    sender = ScriptedSender({"a": ALREADY, "e": ALREADY})
    stats = await reconcile_recent_fills(client=client, send=sender)
    assert sender.sent == ["a", "e"]
    assert stats == ReconcileStats(orders_seen=5, fills=2, already_booked=2)
    assert client.requests[0].status.value == "closed"


@pytest.mark.asyncio
async def test_tallies_recovered_already_booked_queued_and_rejected_and_fires_the_callback() -> None:
    client = FakeClient([_order("r"), _order("a"), _order("q"), _order("x")])
    sender = ScriptedSender({"r": RECOVERED, "a": ALREADY, "q": QUEUED, "x": REJECTED})
    recovered: list[str] = []
    stats = await reconcile_recent_fills(client=client, send=sender, on_recovered=lambda o: recovered.append(o.client_order_id))
    assert stats == ReconcileStats(orders_seen=4, fills=4, recovered=1, already_booked=1, queued=1, rejected=1)
    assert recovered == ["r"]  # only a genuine 201 recovery is surfaced as telemetry


@pytest.mark.asyncio
async def test_a_broker_outage_is_a_logged_no_op_never_a_crash() -> None:
    class ExplodingClient:
        def get_orders(self, request: Any) -> list[Any]:
            raise ConnectionError("alpaca down")

    sender = ScriptedSender({})
    stats = await reconcile_recent_fills(client=ExplodingClient(), send=sender)
    assert stats.error == "ConnectionError: alpaca down"
    assert stats.orders_seen == 0 and sender.sent == []


@pytest.mark.asyncio
async def test_lookback_window_is_passed_to_the_broker_query() -> None:
    client = FakeClient([])
    stats = await reconcile_recent_fills(lookback_hours=6, client=client, send=ScriptedSender({}))
    assert stats == ReconcileStats()
    request = client.requests[0]
    assert request.after is not None and request.limit == 500
