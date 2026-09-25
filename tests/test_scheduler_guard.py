"""The monolith's submitting paths honour the kill switch and the breaker.

``POST /signals/execute`` calls ``run_signal_cycle(submit=True)`` directly;
both checks used to live only in ``trading_loop``, so a manual call placed
orders after an emergency halt or a -2.5% day.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import execution
import scheduler


@pytest.fixture
def no_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*_: object, **__: object) -> dict:
        raise AssertionError("execute_signal must not be reached")

    monkeypatch.setattr(execution, "execute_signal", refuse)


@pytest.mark.asyncio
async def test_manual_execute_is_blocked_while_halted(monkeypatch: pytest.MonkeyPatch, no_orders: None) -> None:
    monkeypatch.setattr(scheduler, "IS_HALTED", True)
    result = await scheduler.run_signal_cycle("TSLA", "TSLA soars", submit=True)
    assert result["executed"] is False
    assert result["rejection"]["code"] == "halted"


@pytest.mark.asyncio
async def test_manual_execute_is_blocked_by_the_circuit_breaker(monkeypatch: pytest.MonkeyPatch, no_orders: None) -> None:
    async def breached() -> Decimal:
        return Decimal("-2.6")

    monkeypatch.setattr(scheduler, "_fetch_daily_pnl_pct", breached)
    result = await scheduler.run_signal_cycle("TSLA", "TSLA soars", submit=True)
    assert result["rejection"]["code"] == "circuit_breaker"


@pytest.mark.asyncio
async def test_no_orders_once_placement_moves_to_the_router(monkeypatch: pytest.MonkeyPatch, no_orders: None) -> None:
    monkeypatch.setenv("ORDERS_VIA_RISK_ROUTER", "true")
    result = await scheduler.run_signal_cycle("TSLA", "TSLA soars", submit=True)
    assert result["rejection"]["code"] == "orders_moved"


def test_execute_endpoint_is_gone_once_placement_moves(monkeypatch: pytest.MonkeyPatch, no_orders: None) -> None:
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setenv("ORDERS_VIA_RISK_ROUTER", "true")
    # Not used as a context manager: that would run the lifespan, which
    # opens a real Alpaca stream with whatever keys .env holds.
    client = TestClient(main.app)
    response = client.post("/signals/execute", json={"ticker": "TSLA", "headline": "TSLA soars"})
    assert response.status_code == 410
    assert client.get("/health").json()["places_orders"] is False
