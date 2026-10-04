"""The owner's console: the signed calls that approve the initial build, raise cash, and stop trading, driven
against the real core app over an in-process transport."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_core_paper import Clock, FakeBroker, policy_file, proposal

import config
from risk_router.core_app import create_core_app
from risk_router.core_ctl import execute, parser

D = Decimal
OWNER_SECRET = "c" * 40
EVENING = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def secrets(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    monkeypatch.setenv("WEBHOOK_SECRET", OWNER_SECRET)
    clear = getattr(config.get_webhook_settings, "cache_clear", None)
    if clear:
        clear()          # the control routes read the secret through a cached settings object
    yield
    if clear:
        clear()


@asynccontextmanager
async def running_app(tmp_path: Path, broker: FakeBroker | None = None) -> AsyncIterator[Any]:
    app = create_core_app(alpaca=broker or FakeBroker(), background=False, state_dir=tmp_path,
                          policy_path=policy_file(tmp_path), now=Clock(EVENING))
    async with app.router.lifespan_context(app):
        assert app.state.disabled is None
        yield app


async def run(app: Any, *argv: str, secret: str = OWNER_SECRET, yes: bool = True) -> tuple[int, str]:
    lines: list[str] = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://core") as http:
        code = await execute(parser().parse_args(list(argv)), http, secret, lambda question: yes, out=lines.append)
    return code, "\n".join(lines)


@pytest.mark.asyncio
async def test_status_shows_what_is_waiting_for_the_owner(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        await app.state.router.receive_plan(await proposal(app.state.router, "2026-09-30"))
        code, out = await run(app, "status")
    assert code == 0
    assert "awaiting_approval" in out and "buy  $1899.81 of VTI" in out and "executes 2026-10-01" in out
    assert "waiting for your signed approval" in out


@pytest.mark.asyncio
async def test_the_initial_build_needs_the_fund_flag_and_a_yes(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        router = app.state.router
        await router.receive_plan(await proposal(router, "2026-09-30"))

        code, out = await run(app, "approve")
        assert code == 1 and "--fund" in out and router.store.data.plan["status"] == "awaiting_approval"

        code, out = await run(app, "approve", "--fund", yes=False)             # asked, and answered no
        assert code == 1 and "not approved" in out and router.store.data.plan["status"] == "awaiting_approval"

        code, out = await run(app, "approve", "--fund")                        # asked, and answered yes
        assert code == 0 and "will execute at the open on 2026-10-01" in out
        assert router.store.data.plan["status"] == "approved"


@pytest.mark.asyncio
async def test_approve_checks_it_is_the_plan_you_meant(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        await app.state.router.receive_plan(await proposal(app.state.router, "2026-09-30"))
        code, out = await run(app, "approve", "deadbeefdeadbeef", "--fund")
        assert code == 1 and "not deadbeefdeadbeef" in out
        assert app.state.router.store.data.plan["status"] == "awaiting_approval"


@pytest.mark.asyncio
async def test_with_nothing_waiting_there_is_nothing_to_approve(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        code, out = await run(app, "approve", "--fund")
    assert code == 1 and "nothing is waiting" in out


@pytest.mark.asyncio
async def test_a_wrong_owner_secret_is_reported_not_silently_ignored(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        await app.state.router.receive_plan(await proposal(app.state.router, "2026-09-30"))
        code, out = await run(app, "approve", "--fund", secret="x" * 40)
        assert code == 1 and "rejected the signature" in out
        assert app.state.router.store.data.plan["status"] == "awaiting_approval"
        code, out = await run(app, "status", secret="x" * 40)
        assert code == 1 and "rejected the signature" in out


@pytest.mark.asyncio
async def test_the_allocators_secret_cannot_approve(tmp_path: Path) -> None:
    """The Allocator's CORE_PLAN_SECRET and the owner's WEBHOOK_SECRET are different keys on purpose."""
    async with running_app(tmp_path) as app:
        await app.state.router.receive_plan(await proposal(app.state.router, "2026-09-30"))
        code, out = await run(app, "approve", "--fund", secret="p" * 40)
    assert code == 1 and "rejected the signature" in out


@pytest.mark.asyncio
async def test_halt_asks_first_and_then_stops_everything(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        code, out = await run(app, "halt", yes=False)
        assert code == 1 and "not halted" in out and not app.state.guard.state.halted
        code, out = await run(app, "halt", "--yes")
        assert code == 0 and "halted" in out and app.state.guard.state.halted
        code, out = await run(app, "status")
        assert "HALTED" in out and "kill switch is on" in out


@pytest.mark.asyncio
async def test_a_refused_raise_cash_request_says_why(tmp_path: Path) -> None:
    async with running_app(tmp_path) as app:
        code, out = await run(app, "raise-cash", "--amount", "1000000", "--need-by", "2026-10-09")
    assert code == 1 and "refused" in out and "exceeds_portfolio" in out


@pytest.mark.asyncio
async def test_a_raise_cash_request_becomes_a_plan_waiting_for_approval(tmp_path: Path) -> None:
    broker = FakeBroker()
    async with running_app(tmp_path, broker) as app:
        router = app.state.router
        clock: Clock = router._now                                        # type: ignore[assignment]
        answer = await router.receive_plan(await proposal(router, "2026-09-30"))
        await router.approve(answer["plan_id"], fund=True)
        clock.set("2026-10-01", 9, 25)
        await router.tick()
        broker.fill_all("buy")
        clock.set("2026-10-01", 9, 40)
        await router.tick()
        clock.set("2026-10-01", 16, 30)
        code, out = await run(app, "raise-cash", "--amount", "400", "--need-by", "2026-10-09")
        assert code == 0 and "waiting for your approval" in out
        code, out = await run(app, "approve", "--yes")                    # a raise-cash plan needs no fund flag
    assert code == 0 and "sell" in out and date(2026, 10, 2).isoformat() in out


def test_an_amount_that_is_not_a_number_is_rejected_before_anything_is_sent() -> None:
    import asyncio

    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing may be sent")

    async def go() -> tuple[int, str]:
        lines: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(never), base_url="http://core") as http:
            code = await execute(parser().parse_args(["raise-cash", "--amount", "lots", "--need-by", "2026-10-09"]),
                                 http, OWNER_SECRET, lambda q: True, out=lines.append)
        return code, "\n".join(lines)

    code, out = asyncio.run(go())
    assert code == 1 and "must be a number" in out


@pytest.mark.asyncio
async def test_test_alert_says_whether_it_reached_you(monkeypatch: pytest.MonkeyPatch) -> None:
    import risk_router.core_alerts as alerts
    from risk_router.core_ctl import execute, parser
    seen: list[httpx.Request] = []
    monkeypatch.delenv("CORE_ALERT_WEBHOOK_URL", raising=False)
    lines: list[str] = []
    assert await execute(parser().parse_args(["test-alert"]), None, "", lambda q: True, out=lines.append) == 1  # type: ignore[arg-type]
    assert "not set" in lines[0]

    monkeypatch.setenv("CORE_ALERT_WEBHOOK_URL", "https://hooks.example/core")
    real = alerts.Alerter

    class Wired(real):                                   # route the real Alerter through an in-process transport
        def __init__(self, url: str | None, fmt: str = "json", **kw: Any) -> None:
            super().__init__(url, fmt, transport=httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200))[1]))

    monkeypatch.setattr("risk_router.core_ctl.Alerter", Wired)
    lines.clear()
    assert await execute(parser().parse_args(["test-alert"]), None, "", lambda q: True, out=lines.append) == 0  # type: ignore[arg-type]
    assert "delivered" in lines[0] and len(seen) == 1
