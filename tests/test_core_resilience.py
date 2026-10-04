"""The long-term core when things go wrong: a process that dies mid-rebalance, an order Alpaca refuses, a
breaker that trips after the open, a proposal sent twice, a journal that was cut short.

Each test here began as a reproduction of a real defect (a plan stuck in ``selling`` forever, a re-sent
proposal that undid the owner's approval, ``effective_from`` that was parsed and never read). They drive the
real ``CoreRouter`` through the same fake account the happy-path tests use."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from test_core_paper import (
    SESSIONS,
    Clock,
    FakeBroker,
    built,
    make_router,
    policy,
    policy_file,
    proposal,
)

import core_paper
from risk_router.alpaca_async import AlpacaError, OrderRejected
from risk_router.core_gatekeeper import CoreRouter, CoreStore, Journal, PlanRejected

D = Decimal
START = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)
QUARTER_END = ["2026-12-30", "2026-12-31", "2027-01-04", "2027-01-05", "2027-01-06", "2027-01-07", "2027-01-08"]


class RecordingAlerter:
    """Stands in for ``core_alerts.Alerter``: remembers what would have been pushed."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> None:
        self.sent.append((title, message))


class FlakyBroker(FakeBroker):
    """A fake account whose order endpoint can refuse, or fail in transit, on command."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.reject_sells: set[str] = set()                # symbols whose sell Alpaca refuses outright (a 4xx)
        self.fail_transient_after: int | None = None       # raise a 503 once this many orders have gone in
        self.cancel_calls = 0

    async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload["side"] == "sell" and payload["symbol"] in self.reject_sells:
            raise OrderRejected(f"rejected {payload['symbol']}", status_code=422)
        if self.fail_transient_after is not None and len(self.submitted) >= self.fail_transient_after:
            self.fail_transient_after = None
            raise AlpacaError("HTTP 503", status_code=503)
        return await super().submit_order(payload)

    async def cancel_all_orders(self) -> int:
        self.cancel_calls += 1
        return await super().cancel_all_orders()


@pytest.fixture(autouse=True)
def restore_the_calendar() -> Iterator[None]:
    backup = list(SESSIONS)
    yield
    SESSIONS[:] = backup


async def build_portfolio(router: CoreRouter, broker: FakeBroker, clock: Clock) -> None:
    """The initial build, through the router's own front door."""
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    clock.set("2026-10-01", 9, 25)
    await router.tick()
    broker.fill_all("buy")
    clock.set("2026-10-01", 9, 40)
    await router.tick()


async def rally_and_submit_sells(router: CoreRouter, broker: FakeBroker, clock: Clock,
                                 rallied: tuple[str, ...] = ("VTI",)) -> None:
    """VTI (and friends) rally, the quarter-end plan is proposed and approved, and its sells go in."""
    for symbol in rallied:
        broker.closes[symbol] = D("140")
    broker.submitted.clear()
    router.store.data.history.clear()                    # this is about execution, not the monthly cap
    SESSIONS[:] = QUARTER_END
    clock.set("2026-12-31", 16, 30)
    answer = await router.receive_plan(await proposal(router, "2026-12-31"))
    if answer["status"] == "awaiting_approval":
        await router.approve(answer["plan_id"])
    clock.set("2027-01-04", 9, 22)
    await router.tick()


async def at_the_sells(tmp_path: Path, broker: FlakyBroker | None = None,
                       alerter: RecordingAlerter | None = None) -> tuple[CoreRouter, FlakyBroker, Clock]:
    """A built portfolio whose quarter-end rebalance has just submitted its sells."""
    broker, clock = broker or FlakyBroker(), Clock(START)
    router = await built(tmp_path, broker, clock)
    router.alerter = alerter
    await rally_and_submit_sells(router, broker, clock)
    return router, broker, clock


def events(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]


# --- a process that dies between the sells and the buys ----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_crash_between_the_sells_and_the_buys_is_abandoned_and_the_cash_is_reinvested(tmp_path: Path) -> None:
    router, broker, clock = await at_the_sells(tmp_path)
    assert router.store.data.plan["status"] == "selling"
    broker.fill_all("sell")
    idle = broker.cash                                   # the process dies here: no tick for the rest of the day

    clock.set("2027-01-05", 9, 25)                       # it restarts the next morning
    await router.tick()
    assert router.store.data.plan is None, "the half-finished plan must be closed, not left in 'selling' forever"
    assert router.store.data.forced is True, "the next close must re-decide it"
    assert not [o for o in broker.submitted if o["side"] == "buy"]
    assert events(tmp_path)[-1]["event"] == "plan_abandoned"
    assert router.store.data.history[-1]["traded_usd"] != "0.00", "what the sells did trade still counts"

    clock.set("2027-01-05", 16, 30)                      # the Allocator's evening decision
    answer = await router.receive_plan(await proposal(router, "2027-01-05"))
    assert answer["status"] in ("approved", "awaiting_approval")
    plan = core_paper.plan_from_json(router.store.data.plan["plan"])
    assert plan.buys and not plan.sells                  # the sells are done; only the buys are left
    assert router.store.data.plan["redecision"] is True
    if answer["status"] == "awaiting_approval":
        await router.approve(answer["plan_id"])
    clock.set("2027-01-06", 9, 22)
    await router.tick()
    broker.fill_all("buy")
    clock.set("2027-01-06", 9, 40)
    await router.tick()
    assert broker.cash < idle / 10, "the idle cash from the crashed rebalance is back in the market"


@pytest.mark.asyncio
async def test_a_plan_still_working_after_the_cutoff_is_abandoned_not_traded_late(tmp_path: Path) -> None:
    router, broker, clock = await at_the_sells(tmp_path)
    broker.fill_all("sell")
    clock.set("2027-01-04", 15, 45)                      # the process comes back mid-afternoon
    await router.tick()
    assert router.store.data.plan is None and router.store.data.forced is True
    assert not [o for o in broker.submitted if o["side"] == "buy"], "no new orders late in the day"
    assert broker.cancel_calls >= 1, "anything still working is cancelled"


@pytest.mark.asyncio
async def test_a_plan_whose_buys_were_placed_but_never_checked_is_closed_the_next_day(tmp_path: Path) -> None:
    router, broker, clock = await at_the_sells(tmp_path)
    broker.fill_all("sell")
    clock.set("2027-01-04", 9, 40)
    await router.tick()                                  # the buys go in
    assert router.store.data.plan["status"] == "buying"
    clock.set("2027-01-05", 9, 25)                       # the process was down all day; the day orders expired
    for o in broker.orders.values():
        if o["side"] == "buy" and o["status"] == "accepted":
            o["status"] = "expired"
    await router.tick()
    assert router.store.data.plan is None and router.store.data.forced is True


@pytest.mark.asyncio
async def test_buys_blocked_by_the_breaker_after_the_open_abandon_the_plan_instead_of_failing_every_tick(
        tmp_path: Path) -> None:
    """Before the open, equity still equals the last close, so the breaker cannot see the day's gap: it can only
    trip once the sells have filled. That used to raise out of ``tick()`` on every call until the process was
    restarted."""
    router, broker, clock = await at_the_sells(tmp_path)
    broker.fill_all("sell")
    router.guard.breaker._cached = None
    broker.equity = D("9700")                            # the market gapped down 3%
    clock.set("2027-01-04", 9, 33)
    await router.tick()                                  # must not raise
    assert router.store.data.plan is None and router.store.data.forced is True
    assert not [o for o in broker.submitted if o["side"] == "buy"]
    assert {"buy_blocked", "plan_abandoned"} <= {e["event"] for e in events(tmp_path)}


@pytest.mark.asyncio
async def test_sells_that_never_fill_leave_drift_that_is_redecided_at_the_close(tmp_path: Path) -> None:
    router, broker, clock = await at_the_sells(tmp_path)
    clock.set("2027-01-04", 10, 5)                       # 35 minutes after the open: the sell deadline has passed
    await router.tick()
    assert broker.cancel_calls >= 1
    clock.set("2027-01-04", 11, 5)
    await router.tick()                                  # whatever buys followed are resolved at their own deadline
    assert router.store.data.plan is None
    assert router.store.data.forced is True, "an unfilled sell leaves drift: re-decide at the close"


# --- orders that are refused or fail in transit ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_order_alpaca_refuses_is_recorded_and_does_not_freeze_the_plan(tmp_path: Path) -> None:
    broker, clock = FlakyBroker(), Clock(START)
    router = await built(tmp_path, broker, clock)
    broker.reject_sells = {"VTI"}
    await rally_and_submit_sells(router, broker, clock)
    plan = router.store.data.plan
    assert plan["status"] == "selling", "a refused sell must not leave the plan retrying until the window shuts"
    (refused,) = plan["orders"].values()
    assert refused["status"] == "rejected"
    assert "order_rejected" in {e["event"] for e in events(tmp_path)}
    clock.set("2027-01-04", 11, 5)
    await router.tick()
    clock.set("2027-01-04", 11, 10)
    await router.tick()
    assert router.store.data.plan is None and router.store.data.forced is True


@pytest.mark.asyncio
async def test_a_transient_failure_while_submitting_resumes_without_duplicating_orders(tmp_path: Path) -> None:
    broker, clock = FlakyBroker(), Clock(START)
    router = await built(tmp_path, broker, clock)
    for symbol in ("VTI", "VXUS"):
        broker.closes[symbol] = D("140")                 # two over-weights: two sells
    broker.submitted.clear()
    router.store.data.history.clear()
    SESSIONS[:] = QUARTER_END
    clock.set("2026-12-31", 16, 30)
    answer = await router.receive_plan(await proposal(router, "2026-12-31"))
    if answer["status"] == "awaiting_approval":
        await router.approve(answer["plan_id"])
    sells = len(core_paper.plan_from_json(router.store.data.plan["plan"]).sells)
    assert sells >= 2
    broker.fail_transient_after = 1                      # the second order fails in transit
    clock.set("2027-01-04", 9, 22)
    with pytest.raises(AlpacaError):
        await router.tick()
    await router.tick()                                  # the next tick, 15 s later, resumes
    placed = [o["client_order_id"] for o in broker.submitted if o["side"] == "sell"]
    assert len(placed) == len(set(placed)) == sells, "every sell exactly once"
    assert router.store.data.plan["status"] == "selling"


# --- proposals --------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sending_the_same_proposal_again_does_not_undo_the_owners_approval(tmp_path: Path) -> None:
    router = make_router(tmp_path, FlakyBroker(), clock=Clock(START))
    first = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(first["plan_id"], fund=True)
    assert router.store.data.plan["status"] == "approved"
    again = await router.receive_plan(await proposal(router, "2026-09-30"))   # a cron retry
    assert again["status"] == "approved" and again["plan_id"] == first["plan_id"]
    assert router.store.data.plan["status"] == "approved"


@pytest.mark.asyncio
async def test_a_proposal_cannot_replace_a_plan_that_is_already_trading(tmp_path: Path) -> None:
    router, broker, clock = await at_the_sells(tmp_path)
    broker.fill_all("sell")
    clock.set("2027-01-04", 10, 0)
    with pytest.raises(PlanRejected, match="plan_in_flight"):
        await router.receive_plan(await proposal(router, "2026-12-31"))
    assert router.store.data.plan["status"] == "selling"


@pytest.mark.asyncio
async def test_the_allocators_state_call_closes_a_stale_plan_so_it_decides_from_the_truth(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from risk_router.core_app import create_core_app
    from swarm.core_allocator import signed_headers
    secret = "p" * 40
    monkeypatch.setenv("CORE_PLAN_SECRET", secret)
    broker, clock = FlakyBroker(), Clock(START)
    app = create_core_app(alpaca=broker, background=False, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=clock)
    async with app.router.lifespan_context(app):
        assert app.state.disabled is None
        router = app.state.router
        await build_portfolio(router, broker, clock)
        await rally_and_submit_sells(router, broker, clock)
        broker.fill_all("sell")
        clock.set("2027-01-05", 16, 30)                  # the router never ticked since: the plan is stale
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://core") as http:
            facts = (await http.get("/v1/core/state", headers=signed_headers(b"", secret))).json()
    assert facts["forced"] is True and facts["plan"] is None


# --- an incomplete first build ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_first_build_that_only_half_fills_can_be_finished_past_the_turnover_cap(tmp_path: Path) -> None:
    broker, clock = FlakyBroker(), Clock(START)
    router = make_router(tmp_path, broker, clock=clock)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    clock.set("2026-10-01", 9, 25)
    await router.tick()
    filled = 0
    for o in broker.orders.values():                     # only three of the six buys fill
        if o["side"] == "buy" and filled < 3:
            price = broker.closes[o["symbol"]]
            qty = (D(o["notional"]) / price).quantize(D("0.000000001"))
            broker.qty[o["symbol"]] = qty
            broker.cash -= qty * price
            o.update(status="filled", filled_qty=str(qty), filled_avg_price=str(price))
            filled += 1
    clock.set("2026-10-01", 11, 5)                       # past the 90-minute buy deadline
    await router.tick()
    assert router.store.data.plan is None
    assert router.store.data.forced is True and router.store.data.initial_incomplete is True

    clock.set("2026-10-01", 16, 30)
    redo = await router.receive_plan(await proposal(router, "2026-10-01"))
    plan = core_paper.plan_from_json(router.store.data.plan["plan"])
    assert sum(plan.buys.values()) / D("10000") > D("0.25"), "the remainder is bigger than the 25% turnover cap"
    assert redo["status"] == "awaiting_approval", "finishing the build needs the owner's signed fund command again"
    assert router.store.data.plan["kind"] == "initial"
    await router.approve(redo["plan_id"], fund=True)
    clock.set("2026-10-02", 9, 25)
    await router.tick()
    broker.fill_all("buy")
    clock.set("2026-10-02", 9, 40)
    await router.tick()
    assert router.store.data.initial_incomplete is False, "a clean finish clears the flag"


# --- the policy's start date --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_plan_that_would_execute_before_the_policy_takes_effect_is_not_made(tmp_path: Path) -> None:
    pol = replace(policy(), effective_from=date(2026, 10, 5))
    router = make_router(tmp_path, FlakyBroker(), pol=pol, clock=Clock(START))
    snap = await core_paper.read_snapshot(router.broker, pol, date(2026, 9, 30), forced=False, targets={})
    assert core_paper.decide(pol, snap) is None
    assert "takes effect" in (core_paper.skip_reason(pol, snap) or "")
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    assert answer["status"] == "no_plan" and "takes effect" in answer["reason"]
    with pytest.raises(PlanRejected, match="not_effective"):
        await router.raise_cash(D("100"), date(2026, 10, 9))

    snap_ok = await core_paper.read_snapshot(router.broker, pol, date(2026, 10, 2), forced=False, targets={})
    assert core_paper.skip_reason(pol, snap_ok) is None and core_paper.decide(pol, snap_ok) is not None


# --- the journal ------------------------------------------------------------------------------------------------------

def test_the_journal_head_is_read_from_the_file_so_losing_the_state_file_does_not_break_the_chain(
        tmp_path: Path) -> None:
    store = CoreStore(tmp_path / "state.json")
    journal = Journal(tmp_path / "journal.jsonl", store, "sha")
    journal.write("one")
    journal.write("two")
    (tmp_path / "state.json").unlink()                   # the state file is lost; the journal survives
    journal2 = Journal(tmp_path / "journal.jsonl", CoreStore(tmp_path / "state.json"), "sha")
    journal2.write("three")
    assert Journal.verify(tmp_path / "journal.jsonl")
    assert journal2.status()["ok"] is True


def test_a_journal_cut_short_is_detected_even_though_the_remaining_chain_is_intact(tmp_path: Path) -> None:
    store = CoreStore(tmp_path / "state.json")
    journal = Journal(tmp_path / "journal.jsonl", store, "sha")
    for name in ("one", "two", "three"):
        journal.write(name)
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")   # the last entry is deleted
    assert Journal.verify(path), "the shortened chain is still internally consistent..."
    status = Journal(path, CoreStore(tmp_path / "state.json"), "sha").status()
    assert status["ok"] is False and "disagree" in status["reason"]


def test_the_app_refuses_to_trade_on_a_tampered_journal_and_reports_it(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    import tier0_core
    from fastapi.testclient import TestClient

    from risk_router.core_app import create_core_app
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    path = policy_file(tmp_path)
    store = CoreStore(tmp_path / "core-plan-state.json")
    journal = Journal(tmp_path / "core-journal.jsonl", store, tier0_core.load_policy(path).sha256)
    for name in ("one", "two", "three"):
        journal.write(name)
    lines = (tmp_path / "core-journal.jsonl").read_text().splitlines()
    (tmp_path / "core-journal.jsonl").write_text("\n".join([lines[0], lines[2]]) + "\n")   # an entry removed
    app = create_core_app(alpaca=FakeBroker(), background=False, state_dir=tmp_path, policy_path=path)
    with TestClient(app) as client:
        health = client.get("/health").json()
    assert health["trading_enabled"] is False and "journal" in health["disabled_reason"]
    assert health["journal"]["ok"] is False


# --- alerts ----------------------------------------------------------------------------------------------------------------

def test_which_events_are_worth_a_push() -> None:
    from risk_router.core_alerts import should_alert
    assert should_alert("plan_abandoned", {})
    assert should_alert("plan_mismatch", {})
    assert should_alert("plan_accepted", {"status": "awaiting_approval"})
    assert not should_alert("plan_accepted", {"status": "approved"})
    assert not should_alert("order_submitted", {})
    assert not should_alert("plan_done", {"orders": {"a": {"status": "filled"}}})
    assert should_alert("plan_done", {"orders": {"a": {"status": "filled"}, "b": {"status": "canceled"}}})


@pytest.mark.asyncio
async def test_an_abandoned_plan_is_pushed(tmp_path: Path) -> None:
    alerter = RecordingAlerter()
    router, broker, clock = await at_the_sells(tmp_path, alerter=alerter)
    broker.fill_all("sell")
    clock.set("2027-01-05", 9, 25)
    await router.tick()
    assert any("abandoned" in title for title, _ in alerter.sent)


@pytest.mark.asyncio
async def test_a_plan_that_needs_the_owner_is_announced(tmp_path: Path) -> None:
    alerter = RecordingAlerter()
    router = make_router(tmp_path, FlakyBroker(), clock=Clock(START))
    router.alerter = alerter
    await router.receive_plan(await proposal(router, "2026-09-30"))
    assert any("approval" in title.lower() for title, _ in alerter.sent)


@pytest.mark.asyncio
async def test_the_alerter_posts_json_and_never_raises() -> None:
    import httpx

    from risk_router.core_alerts import Alerter
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    ok = await Alerter("https://hooks.example/core", transport=httpx.MockTransport(handler)).send("T", "body")
    assert ok and json.loads(seen[0].content)["text"].startswith("T")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert await Alerter("https://hooks.example/core", transport=httpx.MockTransport(boom)).send("T", "b") is False
    assert await Alerter(None).send("T", "b") is False
    Alerter(None).notify("T", "b")                       # no URL: a no-op


@pytest.mark.asyncio
async def test_the_ntfy_format_sends_plain_text_with_a_title() -> None:
    import httpx

    from risk_router.core_alerts import Alerter
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200))[1])
    await Alerter("https://ntfy.sh/mytopic", "ntfy", transport=transport).send("Plan abandoned", "details")
    assert seen[0].content == b"details" and seen[0].headers["Title"] == "Plan abandoned"


def test_the_alerter_is_configured_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from risk_router.core_alerts import Alerter
    monkeypatch.delenv("CORE_ALERT_WEBHOOK_URL", raising=False)
    assert Alerter.from_env().url is None
    monkeypatch.setenv("CORE_ALERT_WEBHOOK_URL", "https://ntfy.sh/topic")
    monkeypatch.setenv("CORE_ALERT_FORMAT", "ntfy")
    alerter = Alerter.from_env()
    assert alerter.url == "https://ntfy.sh/topic" and alerter.fmt == "ntfy"


# --- liveness ---------------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_tick_failures_are_counted_journaled_and_cleared_on_success(tmp_path: Path) -> None:
    router = make_router(tmp_path, FlakyBroker(), clock=Clock(START))
    router.alerter = RecordingAlerter()

    async def failing() -> None:
        raise RuntimeError("alpaca down")

    router.tick = failing  # type: ignore[method-assign]
    for _ in range(3):
        await router.run_tick()
    assert router.tick_failures == 3 and router.last_tick_at is None
    assert "tick_failed" in {e["event"] for e in events(tmp_path)}

    async def fine() -> None:
        return None

    router.tick = fine  # type: ignore[method-assign]
    await router.run_tick()
    assert router.tick_failures == 0 and router.last_tick_at is not None


# --- concurrent callers ---------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_callers_reconciling_at_once_close_the_plan_exactly_once(tmp_path: Path) -> None:
    """The background tick and an HTTP call (the Allocator's /state, an approval) can reach a stale plan together.
    Both used to abandon it: two history entries (double-counting the week's traded value) and two alerts."""
    import asyncio

    class Slow(FlakyBroker):
        async def cancel_all_orders(self) -> int:
            await asyncio.sleep(0.01)                    # an await point inside the abandon, where the callers interleave
            return await super().cancel_all_orders()

    alerter = RecordingAlerter()
    router, broker, clock = await at_the_sells(tmp_path, Slow(), alerter)
    broker.fill_all("sell")
    clock.set("2027-01-05", 9, 25)
    await asyncio.gather(router.reconcile(), router.reconcile(), router.tick())
    assert [e["event"] for e in events(tmp_path)].count("plan_abandoned") == 1
    assert [h["status"] for h in router.store.data.history].count("abandoned") == 1
    assert sum("abandoned" in title for title, _ in alerter.sent) == 1


@pytest.mark.asyncio
async def test_a_plan_with_orders_already_sent_is_in_flight_whatever_its_status_says(tmp_path: Path) -> None:
    """While ``_start`` is part-way through its sells the status is still 'approved'; a proposal arriving then
    must not supersede it and orphan the orders already at Alpaca."""
    router, broker, clock = await at_the_sells(tmp_path)
    router.store.data.plan["status"] = "approved"        # as it is between two sells of a submission in progress
    assert router.store.data.plan["orders"]
    clock.set("2027-01-04", 9, 25)
    with pytest.raises(PlanRejected, match="plan_in_flight"):
        await router.receive_plan(await proposal(router, "2026-12-31"))
    with pytest.raises(PlanRejected, match="plan_in_flight"):
        await router.raise_cash(D("100"), date(2027, 1, 8))


# --- raising cash --------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_raise_cash_request_during_the_session_is_told_to_wait_for_the_close(tmp_path: Path) -> None:
    """Asked at 10:00, the last published close is yesterday's and the plan it would make executes at *today's*
    open, which is already over: it used to be accepted and then expire the moment anyone looked at it."""
    broker, clock = FlakyBroker(), Clock(START)
    router = await built(tmp_path, broker, clock)        # the build filled on 2026-10-01
    clock.set("2026-10-02", 10, 0)
    with pytest.raises(PlanRejected, match="ask_after_the_close"):
        await router.raise_cash(D("400"), date(2026, 10, 9))
    assert router.store.data.plan is None


@pytest.mark.asyncio
async def test_a_raise_cash_request_after_the_close_uses_that_days_close(tmp_path: Path) -> None:
    broker, clock = FlakyBroker(), Clock(START)
    router = await built(tmp_path, broker, clock)
    clock.set("2026-10-01", 16, 30)                      # after 16:20: the 10-01 close is published
    answer = await router.raise_cash(D("400"), date(2026, 10, 9))
    assert router.store.data.plan["decided_on"] == "2026-10-01" and answer["execute_on"] == "2026-10-02"


# --- one router per state directory --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_second_router_on_the_same_state_directory_refuses_to_trade(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leftover process plus a restarted container is the classic way to place every order twice."""
    from risk_router.core_app import create_core_app
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    path = policy_file(tmp_path)
    first = create_core_app(alpaca=FakeBroker(), background=False, state_dir=tmp_path, policy_path=path)
    second = create_core_app(alpaca=FakeBroker(), background=False, state_dir=tmp_path, policy_path=path)
    async with first.router.lifespan_context(first):
        assert first.state.disabled is None
        async with second.router.lifespan_context(second):
            assert second.state.disabled and "already running" in second.state.disabled
            assert first.state.disabled is None
    third = create_core_app(alpaca=FakeBroker(), background=False, state_dir=tmp_path, policy_path=path)
    async with third.router.lifespan_context(third):
        assert third.state.disabled is None, "the lock is released when the first router stops"


@pytest.mark.asyncio
async def test_the_reason_trading_is_disabled_has_no_stray_newline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Alpaca's error bodies end in a newline; it showed up in /health and the pre-flight when a key was rejected."""
    from risk_router.core_app import create_core_app

    class Rejected(FakeBroker):
        async def account(self) -> dict[str, Any]:
            raise AlpacaError('GET /v2/account → HTTP 401: {"message": "unauthorized."}\n', status_code=401)

    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    app = create_core_app(alpaca=Rejected(), background=False, state_dir=tmp_path, policy_path=policy_file(tmp_path))
    async with app.router.lifespan_context(app):
        assert app.state.disabled == 'GET /v2/account → HTTP 401: {"message": "unauthorized."}'
