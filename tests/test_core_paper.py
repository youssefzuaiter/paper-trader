"""The long-term core's paper mode (core design §8): policy limits, the shared decision, the router's
verification, approvals, execution at the open, the breaker and kill switch, the journal, accounts."""

from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import replace
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import core_alloc
import core_paper
import tier0_core
from risk_router.core_gatekeeper import CoreRouter, CoreStore, Journal, PlanRejected
from risk_router.guards import CircuitBreaker, ExecutionGuard
from risk_router.state import StateStore

D = Decimal
NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT / "policy" / "core.toml"
SESSIONS = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]


def policy(**changes: Any) -> tier0_core.CorePolicy:
    """The owner's policy with the two values that belong to the owner (the real account number and the
    start date) replaced, so no test breaks the day either is edited."""
    base = replace(tier0_core.load_policy(POLICY_FILE), account="PA0CORE00001", effective_from=date(2026, 1, 1))
    return replace(base, **changes)


def policy_file(tmp_path: Path, *, account: str = "PA0CORE00001", effective_from: str = "2026-01-01") -> Path:
    """The same, as a file for the code that loads one: only ``account`` and ``effective_from`` are replaced."""
    text = re.sub(r'(?m)^account = ".*"$', f'account = "{account}"', POLICY_FILE.read_text())
    text = re.sub(r'(?m)^effective_from = ".*"$', f'effective_from = "{effective_from}"', text)
    path = tmp_path / "core.toml"
    path.write_text(text)
    return path


class FakeBroker:
    """An in-memory paper account: cash, positions, published closes, the calendar, orders that fill
    when the test says so."""

    def __init__(self, *, account: str = "PA0CORE00001", cash: str = "10000", closes: dict[str, str] | None = None):
        self.number = account
        self.cash = D(cash)
        self.equity = D(cash)
        self.last_equity = D(cash)
        self.qty: dict[str, Decimal] = {}
        self.closes = {s: D(c) for s, c in (closes or dict.fromkeys(policy().symbols, "100")).items()}
        self.orders: dict[str, dict[str, Any]] = {}
        self.submitted: list[dict[str, Any]] = []

    async def account(self) -> dict[str, Any]:
        return {"account_number": self.number, "cash": str(self.cash), "equity": str(self.equity),
                "last_equity": str(self.last_equity)}

    async def positions(self) -> list[dict[str, Any]]:
        return [{"symbol": s, "qty": str(q)} for s, q in self.qty.items() if q]

    async def daily_bars(self, symbols: list[str], start: str, end: str) -> dict[str, list[dict[str, Any]]]:
        return {s: [{"t": f"{start}T04:00:00Z", "o": str(self.closes[s]), "c": str(self.closes[s])}] for s in symbols}

    async def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        return [{"date": d} for d in SESSIONS if start <= d <= end]

    async def latest_quote(self, symbol: str) -> Any:
        return type("Q", (), {"ask": self.closes[symbol], "bid": self.closes[symbol]})()

    async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.submitted.append(payload)
        order = {**payload, "status": "accepted", "filled_qty": "0", "filled_avg_price": None}
        self.orders[payload["client_order_id"]] = order
        return order

    async def order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        return self.orders.get(client_order_id)

    async def cancel_all_orders(self) -> int:
        n = 0
        for o in self.orders.values():
            if o["status"] == "accepted":
                o["status"] = "canceled"
                n += 1
        return n

    def fill_all(self, side: str) -> None:
        """Fill every working order of ``side`` at the close, moving cash and positions."""
        for o in self.orders.values():
            if o["side"] != side or o["status"] != "accepted":
                continue
            price = self.closes[o["symbol"]]
            qty = D(o["qty"]) if "qty" in o else (D(o["notional"]) / price).quantize(D("0.000000001"))
            self.qty[o["symbol"]] = self.qty.get(o["symbol"], D(0)) + (qty if side == "buy" else -qty)
            self.cash += -qty * price if side == "buy" else qty * price
            o.update(status="filled", filled_qty=str(qty), filled_avg_price=str(price))


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def set(self, day: str, hh: int, mm: int) -> None:
        self.at = datetime.combine(date.fromisoformat(day), time(hh, mm), NY).astimezone(UTC)


def make_router(tmp_path: Path, broker: FakeBroker, pol: tier0_core.CorePolicy | None = None,
                clock: Clock | None = None) -> CoreRouter:
    pol = pol or policy()
    state = StateStore(tmp_path / "core-router-state.json")
    clock = clock or Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    guard = ExecutionGuard(state, CircuitBreaker(broker, state, now=clock))
    store = CoreStore(tmp_path / "core-plan-state.json")
    return CoreRouter(broker, guard, pol, store, Journal(tmp_path / "journal.jsonl", store, pol.sha256), now=clock)


async def proposal(router: CoreRouter, session: str) -> dict[str, Any]:
    """What an honest Allocator would send (it calls the same shared reader and decision)."""
    data = router.store.data
    snap = await core_paper.read_snapshot(router.broker, router.policy, date.fromisoformat(session),
                                          forced=data.forced, targets={s: D(w) for s, w in data.targets.items()})
    plan = core_paper.decide(router.policy, snap)
    return {"session": session, "inputs": snap.digest(), "plan": core_paper.plan_to_json(plan) if plan else None}


# --- the policy and its limits ------------------------------------------------------------------------------

def test_the_committed_policy_is_inside_every_tier0_limit() -> None:
    """A committed policy that breaks a ceiling would disable trading at start-up; catch it here instead."""
    raw = tier0_core.load_policy(POLICY_FILE)
    assert tier0_core.violations(raw) == []
    assert raw.rule.name == "quarterly" and raw.buffer_symbol == "BIL" and sum(raw.mix.values()) == 1


def test_an_unset_or_malformed_account_fails_closed() -> None:
    raw = tier0_core.load_policy(POLICY_FILE)
    expected = ["account is not set to a paper account number (PA…)"]
    assert tier0_core.violations(replace(raw, account="UNSET")) == expected
    assert tier0_core.violations(replace(raw, account="")) == expected
    assert tier0_core.violations(replace(raw, account="AB123")) == expected


def test_the_policy_is_the_registered_evidence() -> None:
    """The mix must be exactly the registered configuration the owner chose, and cite its run."""
    from backtest import core_grid as G
    weights, _ = G.weights_fn("M3", buffer=D("0.05"))
    assert weights(None) == policy().mix
    cfg = G.Config("buffer", "full", "M3", "quarterly", "central", G.OWNER_SIZE, buffer=D("0.05"))
    assert cfg.run_key == policy().evidence_config


@pytest.mark.parametrize(("change", "fragment"), [
    ({"mix": {"VTI": D("0.6"), "BND": D("0.3")}}, "sum to"),
    ({"mix": {"VTI": D("0.5"), "TSLA": D("0.5")}}, "outside the registered"),
    ({"mix": {"VTI": D("0.85"), "BTC/USD": D("0.15")}, "buffer_symbol": None}, "crypto weight"),
    ({"max_order_usd": D("200000")}, "max_order_usd"),
    ({"max_turnover_pct": D("40")}, "max_turnover"),
    ({"drawdown_ladder": "on"}, "not built"),
    ({"min_order_usd": D("0.5")}, "below Alpaca"),
    ({"buffer_target": D("0.10")}, "buffer BIL"),
])
def test_a_policy_beyond_any_tier0_limit_is_refused(change: dict[str, Any], fragment: str) -> None:
    assert any(fragment in v for v in tier0_core.violations(policy(**change)))


def test_the_decision_uses_core_allocs_functions() -> None:
    """C12's paper half: the shared decision calls the engine's own functions, it does not copy them."""
    calls = {ast.unparse(n.func) for n in ast.walk(ast.parse(inspect.getsource(core_paper.decide)))
             if isinstance(n, ast.Call)}
    assert {"core_alloc.rebalance_due", "core_alloc.plan_rebalance", "core_alloc.current_weights"} <= calls


# --- the decision --------------------------------------------------------------------------------------------

def snap(**kw: Any) -> core_paper.Snapshot:
    pol = policy()
    base = {"session": date(2026, 9, 30), "next_session": date(2026, 10, 1), "qty": dict.fromkeys(pol.symbols, D(0)),
            "cash": D("10000"), "closes": dict.fromkeys(pol.symbols, D("100")), "initial": True}
    return core_paper.Snapshot(**{**base, **kw})


def test_the_initial_build_buys_the_policy_mix_with_the_reserve_kept() -> None:
    plan = core_paper.decide(policy(), snap())
    assert plan is not None and not plan.sells and plan.reason == "initial"
    assert plan.buys["VTI"] == D("1899.81") and plan.buys["BIL"] == D("499.95")  # 19% / 5% of $9,999


def test_quarterly_rebalances_only_at_a_quarters_last_close_or_when_forced() -> None:
    pol = policy()
    drifted = dict.fromkeys(pol.symbols, D("19"))
    drifted["VTI"], drifted["BIL"] = D("25"), D("5")
    mid = snap(session=date(2026, 10, 14), next_session=date(2026, 10, 15), qty=drifted, cash=D(0), initial=False)
    assert core_paper.decide(pol, mid) is None
    assert core_paper.decide(pol, replace(mid, forced=True)) is not None
    end = replace(mid, session=date(2026, 9, 30), next_session=date(2026, 10, 1))
    plan = core_paper.decide(pol, end)
    assert plan is not None and "VTI" in plan.sells


# --- verification and approvals -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_honest_proposal_is_admitted_and_the_build_waits_for_the_fund_command(tmp_path: Path) -> None:
    broker = FakeBroker()
    router = make_router(tmp_path, broker)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    assert answer["status"] == "awaiting_approval" and answer["execute_on"] == "2026-10-01"
    with pytest.raises(PlanRejected, match="fund"):
        await router.approve(answer["plan_id"])
    assert (await router.approve(answer["plan_id"], fund=True))["status"] == "approved"


@pytest.mark.asyncio
async def test_a_build_on_an_over_funded_account_is_refused(tmp_path: Path) -> None:
    broker = FakeBroker(cash="100000")  # Alpaca's default paper balance, not the owner's $10,000
    router = make_router(tmp_path, broker)
    with pytest.raises(PlanRejected, match="over_funded"):
        await router.receive_plan(await proposal(router, "2026-09-30"))


@pytest.mark.asyncio
async def test_a_proposal_that_differs_from_the_recomputation_is_rejected(tmp_path: Path) -> None:
    router = make_router(tmp_path, FakeBroker())
    honest = await proposal(router, "2026-09-30")
    forged = json.loads(json.dumps(honest))
    forged["plan"]["buys"]["VTI"] = "5000.00"
    with pytest.raises(PlanRejected, match="plan_mismatch"):
        await router.receive_plan(forged)
    with pytest.raises(PlanRejected, match="plan_mismatch"):
        await router.receive_plan({**honest, "inputs": "0" * 64})
    lines = (tmp_path / "journal.jsonl").read_text().splitlines()
    assert sum(json.loads(line)["event"] == "plan_mismatch" for line in lines) == 2


@pytest.mark.asyncio
async def test_a_rebalance_over_the_turnover_cap_is_rejected(tmp_path: Path) -> None:
    broker = FakeBroker(cash="0")
    broker.qty = {"VTI": D("100")}  # everything in one fund: a rebalance would trade 80% of it
    broker.qty.update({s: D(0) for s in policy().symbols if s != "VTI"})
    router = make_router(tmp_path, broker)
    with pytest.raises(PlanRejected, match="turnover"):
        await router.receive_plan(await proposal(router, "2026-09-30"))


# --- execution -------------------------------------------------------------------------------------------------

async def built(tmp_path: Path, broker: FakeBroker, clock: Clock) -> CoreRouter:
    router = make_router(tmp_path, broker, clock=clock)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    clock.set("2026-10-01", 9, 25)
    await router.tick()
    broker.fill_all("buy")
    clock.set("2026-10-01", 9, 40)
    await router.tick()
    return router


@pytest.mark.asyncio
async def test_the_initial_build_goes_in_pre_open_as_collared_notional_limits(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = await built(tmp_path, broker, clock)
    buys = [o for o in broker.submitted if o["side"] == "buy"]
    assert len(buys) == 6 and all(o["type"] == "limit" and o["time_in_force"] == "day" for o in buys)
    assert all(D(o["limit_price"]) == D("103.00") for o in buys)  # the close + the 3% collar
    assert sum(D(o["notional"]) for o in buys) <= D("9999")
    assert router.store.data.plan is None and router.store.data.history[0]["day"] == "2026-10-01"
    assert router.store.data.targets == {s: str(w) for s, w in policy().mix.items()}
    assert Journal.verify(tmp_path / "journal.jsonl")


@pytest.mark.asyncio
async def test_a_rebalance_sells_at_the_open_then_buys_with_the_cash_it_has(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = await built(tmp_path, broker, clock)
    broker.closes["VTI"] = D("140")  # VTI rallies; at the quarter's end it is over-weight
    broker.submitted.clear()
    router.store.data.history.clear()  # this test is about execution, not the monthly cap
    sessions_backup = list(SESSIONS)
    SESSIONS[:] = ["2026-12-30", "2026-12-31", "2027-01-04"]
    try:
        clock.set("2026-12-31", 16, 30)
        answer = await router.receive_plan(await proposal(router, "2026-12-31"))
        if answer["status"] == "awaiting_approval":
            await router.approve(answer["plan_id"])
        clock.set("2027-01-04", 9, 22)
        await router.tick()
        sells = [o for o in broker.submitted if o["side"] == "sell"]
        assert [o["symbol"] for o in sells] == ["VTI"] and D(sells[0]["limit_price"]) == D("135.80")  # 140 − 3%
        assert not [o for o in broker.submitted if o["side"] == "buy"]  # buys wait for the sells
        broker.fill_all("sell")
        clock.set("2027-01-04", 9, 33)
        await router.tick()
        buys = [o for o in broker.submitted if o["side"] == "buy"]
        assert buys and sum(D(o["notional"]) for o in buys) <= broker.cash
        broker.fill_all("buy")
        clock.set("2027-01-04", 9, 50)
        await router.tick()
        assert router.store.data.plan is None
    finally:
        SESSIONS[:] = sessions_backup


@pytest.mark.asyncio
async def test_the_breaker_defers_a_plan_that_buys_whole(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = make_router(tmp_path, broker, clock=clock)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    broker.equity = D("9700")  # −3% on the day: the −2.5% breaker trips
    clock.set("2026-10-01", 9, 25)
    await router.tick()
    assert broker.submitted == [] and router.store.data.plan is None and router.store.data.forced is True


@pytest.mark.asyncio
async def test_the_kill_switch_blocks_everything(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = make_router(tmp_path, broker, clock=clock)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    router.guard.halt("test")
    clock.set("2026-10-01", 9, 25)
    await router.tick()
    assert broker.submitted == [] and router.store.data.forced is False


@pytest.mark.asyncio
async def test_a_plan_that_misses_its_session_expires_and_is_redecided(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = make_router(tmp_path, broker, clock=clock)
    await router.receive_plan(await proposal(router, "2026-09-30"))
    clock.set("2026-10-02", 9, 25)  # the owner never sent the fund command
    await router.tick()
    assert router.store.data.plan is None and router.store.data.forced is True


@pytest.mark.asyncio
async def test_raise_cash_is_a_sells_only_plan_that_waits_for_approval_and_passes_the_breaker(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = await built(tmp_path, broker, clock)
    clock.set("2026-10-01", 16, 30)
    answer = await router.raise_cash(D("400"), date(2026, 10, 9))
    assert answer["status"] == "awaiting_approval"
    plan = core_paper.plan_from_json(router.store.data.plan["plan"])
    assert not plan.buys and set(plan.sells) == {"BIL"}  # the buffer first
    await router.approve(answer["plan_id"])
    broker.equity = D("9000")  # a bad day: sells that raise cash still go out
    clock.set("2026-10-02", 9, 25)
    await router.tick()
    assert [o["side"] for o in broker.submitted[-1:]] == ["sell"]


# --- accounts and the journal -----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_router_refuses_any_account_but_the_policys(tmp_path: Path) -> None:
    with pytest.raises(PlanRejected, match="wrong_account"):
        await make_router(tmp_path, FakeBroker(account="PA0NEWS00002")).check_account()
    await make_router(tmp_path, FakeBroker()).check_account()


@pytest.mark.asyncio
async def test_the_news_router_refuses_the_core_account(tmp_path: Path) -> None:
    from risk_router.app import CoreAccountRefused, refuse_core_account
    path = policy_file(tmp_path)
    with pytest.raises(CoreAccountRefused):
        await refuse_core_account(FakeBroker(), path)
    await refuse_core_account(FakeBroker(account="PA0NEWS00002"), path)
    # The committed policy names the real core account: the news router must refuse exactly that one.
    real = tier0_core.load_policy(POLICY_FILE).account
    with pytest.raises(CoreAccountRefused):
        await refuse_core_account(FakeBroker(account=real), POLICY_FILE)
    await refuse_core_account(FakeBroker(account="PA0NEWS00002"), POLICY_FILE)
    (tmp_path / "unset").mkdir()
    unset = policy_file(tmp_path / "unset", account="UNSET")
    await refuse_core_account(FakeBroker(), unset)  # core account not set yet: nothing to refuse


@pytest.mark.asyncio
async def test_the_journal_is_a_hash_chain(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    await built(tmp_path, broker, clock)
    path = tmp_path / "journal.jsonl"
    assert Journal.verify(path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + lines[3:]) + "\n")  # delete one entry
    assert not Journal.verify(path)


def test_shortfall_against_the_open_and_the_modelled_fill() -> None:
    s = core_paper.shortfall_bps("buy", D("100.05"), D("100"), D("0.0002"), D("0.0002"))
    assert s["vs_open_bps"] == D("5.0000") and s["vs_modelled_bps"] < s["vs_open_bps"]
    assert core_paper.shortfall_bps("sell", D("99.95"), D("100"), D(0), D(0))["vs_open_bps"] == D("5.0000")


def test_core_alloc_has_the_shared_due_rule() -> None:
    rule = core_alloc.RULES["quarterly"]
    w, t = {"VTI": D("0.3")}, {"VTI": D("0.19")}
    assert core_alloc.rebalance_due(rule, frozenset({"month", "quarter"}), w, t)
    assert core_alloc.rebalance_due(rule, frozenset(), w, t, forced=True)
    assert not core_alloc.rebalance_due(rule, frozenset({"month"}), w, t)


# --- the app and the Allocator, end to end ------------------------------------------------------------------------

def _signed(body: bytes, secret: str) -> dict[str, str]:
    import time as _time

    import webhook
    ts = str(int(_time.time()))
    return {webhook.TIMESTAMP_HEADER: ts, webhook.SIGNATURE_HEADER: webhook.sign(body, ts, secret),
            "Content-Type": "application/json"}


def test_the_app_fails_closed_until_the_account_is_set_and_halt_still_works(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from risk_router.core_app import create_core_app
    plan_secret, control_secret = "p" * 40, "c" * 40
    monkeypatch.setenv("CORE_PLAN_SECRET", plan_secret)
    monkeypatch.setenv("WEBHOOK_SECRET", control_secret)
    monkeypatch.setenv("WEBHOOK_URL", "https://example.invalid/receipts")
    app = create_core_app(alpaca=FakeBroker(), background=False, state_dir=tmp_path,
                          policy_path=policy_file(tmp_path, account="UNSET"))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["trading_enabled"] is False and "account is not set" in health["disabled_reason"]
        body = b"{}"
        assert client.post("/v1/core/plans", content=body, headers=_signed(body, plan_secret)).status_code == 503
        assert client.post("/v1/core/plans", content=body).status_code == 403  # unsigned: refused first
        halted = client.post("/v1/control/halt", content=b"", headers=_signed(b"", control_secret))
        assert halted.status_code == 200 and halted.json()["halted"] is True


@pytest.mark.asyncio
async def test_the_allocators_signed_proposal_is_admitted_by_the_router(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from risk_router.core_app import create_core_app
    from swarm.core_allocator import propose
    secret = "p" * 40
    monkeypatch.setenv("CORE_PLAN_SECRET", secret)
    path = policy_file(tmp_path)
    broker = FakeBroker()
    app = create_core_app(alpaca=broker, background=False, state_dir=tmp_path, policy_path=path)
    async with app.router.lifespan_context(app):
        assert app.state.disabled is None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://core") as http:
            answer = await propose(broker, http, tier0_core.load_policy(path), date(2026, 9, 30), secret)
    assert answer["http_status"] == 200 and answer["status"] == "awaiting_approval"
    assert answer["execute_on"] == "2026-10-01"
