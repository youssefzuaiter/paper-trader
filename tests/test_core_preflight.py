"""The pre-flight check, the Allocator's behaviour on a day with no session, and the health endpoint's account of
what needs a person. All of it read-only against the same fake account the other core tests use."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_core_paper import Clock, FakeBroker, make_router, policy, policy_file, proposal

from risk_router.alpaca_async import AlpacaError
from risk_router.core_gatekeeper import CoreStore, Journal
from risk_router.core_preflight import FAIL, PASS, WARN, Check, render, run_checks

D = Decimal
# Repeated characters on purpose: zero entropy, so no secret scanner mistakes a fixture for a real key.
GOOD_ENV = {"CORE_ALPACA_KEY_ID": "k" * 20, "CORE_ALPACA_SECRET_KEY": "s" * 30,
            "CORE_PLAN_SECRET": "a" * 40, "WEBHOOK_SECRET": "b" * 40, "CORE_ROUTER_URL": "http://127.0.0.1:8080",
            "CORE_ALERT_WEBHOOK_URL": "https://ntfy.sh/some-topic"}
EVENING = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)     # 17:00 in New York: the 2026-09-30 close is published


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


async def preflight(tmp_path: Path, broker: Any = None, *, env: dict[str, str] | None = None,
                    pol: Any = None, now: datetime = EVENING) -> list[Check]:
    return await run_checks(broker or FakeBroker(), pol or policy(), env=GOOD_ENV if env is None else env,
                            state_dir=tmp_path / "state", now=now)


@pytest.mark.asyncio
async def test_a_correctly_set_up_empty_account_is_ready_and_shows_the_first_plan(tmp_path: Path) -> None:
    broker = FakeBroker()
    checks = by_name(await preflight(tmp_path, broker))
    assert {c.status for c in checks.values()} == {PASS}, render(list(checks.values()))
    assert "buy $1899.81 VTI" in checks["dry run"].detail and "waits for your signed approval" in checks["dry run"].detail
    assert broker.submitted == [], "a preflight never places an order"
    assert "READY" in render(list(checks.values())) and "NOT READY" not in render(list(checks.values()))


@pytest.mark.asyncio
async def test_the_keys_of_another_account_are_a_failure(tmp_path: Path) -> None:
    checks = by_name(await preflight(tmp_path, FakeBroker(account="PA0NEWS00002")))
    assert checks["account"].status == FAIL and "PA0NEWS00002" in checks["account"].detail


@pytest.mark.asyncio
async def test_an_account_left_at_the_default_hundred_thousand_is_a_failure(tmp_path: Path) -> None:
    checks = by_name(await preflight(tmp_path, FakeBroker(cash="100000")))
    assert checks["funding"].status == FAIL and "reset the paper account" in checks["funding"].detail


@pytest.mark.asyncio
async def test_a_position_outside_the_policy_is_a_failure(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.qty = {"TSLA": D("3")}
    assert by_name(await preflight(tmp_path, broker))["holdings"].status == FAIL


@pytest.mark.asyncio
async def test_secrets_are_checked_by_name_and_never_printed(tmp_path: Path) -> None:
    tiny = "q" * 12
    env = {**GOOD_ENV, "CORE_PLAN_SECRET": tiny, "WEBHOOK_SECRET": "b" * 40}
    checks = await preflight(tmp_path, env=env)
    rendered = render(checks)
    assert by_name(checks)["signing secrets"].status == FAIL and "CORE_PLAN_SECRET" in rendered
    for value in (tiny, "b" * 40, GOOD_ENV["CORE_ALPACA_SECRET_KEY"]):
        assert value not in rendered


@pytest.mark.asyncio
async def test_the_allocators_secret_must_differ_from_the_owners(tmp_path: Path) -> None:
    """Otherwise a compromised Allocator could sign approvals and halts as well as proposals."""
    checks = by_name(await preflight(tmp_path, env={**GOOD_ENV, "WEBHOOK_SECRET": GOOD_ENV["CORE_PLAN_SECRET"]}))
    assert checks["secrets are distinct"].status == FAIL


@pytest.mark.asyncio
async def test_missing_keys_stop_the_check_before_any_network_call(tmp_path: Path) -> None:
    class Untouchable:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"the broker must not be used without keys ({name})")

    env = {k: v for k, v in GOOD_ENV.items() if not k.startswith("CORE_ALPACA")}
    checks = by_name(await run_checks(Untouchable(), policy(), env=env, state_dir=tmp_path / "s", now=EVENING))
    assert checks["alpaca keys"].status == FAIL and "account" not in checks


@pytest.mark.asyncio
async def test_a_revoked_key_is_reported_cleanly_not_as_a_traceback(tmp_path: Path) -> None:
    class Revoked(FakeBroker):
        async def account(self) -> dict[str, Any]:
            raise AlpacaError("GET /v2/account → HTTP 401: unauthorized", status_code=401)

    checks = by_name(await preflight(tmp_path, Revoked()))
    assert checks["alpaca"].status == FAIL and "wrong, or revoked" in checks["alpaca"].detail


@pytest.mark.asyncio
async def test_a_missing_alert_webhook_is_advice_not_a_blocker(tmp_path: Path) -> None:
    env = {k: v for k, v in GOOD_ENV.items() if k != "CORE_ALERT_WEBHOOK_URL"}
    checks = by_name(await preflight(tmp_path, env=env))
    assert checks["alerts"].status == WARN
    assert "NOT READY" not in render(list(checks.values()))


@pytest.mark.asyncio
async def test_the_website_mirror_is_optional_and_a_mistake_in_it_is_advice_not_a_blocker(tmp_path: Path) -> None:
    unset = by_name(await preflight(tmp_path))
    assert unset["website mirror"].status == PASS and "optional" in unset["website mirror"].detail

    url = "https://pfw.example/api/webhooks/core"
    good = by_name(await preflight(tmp_path, env={**GOOD_ENV, "CORE_PFW_SYNC_URL": url}))
    assert good["website mirror"].status == PASS and url in good["website mirror"].detail

    plain = by_name(await preflight(tmp_path, env={**GOOD_ENV, "CORE_PFW_SYNC_URL": "http://pfw.example/api/webhooks/core"}))
    assert plain["website mirror"].status == WARN and "https" in plain["website mirror"].detail
    assert "NOT READY" not in render(list(plain.values()))        # advice: it cannot stop the core trading

    weak = by_name(await preflight(tmp_path, env={**GOOD_ENV, "CORE_PFW_SYNC_URL": url, "WEBHOOK_SECRET": "short"}))
    assert weak["website mirror"].status == WARN and "WEBHOOK_SECRET" in weak["website mirror"].detail
    assert "short" not in weak["website mirror"].detail.replace("shorter", "")     # names the variable, never the value


@pytest.mark.asyncio
async def test_a_policy_that_starts_after_the_next_session_is_flagged(tmp_path: Path) -> None:
    pol = replace(policy(), effective_from=date(2026, 10, 5))
    checks = by_name(await preflight(tmp_path, pol=pol))
    assert checks["calendar"].status == WARN and "2026-10-05" in checks["calendar"].detail
    assert "no plan" in checks["dry run"].detail and "takes effect" in checks["dry run"].detail


@pytest.mark.asyncio
async def test_the_dry_run_fails_when_the_router_would_reject_the_plan(tmp_path: Path) -> None:
    broker = FakeBroker(cash="0")
    broker.qty = {"VTI": D("100")}                       # everything in one fund: the rebalance breaks the turnover cap
    broker.qty.update({s: D(0) for s in policy().symbols if s != "VTI"})
    pol = policy()
    # make it a due day so the Allocator proposes: the last session of the quarter
    broker.closes = {s: D("100") for s in pol.symbols}
    checks = by_name(await preflight(tmp_path, broker, now=datetime(2026, 9, 30, 21, 0, tzinfo=UTC)))
    assert checks["dry run"].status == FAIL and "REJECT" in checks["dry run"].detail


@pytest.mark.asyncio
async def test_a_tampered_journal_is_a_failure_and_the_files_are_left_alone(tmp_path: Path) -> None:
    state = tmp_path / "state"
    store = CoreStore(state / "core-plan-state.json")
    journal = Journal(state / "core-journal.jsonl", store, "sha")
    for name in ("one", "two", "three"):
        journal.write(name)
    lines = (state / "core-journal.jsonl").read_text().splitlines()
    (state / "core-journal.jsonl").write_text("\n".join([lines[0], lines[2]]) + "\n")
    before = {p.name: p.read_bytes() for p in state.iterdir()}
    checks = by_name(await preflight(tmp_path))
    assert checks["journal"].status == FAIL
    assert {p.name: p.read_bytes() for p in state.iterdir()} == before, "the preflight only reads the state"


@pytest.mark.asyncio
async def test_a_state_directory_that_does_not_exist_yet_is_not_created(tmp_path: Path) -> None:
    await preflight(tmp_path)
    assert not (tmp_path / "state").exists()


# --- the Allocator on a day with no session -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_allocator_does_nothing_on_a_day_with_no_session(capsys: pytest.CaptureFixture[str]) -> None:
    from swarm.core_allocator import run

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the router must not be called when there is nothing to decide")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://core") as http:
        code = await run(FakeBroker(), http, policy(), date(2026, 10, 3), "p" * 40)     # a Saturday
    assert code == 0 and json.loads(capsys.readouterr().out)["status"] == "skipped"


# --- what /health says needs a person ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_lists_a_plan_waiting_for_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from risk_router.core_app import create_core_app
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    broker, clock = FakeBroker(), Clock(EVENING)
    app = create_core_app(alpaca=broker, background=False, state_dir=tmp_path, policy_path=policy_file(tmp_path), now=clock)
    async with app.router.lifespan_context(app):
        calm = await _health(app)
        assert calm["status"] == "ok" and calm["attention"] == [] and calm["journal"]["ok"] is True
        await app.state.router.receive_plan(await proposal(app.state.router, "2026-09-30"))
        busy = await _health(app)
    assert busy["status"] == "attention"
    assert any("waiting for your signed approval" in line for line in busy["attention"])


@pytest.mark.asyncio
async def test_health_notices_when_the_execution_loop_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from risk_router.core_app import create_core_app
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    clock = Clock(EVENING)
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=clock)
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.05)                       # the loop's first tick
        assert (await _health(app))["attention"] == []
        clock.at = clock.at + timedelta(minutes=5)      # five minutes pass with no further tick
        stale = await _health(app)
    assert any("stopped ticking" in line for line in stale["attention"])


async def _health(app: Any) -> dict[str, Any]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://core") as http:
        return (await http.get("/health")).json()


# --- two more edges in the router -----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_plan_whose_window_passed_while_the_process_was_down_expires_the_same_day(tmp_path: Path) -> None:
    clock = Clock(EVENING)
    router = make_router(tmp_path, FakeBroker(), clock=clock)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    clock.set("2026-10-01", 9, 40)                       # the window (09:20-09:27) is gone and nothing was sent
    await router.tick()
    assert router.store.data.plan is None and router.store.data.forced is True


@pytest.mark.asyncio
async def test_an_approval_that_arrives_after_the_window_finds_the_plan_closed(tmp_path: Path) -> None:
    from risk_router.core_gatekeeper import PlanRejected
    clock = Clock(EVENING)
    router = make_router(tmp_path, FakeBroker(), clock=clock)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    clock.set("2026-10-01", 9, 40)
    with pytest.raises(PlanRejected, match="no_such_plan"):
        await router.approve(answer["plan_id"], fund=True)
    assert router.store.data.forced is True


@pytest.mark.asyncio
async def test_asking_for_more_cash_than_the_portfolio_could_raise_is_a_clean_refusal(tmp_path: Path) -> None:
    from risk_router.core_gatekeeper import PlanRejected
    router = make_router(tmp_path, FakeBroker(), clock=Clock(EVENING))
    with pytest.raises(PlanRejected, match="exceeds_portfolio"):
        await router.raise_cash(D("1000000"), date(2026, 10, 9))


def test_a_crash_between_appending_an_entry_and_saving_the_anchor_is_not_tampering(tmp_path: Path) -> None:
    store = CoreStore(tmp_path / "state.json")
    journal = Journal(tmp_path / "journal.jsonl", store, "sha")
    journal.write("one")
    anchor_before = store.data.journal_head
    journal.write("two")
    store.data.journal_head = anchor_before              # the process died before the second save
    store.save()
    fresh = Journal(tmp_path / "journal.jsonl", CoreStore(tmp_path / "state.json"), "sha")
    assert fresh.status(repair=False)["ok"] is True
    assert fresh.store.data.journal_head == anchor_before, "repair=False leaves the file as it found it"
    assert fresh.status()["ok"] is True and fresh.store.data.journal_head == fresh._head
