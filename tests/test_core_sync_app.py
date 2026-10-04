"""The website mirror inside the running router: started with it, never in its way, and honest on ``/health``."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_core_paper import Clock, FakeBroker, policy, policy_file
from test_core_preflight import _health
from test_core_sync import SECRET, URL, FakePfw

import webhook
from risk_router import core_sync
from risk_router.core_app import create_core_app, health_payload

EVENING = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)


def a_sync(tmp_path: Path, pfw: FakePfw, app_holder: dict[str, Any]) -> core_sync.CoreSync:
    return core_sync.CoreSync(
        settings=core_sync.SyncSettings(url=URL, secret=SECRET),
        journal_path=tmp_path / core_sync.JOURNAL_FILE, state_path=tmp_path / core_sync.STATE_FILE,
        status=lambda: health_payload(app_holder["app"].state), broker=FakeBroker(), policy=policy(),
        transport=httpx.MockTransport(pfw.handler))


async def wait_for(condition: Any, seconds: float = 5.0) -> None:
    for _ in range(int(seconds / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting")


@pytest.mark.asyncio
async def test_the_router_mirrors_its_own_journal_and_health_to_the_website(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    pfw, holder = FakePfw(), {}
    sync = a_sync(tmp_path, pfw, holder)
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=Clock(EVENING), pfw_sync=sync)
    holder["app"] = app
    async with app.router.lifespan_context(app):
        await wait_for(lambda: pfw.report_requests())
        health = await _health(app)

        journal = pfw.journal_requests()[0]
        assert [json.loads(e["raw"])["event"] for e in journal["entries"]] == ["started"]       # what start-up wrote
        report = pfw.report_requests()[0]
        assert report["status"]["trading_enabled"] is True and report["status"]["attention"] == []
        assert report["status"]["policy_sha256"] == app.state.policy.sha256     # the one the router actually loaded
        for request in pfw.requests:
            assert webhook.verify(request.content, request.headers[webhook.TIMESTAMP_HEADER],
                                  request.headers[webhook.SIGNATURE_HEADER], SECRET)

    assert health["pfw_sync"]["enabled"] is True and health["pfw_sync"]["last_success_at"] is not None
    assert health["pfw_sync"]["pending_entries"] == 0 and health["pfw_sync"]["stuck"] is False
    assert SECRET not in json.dumps(health)
    assert health["status"] == "ok"


@pytest.mark.asyncio
async def test_a_router_that_cannot_trade_still_tells_the_website_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    pfw, holder = FakePfw(), {}
    sync = a_sync(tmp_path, pfw, holder)
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path,
                          policy_path=policy_file(tmp_path, account="UNSET"), now=Clock(EVENING), pfw_sync=sync)
    holder["app"] = app
    async with app.router.lifespan_context(app):
        await wait_for(lambda: pfw.report_requests())
    status = pfw.report_requests()[0]["status"]
    assert status["trading_enabled"] is False and "account is not set" in status["disabled_reason"]
    assert any("trading is disabled" in line for line in status["attention"])
    assert status["policy_sha256"] == app.state.policy.sha256       # it loaded, and failed the hard limits


@pytest.mark.asyncio
async def test_a_router_that_never_got_as_far_as_a_journal_still_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No policy file at all: no router object, no journal, and the website is still told, in the wire's own terms."""
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    pfw, holder = FakePfw(), {}
    sync = a_sync(tmp_path, pfw, holder)
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path,
                          policy_path=tmp_path / "missing.toml", now=Clock(EVENING), pfw_sync=sync)
    holder["app"] = app
    async with app.router.lifespan_context(app):
        await wait_for(lambda: pfw.report_requests())
    status = pfw.report_requests()[0]["status"]
    assert status["trading_enabled"] is False and status["policy_sha256"] is None
    assert status["journal"] == {"ok": False, "entries": 0, "reason": "the router has no journal: it did not start"}
    assert pfw.journal_requests() == []                              # there is no journal to send


@pytest.mark.asyncio
async def test_a_website_that_is_down_does_not_slow_or_stop_the_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)

    def dead(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    holder: dict[str, Any] = {}
    sync = core_sync.CoreSync(settings=core_sync.SyncSettings(url=URL, secret=SECRET),
                              journal_path=tmp_path / core_sync.JOURNAL_FILE, state_path=tmp_path / core_sync.STATE_FILE,
                              status=lambda: health_payload(holder["app"].state), broker=FakeBroker(), policy=policy(),
                              transport=httpx.MockTransport(dead))
    clock = Clock(EVENING)
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=clock, pfw_sync=sync)
    holder["app"] = app
    async with app.router.lifespan_context(app):
        await wait_for(lambda: sync.failures >= 1)
        await asyncio.sleep(0.05)                       # the router's own first tick
        health = await _health(app)
    assert health["trading_enabled"] is True and health["tick"]["failures"] == 0
    assert health["pfw_sync"]["consecutive_failures"] >= 1 and "ConnectError" in health["pfw_sync"]["last_error"]
    assert health["attention"] == []                    # a blip is not news; only a long failure is


@pytest.mark.asyncio
async def test_a_misconfigured_mirror_is_reported_and_leaves_trading_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    monkeypatch.setenv("CORE_PFW_SYNC_URL", URL)
    monkeypatch.setenv("WEBHOOK_SECRET", "short")
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=Clock(EVENING))
    async with app.router.lifespan_context(app):
        health = await _health(app)
    assert health["trading_enabled"] is True and health["pfw_sync"] == {"enabled": False}
    assert any("website mirror is misconfigured" in line and "WEBHOOK_SECRET" in line for line in health["attention"])


@pytest.mark.asyncio
async def test_no_url_means_no_mirror_and_no_change_to_anything_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    monkeypatch.delenv("CORE_PFW_SYNC_URL", raising=False)
    app = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=Clock(EVENING))
    async with app.router.lifespan_context(app):
        health = await _health(app)
    assert app.state.pfw_sync is None and health["pfw_sync"] == {"enabled": False}
    assert health["status"] == "ok" and health["attention"] == []
    assert not (tmp_path / core_sync.STATE_FILE).exists()


@pytest.mark.asyncio
async def test_a_second_router_on_the_same_state_does_not_mirror_the_journal_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_PLAN_SECRET", "p" * 40)
    pfw, holder, other_holder = FakePfw(), {}, {}
    first = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                            now=Clock(EVENING), pfw_sync=a_sync(tmp_path, pfw, holder))
    holder["app"] = first
    second_pfw = FakePfw()
    second = create_core_app(alpaca=FakeBroker(), background=True, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                             now=Clock(EVENING), pfw_sync=a_sync(tmp_path, second_pfw, other_holder))
    other_holder["app"] = second
    async with first.router.lifespan_context(first):
        async with second.router.lifespan_context(second):
            assert second.state.disabled and "already running" in second.state.disabled
            await asyncio.sleep(0.2)
            assert second.state.pfw_sync is None and second_pfw.requests == []
        assert first.state.pfw_sync is not None


def test_health_payload_is_what_the_health_route_returns(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    app = create_core_app(alpaca=FakeBroker(), background=False, state_dir=tmp_path, policy_path=policy_file(tmp_path),
                          now=Clock(EVENING))
    with TestClient(app) as client:
        assert client.get("/health").json() == health_payload(app.state)
