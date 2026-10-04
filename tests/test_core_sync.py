"""The long-term core's mirror to PFW (``risk_router/core_sync.py``): the journal as its own outbox, the router's
self-report, and the one promise everything rests on: syncing is read-only and can never hurt trading.

The PFW end is a fake that behaves like the real route (``POST /api/webhooks/core``): it stores entries by index,
answers a replay as a duplicate, a batch beyond what it holds as a gap with where it is, and a different entry at a
held index as a conflict. Requests go through ``httpx.MockTransport``, so the real signing and headers are exercised."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_core_paper import Clock, FakeBroker, built, make_router, policy

import webhook
from risk_router import core_sync
from risk_router.core_gatekeeper import CoreStore, Journal

D = Decimal
SECRET = "t" * 40            # repeated characters on purpose: zero entropy, so no secret scanner mistakes it for a key
URL = "https://pfw.example/api/webhooks/core"
T0 = datetime(2026, 10, 5, 14, 0, tzinfo=UTC)


class Time:
    """A clock the test moves."""

    def __init__(self, at: datetime = T0) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)


class FakePfw:
    """What ``POST /api/webhooks/core`` does, as far as the router can tell."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.stored: dict[str, dict[int, str]] = {}                 # chain id -> index -> hash
        self.reports: list[dict[str, Any]] = []
        self.script: list[Callable[[dict[str, Any]], httpx.Response | None]] = []   # one-shot overrides, oldest first

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.script:
            override = self.script.pop(0)(body)
            if override is not None:
                return override
        if body["kind"] == "report":
            self.reports.append(body)
            return httpx.Response(201, json={"ok": True, "status": "recorded", "snapshot_stored": body["account"] is not None})
        chain = self.stored.setdefault(body["chain_id"], {})
        entries = body["entries"]
        if entries[0]["index"] > len(chain):
            return httpx.Response(409, json={"ok": False, "error": "gap", "next_index": len(chain)})
        new = 0
        for entry in entries:
            held = chain.get(entry["index"])
            if held is not None and held != entry["hash"]:
                return httpx.Response(409, json={"ok": False, "error": "chain_conflict", "index": entry["index"],
                                                 "detail": "the mirror already holds a different entry at this index"})
            if held is None:
                chain[entry["index"]] = entry["hash"]
                new += 1
        return httpx.Response(201 if new else 200, json={"ok": True, "status": "recorded" if new else "duplicate",
                                                          "chain_id": body["chain_id"], "next_index": len(chain)})

    def journal_requests(self) -> list[dict[str, Any]]:
        return [b for b in self.bodies if b["kind"] == "journal"]

    def report_requests(self) -> list[dict[str, Any]]:
        return [b for b in self.bodies if b["kind"] == "report"]


def respond(status: int, body: dict[str, Any]) -> Callable[[dict[str, Any]], httpx.Response]:
    return lambda _request: httpx.Response(status, json=body)


class AccountBroker(FakeBroker):
    """A fake account that reports what Alpaca's real ``/v2/account`` and ``/v2/positions`` do: money as decimal
    strings, a market value and prices on every position."""

    async def account(self) -> dict[str, Any]:
        return {**await super().account(), "equity": "10000.125", "cash": "1.00", "last_equity": "9990.25"}

    async def positions(self) -> list[dict[str, Any]]:
        return [{"symbol": "VTI", "qty": "18.500000000", "market_value": "1900.00", "avg_entry_price": "102.6912",
                 "current_price": "102.70"},
                {"symbol": "BIL", "qty": "5.4", "market_value": "491.40", "avg_entry_price": "91", "current_price": "91.01"}]


def status_ok() -> dict[str, Any]:
    return {"status": "ok", "environment": "paper", "mode": "core", "trading_enabled": True, "disabled_reason": None,
            "halted": False, "halt_reason": None, "policy_sha256": "a" * 64, "policy_effective_from": "2026-10-05",
            "plan": None, "journal": {"ok": True, "entries": 6, "head": "abcdef012345", "anchored": True, "reason": None},
            "tick": {"last_age_seconds": 4.2, "failures": 0}, "attention": [], "limits": {}}


def make_sync(tmp_path: Path, pfw: FakePfw, *, journal_path: Path | None = None, broker: FakeBroker | None = None,
              status: Callable[[], dict[str, Any]] = status_ok, clock: Time | None = None,
              **kwargs: Any) -> core_sync.CoreSync:
    return core_sync.CoreSync(
        settings=core_sync.SyncSettings(url=URL, secret=SECRET),
        journal_path=journal_path or tmp_path / "journal.jsonl",
        state_path=tmp_path / "core-sync-state.json",
        status=status, broker=broker or AccountBroker(), policy=policy(),
        transport=httpx.MockTransport(pfw.handler), clock=clock or Time(), **kwargs)


def write_journal(tmp_path: Path, count: int, *, name: str = "journal.jsonl", start: int = 0) -> Path:
    """A journal of ``count`` entries, written by the router's own ``Journal`` class, so the hashing and the line
    format are the real ones."""
    path = tmp_path / name
    store = CoreStore(tmp_path / f"{name}.state.json")
    journal = Journal(path, store, policy().sha256)
    for i in range(start, start + count):
        journal.write("started" if i == 0 else "no_plan", session=f"2026-10-{(i % 28) + 1:02d}", reason="not a rebalance day")
    return path


def lines_of(path: Path) -> list[str]:
    return [line for line in path.read_text().split("\n") if line]


# --- settings --------------------------------------------------------------------------------------------------

def test_the_sync_is_off_unless_a_url_is_set() -> None:
    assert core_sync.SyncSettings.from_env({}) is None
    assert core_sync.SyncSettings.from_env({"CORE_PFW_SYNC_URL": "   ", "WEBHOOK_SECRET": SECRET}) is None


def test_a_url_with_a_good_secret_is_a_configuration() -> None:
    settings = core_sync.SyncSettings.from_env({"CORE_PFW_SYNC_URL": f" {URL} ", "WEBHOOK_SECRET": f" {SECRET} "})
    assert settings == core_sync.SyncSettings(url=URL, secret=SECRET)


@pytest.mark.parametrize("secret", ["", "short"])
def test_a_url_without_a_usable_secret_is_refused_naming_the_variable(secret: str) -> None:
    with pytest.raises(core_sync.SyncConfigError, match="WEBHOOK_SECRET"):
        core_sync.SyncSettings.from_env({"CORE_PFW_SYNC_URL": URL, "WEBHOOK_SECRET": secret})


def test_plain_http_is_refused_except_for_the_local_machine() -> None:
    with pytest.raises(core_sync.SyncConfigError, match="https"):
        core_sync.SyncSettings.from_env({"CORE_PFW_SYNC_URL": "http://pfw.example/api/webhooks/core", "WEBHOOK_SECRET": SECRET})
    for local in ("http://localhost:3000/api/webhooks/core", "http://127.0.0.1:3000/api/webhooks/core"):
        assert core_sync.SyncSettings.from_env({"CORE_PFW_SYNC_URL": local, "WEBHOOK_SECRET": SECRET}) is not None


def test_a_url_that_is_not_a_url_is_refused() -> None:
    with pytest.raises(core_sync.SyncConfigError):
        core_sync.SyncSettings.from_env({"CORE_PFW_SYNC_URL": "pfw.example/core", "WEBHOOK_SECRET": SECRET})


def test_the_settings_never_show_the_secret() -> None:
    settings = core_sync.SyncSettings(url=URL, secret=SECRET)
    assert SECRET not in repr(settings) and SECRET not in str(settings)


# --- reading the journal ----------------------------------------------------------------------------------------

def test_no_journal_is_an_empty_one_not_a_problem(tmp_path: Path) -> None:
    assert core_sync.read_journal(tmp_path / "nope.jsonl") == core_sync.JournalRead(entries=[], problem=None)


def test_every_line_is_an_entry_with_the_routers_own_hash(tmp_path: Path) -> None:
    path = write_journal(tmp_path, 4)
    read = core_sync.read_journal(path)
    assert read.problem is None and [e.index for e in read.entries] == [0, 1, 2, 3]
    lines = lines_of(path)
    assert [e.raw for e in read.entries] == lines
    assert [e.hash for e in read.entries] == [hashlib.sha256(line.encode()).hexdigest() for line in lines]
    assert [e.prev for e in read.entries] == ["", read.entries[0].hash, read.entries[1].hash, read.entries[2].hash]
    # The last hash is the one the router itself holds as its head: the same bytes, the same function.
    assert read.entries[-1].hash == Journal(path, CoreStore(tmp_path / "other.json"), policy().sha256)._head


def test_a_line_still_being_written_is_left_for_the_next_pass(tmp_path: Path) -> None:
    path = write_journal(tmp_path, 3)
    with path.open("a") as handle:
        handle.write('{"at": "2026-10-05T14:00:00+00:00", "event": "no_pl')       # no newline yet
    read = core_sync.read_journal(path)
    assert read.problem is None and len(read.entries) == 3


def test_blank_lines_are_skipped_as_the_router_skips_them(tmp_path: Path) -> None:
    path = write_journal(tmp_path, 3)
    lines = lines_of(path)
    path.write_text("\n".join([lines[0], "", lines[1], "  ", lines[2]]) + "\n")
    assert [e.index for e in core_sync.read_journal(path).entries] == [0, 1, 2]


def test_an_altered_entry_stops_the_chain_there_and_says_so(tmp_path: Path) -> None:
    path = write_journal(tmp_path, 5)
    lines = lines_of(path)
    lines[2] = lines[2].replace("not a rebalance day", "something else entirely")
    path.write_text("\n".join(lines) + "\n")
    read = core_sync.read_journal(path)
    assert [e.index for e in read.entries] == [0, 1, 2]       # entry 2 itself still links to 1; 3 no longer links to it
    assert read.problem is not None and "3" in read.problem and "chain" in read.problem


def test_a_removed_entry_is_found_the_same_way(tmp_path: Path) -> None:
    path = write_journal(tmp_path, 5)
    lines = lines_of(path)
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    read = core_sync.read_journal(path)
    assert [e.index for e in read.entries] == [0]
    assert read.problem is not None


def test_a_line_that_is_not_json_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    path = write_journal(tmp_path, 2)
    with path.open("a") as handle:
        handle.write("this is not json\n")
    read = core_sync.read_journal(path)
    assert len(read.entries) == 2 and read.problem is not None


# --- the cursor ------------------------------------------------------------------------------------------------

def test_the_cursor_survives_a_restart(tmp_path: Path) -> None:
    store = core_sync.CursorStore(tmp_path / "core-sync-state.json")
    assert store.load() == core_sync.Cursor(chain_id=None, next_index=0)
    store.save(core_sync.Cursor(chain_id="c" * 64, next_index=7))
    assert core_sync.CursorStore(tmp_path / "core-sync-state.json").load() == core_sync.Cursor(chain_id="c" * 64, next_index=7)


@pytest.mark.parametrize("content", ["{not json", "[]", '{"next_index": "seven"}', '{"next_index": -3, "chain_id": 5}'])
def test_a_damaged_cursor_starts_over_rather_than_stopping_the_sync(tmp_path: Path, content: str) -> None:
    path = tmp_path / "core-sync-state.json"
    path.write_text(content)
    assert core_sync.CursorStore(path).load() == core_sync.Cursor(chain_id=None, next_index=0)


# --- the account -------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_account_is_integer_cents_and_decimal_share_counts() -> None:
    account = await core_sync.build_account(AccountBroker(), policy(), T0)
    assert account is not None
    assert account["taken_at"] == "2026-10-05T14:00:00Z"
    assert account["equity_usd_cents"] == 1_000_013          # 10000.125 -> half a cent rounds up
    assert account["cash_usd_cents"] == 100
    assert account["last_equity_usd_cents"] == 999_025
    assert account["positions"][0] == {"symbol": "VTI", "qty": "18.5", "market_value_usd_cents": 190_000,
                                      "avg_entry_price_usd_cents": 10_269, "current_price_usd_cents": 10_270}
    assert account["positions"][1]["avg_entry_price_usd_cents"] == 9_100
    assert all(isinstance(v, int) for k, v in account.items() if k.endswith("_cents") and v is not None)
    assert account["targets"] == {s: str(w) for s, w in policy().mix.items()}


@pytest.mark.asyncio
async def test_a_position_the_broker_gave_no_value_for_is_sent_without_one() -> None:
    class Sparse(AccountBroker):
        async def positions(self) -> list[dict[str, Any]]:
            return [{"symbol": "VTI", "qty": "3"}]

        async def account(self) -> dict[str, Any]:
            return {"equity": "100", "cash": "100"}

    account = await core_sync.build_account(Sparse(), policy(), T0)
    assert account is not None
    assert account["last_equity_usd_cents"] is None
    assert account["positions"] == [{"symbol": "VTI", "qty": "3", "market_value_usd_cents": None,
                                    "avg_entry_price_usd_cents": None, "current_price_usd_cents": None}]


@pytest.mark.asyncio
async def test_an_account_that_cannot_be_read_is_no_account_not_an_error() -> None:
    class Down(AccountBroker):
        async def account(self) -> dict[str, Any]:
            raise RuntimeError("alpaca is down")

    assert await core_sync.build_account(Down(), policy(), T0) is None


@pytest.mark.asyncio
async def test_an_account_with_no_equity_is_no_account() -> None:
    class Odd(AccountBroker):
        async def account(self) -> dict[str, Any]:
            return {"cash": "1"}

    assert await core_sync.build_account(Odd(), policy(), T0) is None


# --- the status ------------------------------------------------------------------------------------------------

def test_the_status_is_the_routers_health_in_the_wire_shape() -> None:
    health = {**status_ok(), "plan": {"id": "a1b2c3d4e5f60718", "kind": "initial", "status": "awaiting_approval",
                                      "execute_on": "2026-10-05", "extra": "ignored"},
              "attention": ["plan a1b2c3d4e5f60718 is waiting for your signed approval"]}
    assert core_sync.wire_status(health) == {
        "trading_enabled": True, "disabled_reason": None, "halted": False, "halt_reason": None,
        "policy_sha256": "a" * 64, "policy_effective_from": "2026-10-05",
        "plan": {"id": "a1b2c3d4e5f60718", "kind": "initial", "status": "awaiting_approval", "execute_on": "2026-10-05"},
        "journal": {"ok": True, "entries": 6, "reason": None},
        "tick_age_seconds": 4.2, "tick_failures": 0,
        "attention": ["plan a1b2c3d4e5f60718 is waiting for your signed approval"]}


def test_a_router_that_never_started_still_has_a_status() -> None:
    health = {**status_ok(), "trading_enabled": False, "disabled_reason": "another core router is already running",
              "policy_sha256": None, "policy_effective_from": None, "journal": None, "tick": None,
              "attention": ["trading is disabled: another core router is already running"]}
    wire = core_sync.wire_status(health)
    assert wire["trading_enabled"] is False and wire["policy_sha256"] is None
    assert wire["journal"] == {"ok": False, "entries": 0, "reason": "the router has no journal: it did not start"}
    assert wire["tick_age_seconds"] is None and wire["tick_failures"] == 0


def test_long_text_is_cut_to_what_pfw_accepts() -> None:
    health = {**status_ok(), "disabled_reason": "x" * 900, "attention": [f"item {i} " + "y" * 900 for i in range(30)]}
    wire = core_sync.wire_status(health)
    assert len(wire["disabled_reason"]) <= 500
    assert len(wire["attention"]) == 20 and all(len(a) <= 500 for a in wire["attention"])


# --- one pass ----------------------------------------------------------------------------------------------------

def verify_signature(request: httpx.Request) -> None:
    timestamp = request.headers[webhook.TIMESTAMP_HEADER]
    assert abs(int(timestamp) - int(datetime.now(UTC).timestamp())) < 300
    assert webhook.verify(request.content, timestamp, request.headers[webhook.SIGNATURE_HEADER], SECRET)


@pytest.mark.asyncio
async def test_a_pass_sends_the_journal_then_a_report_with_the_account_and_signs_both(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 6)
    await make_sync(tmp_path, pfw, journal_path=path).run_once()

    assert [b["kind"] for b in pfw.bodies] == ["journal", "report"]
    for request in pfw.requests:
        verify_signature(request)
        assert request.url == URL
        assert request.headers[webhook.IDEMPOTENCY_HEADER] == json.loads(request.content)["idempotency_key"]
        assert request.content == webhook.canonical_json(json.loads(request.content))      # signed bytes are the sent bytes
    journal = pfw.journal_requests()[0]
    assert journal["schema_version"] == 1
    assert journal["chain_id"] == hashlib.sha256(lines_of(path)[0].encode()).hexdigest()
    assert [e["index"] for e in journal["entries"]] == [0, 1, 2, 3, 4, 5]
    assert journal["entries"][3]["raw"] == lines_of(path)[3]
    report = pfw.report_requests()[0]
    assert report["reported_at"] == "2026-10-05T14:00:00Z" and report["account"]["equity_usd_cents"] == 1_000_013
    assert report["status"]["journal"] == {"ok": True, "entries": 6, "reason": None}


@pytest.mark.asyncio
async def test_the_cursor_moves_only_when_pfw_says_it_has_the_entries(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 4)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    pfw.script.append(respond(503, {"error": "core_mirror_unavailable"}))      # the first journal request fails
    await sync.run_once()
    assert core_sync.CursorStore(tmp_path / "core-sync-state.json").load().next_index == 0
    assert sync.health()["consecutive_failures"] == 1 and sync.health()["pending_entries"] == 4

    await sync.run_once()                                                       # the same entries again, now accepted
    assert core_sync.CursorStore(tmp_path / "core-sync-state.json").load().next_index == 4
    assert sync.health()["consecutive_failures"] == 0 and sync.health()["pending_entries"] == 0
    assert [e["index"] for e in pfw.journal_requests()[-1]["entries"]] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_a_restart_carries_on_from_the_cursor_instead_of_resending(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 4)
    await make_sync(tmp_path, pfw, journal_path=path).run_once()
    write_journal(tmp_path, 2, start=4)                                         # two more entries after the first four
    # (write_journal appends to a fresh Journal object over the same file: the chain continues from the file's tail)
    pfw.bodies.clear()
    await make_sync(tmp_path, pfw, journal_path=path).run_once()                # a new process, the same state files
    assert [[e["index"] for e in b["entries"]] for b in pfw.journal_requests()] == [[4, 5]]


@pytest.mark.asyncio
async def test_only_what_is_new_is_sent_as_the_journal_grows(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 3)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    await sync.run_once()
    pfw.bodies.clear()
    await sync.run_once()
    assert pfw.journal_requests() == []                                         # nothing new: no request at all
    write_journal(tmp_path, 2, start=3)
    await sync.run_once()
    assert [[e["index"] for e in b["entries"]] for b in pfw.journal_requests()] == [[3, 4]]


@pytest.mark.asyncio
async def test_a_long_journal_goes_in_batches_within_one_pass(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 120)
    await make_sync(tmp_path, pfw, journal_path=path).run_once()
    sizes = [len(b["entries"]) for b in pfw.journal_requests()]
    assert sizes == [core_sync.BATCH_SIZE, core_sync.BATCH_SIZE, 120 - 2 * core_sync.BATCH_SIZE]
    assert len(pfw.stored[next(iter(pfw.stored))]) == 120


@pytest.mark.asyncio
async def test_a_mirror_that_is_behind_the_cursor_is_caught_up_not_left_with_a_hole(tmp_path: Path) -> None:
    """PFW's database was restored from a backup: it holds three entries, the router's cursor says six."""
    pfw, path = FakePfw(), write_journal(tmp_path, 6)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    await sync.run_once()
    chain = next(iter(pfw.stored))
    for index in (3, 4, 5):
        del pfw.stored[chain][index]
    write_journal(tmp_path, 1, start=6)                                         # one new entry: the batch starts at 6, PFW is at 3
    pfw.bodies.clear()
    await sync.run_once()
    sent = [[e["index"] for e in b["entries"]] for b in pfw.journal_requests()]
    assert sent[0] == [6] and sent[1] == [3, 4, 5, 6]                           # the gap answer rewound it, and it resent
    assert sorted(pfw.stored[chain]) == [0, 1, 2, 3, 4, 5, 6]
    assert core_sync.CursorStore(tmp_path / "core-sync-state.json").load().next_index == 7


@pytest.mark.asyncio
async def test_two_versions_of_one_entry_stops_the_journal_and_says_so(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 3)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    pfw.script.append(respond(409, {"ok": False, "error": "chain_conflict", "index": 1, "detail": "different entry"}))
    await sync.run_once()
    health = sync.health()
    assert health["stuck"] is True and "chain_conflict" in health["last_error"] and "different entry" in health["last_error"]
    assert core_sync.CursorStore(tmp_path / "core-sync-state.json").load().next_index == 0


@pytest.mark.asyncio
async def test_a_rejected_batch_is_not_retried_in_a_tight_loop(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 3)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    pfw.script.append(respond(400, {"ok": False, "error": "invalid_entry", "detail": "entry 1: bad hash"}))
    await sync.run_once()
    assert sync.health()["stuck"] is True
    assert sync.next_delay() == core_sync.PERMANENT_RETRY_SECONDS
    assert len(pfw.journal_requests()) == 1


@pytest.mark.asyncio
async def test_transient_failures_back_off_and_a_success_resets_it(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 2)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    assert sync.next_delay() == core_sync.SYNC_TICK_SECONDS
    for _ in range(3):
        pfw.script.append(respond(503, {}))
        await sync.run_once()
    assert sync.next_delay() == core_sync.SYNC_TICK_SECONDS * 2 ** 3
    for _ in range(10):
        pfw.script.append(respond(503, {}))
        await sync.run_once()
    assert sync.next_delay() == core_sync.BACKOFF_MAX_SECONDS
    await sync.run_once()
    assert sync.next_delay() == core_sync.SYNC_TICK_SECONDS and sync.health()["consecutive_failures"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "stuck"), [(429, False), (500, False), (502, False), (503, False),
                                               (400, True), (403, True), (404, True), (422, True)])
async def test_only_a_rejection_that_will_not_fix_itself_stops_the_retries(tmp_path: Path, status: int, stuck: bool) -> None:
    """Throttled and server errors come back by themselves; a refusal (the wrong secret, a route that is not there
    yet, a bad request) needs a person, and is retried slowly rather than hammered."""
    pfw = FakePfw()
    sync = make_sync(tmp_path, pfw, journal_path=write_journal(tmp_path, 2))
    pfw.script.append(respond(status, {"error": "whatever"}))
    await sync.run_once()
    assert sync.health()["stuck"] is stuck
    assert sync.next_delay() == (core_sync.PERMANENT_RETRY_SECONDS if stuck else core_sync.SYNC_TICK_SECONDS * 2)


@pytest.mark.asyncio
async def test_the_attention_line_appears_exactly_when_a_failure_has_lasted_the_threshold(tmp_path: Path) -> None:
    pfw, clock = FakePfw(), Time()
    sync = make_sync(tmp_path, pfw, journal_path=write_journal(tmp_path, 2), clock=clock)
    pfw.script.append(respond(503, {}))
    await sync.run_once()
    clock.advance(core_sync.STUCK_AFTER_SECONDS - 1)
    assert sync.attention() is None
    clock.advance(1)
    assert sync.attention() is not None


@pytest.mark.asyncio
async def test_a_dead_network_is_a_failed_pass_not_an_exception(tmp_path: Path) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    sync = core_sync.CoreSync(settings=core_sync.SyncSettings(url=URL, secret=SECRET),
                              journal_path=write_journal(tmp_path, 2), state_path=tmp_path / "s.json",
                              status=status_ok, broker=AccountBroker(), policy=policy(),
                              transport=httpx.MockTransport(refuse), clock=Time())
    await sync.run_once()
    assert sync.health()["consecutive_failures"] == 1 and "ConnectError" in sync.health()["last_error"]


@pytest.mark.asyncio
async def test_a_journal_that_starts_over_starts_a_new_chain_from_the_top(tmp_path: Path) -> None:
    """The state directory was reset: a new journal, a new first entry, so a new chain id and a cursor back at 0."""
    pfw, path = FakePfw(), write_journal(tmp_path, 4)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    await sync.run_once()
    first_chain = next(iter(pfw.stored))
    path.unlink()
    write_journal(tmp_path, 3, name="journal.jsonl")
    # a different first line: the new journal's first entry carries a different timestamp, so it hashes differently
    pfw.bodies.clear()
    await sync.run_once()
    chains = {b["chain_id"] for b in pfw.journal_requests()}
    assert len(chains) == 1 and first_chain not in chains
    assert [e["index"] for e in pfw.journal_requests()[0]["entries"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_a_broken_journal_sends_what_still_verifies_and_reports_the_rest_as_a_problem(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 5)
    lines = lines_of(path)
    lines[2] = lines[2].replace("not a rebalance day", "tampered")
    path.write_text("\n".join(lines) + "\n")
    sync = make_sync(tmp_path, pfw, journal_path=path)
    await sync.run_once()
    assert [e["index"] for b in pfw.journal_requests() for e in b["entries"]] == [0, 1, 2]
    assert sync.health()["journal_problem"] and "chain" in sync.health()["journal_problem"]


# --- the report ---------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reports_come_on_a_schedule_and_the_account_less_often(tmp_path: Path) -> None:
    pfw, clock = FakePfw(), Time()
    sync = make_sync(tmp_path, pfw, journal_path=write_journal(tmp_path, 2), clock=clock)
    await sync.run_once()
    assert len(pfw.report_requests()) == 1 and pfw.report_requests()[0]["account"] is not None

    clock.advance(core_sync.SYNC_TICK_SECONDS)
    await sync.run_once()
    assert len(pfw.report_requests()) == 1                                      # not yet due

    clock.advance(core_sync.REPORT_INTERVAL_SECONDS)
    await sync.run_once()
    assert len(pfw.report_requests()) == 2 and pfw.report_requests()[1]["account"] is None

    clock.advance(core_sync.ACCOUNT_INTERVAL_SECONDS)
    await sync.run_once()
    assert len(pfw.report_requests()) == 3 and pfw.report_requests()[2]["account"] is not None


@pytest.mark.asyncio
async def test_new_journal_entries_bring_a_report_straight_after(tmp_path: Path) -> None:
    pfw, clock = FakePfw(), Time()
    path = write_journal(tmp_path, 2)
    sync = make_sync(tmp_path, pfw, journal_path=path, clock=clock)
    await sync.run_once()
    clock.advance(core_sync.SYNC_TICK_SECONDS)
    write_journal(tmp_path, 1, start=2)
    await sync.run_once()
    assert [b["kind"] for b in pfw.bodies][-2:] == ["journal", "report"]


@pytest.mark.asyncio
async def test_a_plan_closing_brings_the_account_with_the_next_report_but_an_ordinary_entry_does_not(tmp_path: Path) -> None:
    """The account is worth sending the moment a plan closes (the holdings just changed), not at the next hour."""
    pfw, clock = FakePfw(), Time()
    path = write_journal(tmp_path, 2)
    sync = make_sync(tmp_path, pfw, journal_path=path, clock=clock)
    await sync.run_once()                                                      # the first report carries the account
    assert pfw.report_requests()[-1]["account"] is not None

    store = CoreStore(tmp_path / "other.state.json")
    clock.advance(core_sync.SYNC_TICK_SECONDS)
    Journal(path, store, policy().sha256).write("no_plan", session="2026-10-06", reason="not a rebalance day")
    await sync.run_once()
    assert pfw.report_requests()[-1]["account"] is None                         # news, but not an account change

    clock.advance(core_sync.SYNC_TICK_SECONDS)
    Journal(path, store, policy().sha256).write("plan_done", plan_id="a1b2c3d4e5f60718", kind="initial", reason=None, orders={})
    await sync.run_once()
    assert pfw.report_requests()[-1]["account"] is not None                     # a plan closed: send the holdings now
    assert len(pfw.report_requests()) == 3

    clock.advance(core_sync.SYNC_TICK_SECONDS)
    Journal(path, store, policy().sha256).write("no_plan", session="2026-10-07", reason="not a rebalance day")
    await sync.run_once()
    assert pfw.report_requests()[-1]["account"] is None                         # and only that once


@pytest.mark.asyncio
async def test_a_report_that_fails_does_not_hold_back_the_journal_or_the_next_report(tmp_path: Path) -> None:
    pfw, path = FakePfw(), write_journal(tmp_path, 3)
    sync = make_sync(tmp_path, pfw, journal_path=path)
    pfw.script += [lambda body: None, respond(503, {})]          # the journal is accepted, then the report fails
    await sync.run_once()
    assert core_sync.CursorStore(tmp_path / "core-sync-state.json").load().next_index == 3
    assert sync.health()["consecutive_failures"] == 1
    await sync.run_once()
    assert sync.health()["consecutive_failures"] == 0 and len(pfw.reports) == 1


@pytest.mark.asyncio
async def test_a_status_that_cannot_be_read_skips_the_report_but_not_the_journal(tmp_path: Path) -> None:
    def broken() -> dict[str, Any]:
        raise RuntimeError("health blew up")

    pfw, path = FakePfw(), write_journal(tmp_path, 3)
    sync = make_sync(tmp_path, pfw, journal_path=path, status=broken)
    await sync.run_once()
    assert [b["kind"] for b in pfw.bodies] == ["journal"]
    assert "health blew up" in (sync.health()["last_error"] or "")


@pytest.mark.asyncio
async def test_an_account_that_cannot_be_read_still_gets_the_status_through(tmp_path: Path) -> None:
    class Down(AccountBroker):
        async def positions(self) -> list[dict[str, Any]]:
            raise RuntimeError("alpaca is down")

    pfw = FakePfw()
    await make_sync(tmp_path, pfw, journal_path=write_journal(tmp_path, 2), broker=Down()).run_once()
    assert pfw.report_requests()[0]["account"] is None and pfw.report_requests()[0]["status"]["trading_enabled"] is True


# --- the promises ---------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_syncing_never_touches_the_journal_or_the_plan_state(tmp_path: Path) -> None:
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = await built(tmp_path, broker, clock)
    journal_path, state_path = tmp_path / "journal.jsonl", tmp_path / "core-plan-state.json"
    before = (journal_path.read_bytes(), state_path.read_bytes())
    sync = make_sync(tmp_path, FakePfw(), journal_path=journal_path)
    for _ in range(3):
        await sync.run_once()
    assert (journal_path.read_bytes(), state_path.read_bytes()) == before
    assert router.journal.status()["ok"]


@pytest.mark.asyncio
async def test_syncing_does_not_place_orders_or_cancel_them(tmp_path: Path) -> None:
    class Watching(AccountBroker):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.calls.append("submit_order")
            return await super().submit_order(payload)

        async def cancel_all_orders(self) -> int:
            self.calls.append("cancel_all_orders")
            return await super().cancel_all_orders()

    broker = Watching()
    await make_sync(tmp_path, FakePfw(), journal_path=write_journal(tmp_path, 3), broker=broker).run_once()
    assert broker.calls == []


@pytest.mark.asyncio
async def test_the_secret_is_never_logged_or_shown(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pfw = FakePfw()
    pfw.script.append(respond(503, {"echo": "nothing secret here"}))
    sync = make_sync(tmp_path, pfw, journal_path=write_journal(tmp_path, 2))
    with caplog.at_level(logging.DEBUG):
        await sync.run_once()
        await sync.run_once()
    assert SECRET not in caplog.text
    assert SECRET not in json.dumps(sync.health())
    assert all(SECRET not in str(request.url) for request in pfw.requests)


@pytest.mark.asyncio
async def test_run_forever_survives_whatever_a_pass_throws_and_can_be_cancelled(tmp_path: Path) -> None:
    import asyncio

    sync = make_sync(tmp_path, FakePfw(), journal_path=write_journal(tmp_path, 2))
    calls = 0

    async def exploding() -> None:
        nonlocal calls
        calls += 1
        raise ZeroDivisionError("a bug in a pass")

    sync.run_once = exploding            # type: ignore[method-assign]
    sleeps: list[float] = []

    async def instant_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError
        await asyncio.sleep(0)

    sync._sleep = instant_sleep          # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        await sync.run_forever()
    assert calls == 3 and len(sleeps) == 3


def test_the_health_view_says_what_a_person_needs(tmp_path: Path) -> None:
    sync = make_sync(tmp_path, FakePfw(), journal_path=write_journal(tmp_path, 3))
    health = sync.health()
    assert health["enabled"] is True
    assert set(health) >= {"enabled", "last_success_at", "consecutive_failures", "last_error", "stuck",
                           "pending_entries", "journal_problem", "chain", "next_index"}
    assert "url" not in health and URL not in json.dumps(health)      # /health is unauthenticated: it does not name the website
    assert health["last_success_at"] is None and health["stuck"] is False


@pytest.mark.asyncio
async def test_a_sync_that_has_been_failing_a_while_asks_for_attention_and_a_brief_blip_does_not(tmp_path: Path) -> None:
    pfw, clock = FakePfw(), Time()
    sync = make_sync(tmp_path, pfw, journal_path=write_journal(tmp_path, 2), clock=clock)
    await sync.run_once()
    pfw.script.append(respond(503, {}))
    clock.advance(core_sync.REPORT_INTERVAL_SECONDS + 1)
    await sync.run_once()
    assert sync.attention() is None                      # one failure just now: a PFW redeploy, not news
    for _ in range(3):
        pfw.script.append(respond(503, {}))
        clock.advance(core_sync.STUCK_AFTER_SECONDS)
        await sync.run_once()
    message = sync.attention()
    assert message is not None and "website" in message and "503" in message


# --- the wire contract, as PFW sees it ---------------------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "pfw_core_sync_contract.json"


@pytest.mark.asyncio
async def test_the_payloads_are_exactly_what_pfws_route_is_tested_against(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole initial build, driven through the real router with a frozen clock, then sent. The two request bodies
    are compared with ``tests/fixtures/pfw_core_sync_contract.json``, a copy of which lives in PFW's repository
    (``tests/fixtures/core-sync-contract.json``) and is replayed there through ``POST /api/webhooks/core``. Change
    the wire format here and this fails; update the fixture (``UPDATE_PFW_FIXTURE=1``), copy it across, and PFW's
    own test tells you whether its route still accepts it."""
    import os
    from dataclasses import replace

    from test_core_paper import proposal

    from risk_router import core_gatekeeper

    ticks = iter(range(10_000))

    class Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Frozen:        # type: ignore[override]
            return cls(2026, 10, 1, 13, 20, 0, tzinfo=UTC) + timedelta(seconds=30 * next(ticks))

    monkeypatch.setattr(core_gatekeeper, "datetime", Frozen)
    fixed_sha = "4f3c" * 16                                # so editing policy/core.toml does not rewrite the fixture
    broker, clock = FakeBroker(), Clock(datetime(2026, 9, 30, 21, 0, tzinfo=UTC))
    router = make_router(tmp_path, broker, pol=replace(policy(), sha256=fixed_sha), clock=clock)
    router.journal.write("started", account=router.policy.account)
    answer = await router.receive_plan(await proposal(router, "2026-09-30"))
    await router.approve(answer["plan_id"], fund=True)
    clock.set("2026-10-01", 9, 25)
    await router.tick()
    broker.fill_all("buy")
    clock.set("2026-10-01", 9, 40)
    await router.tick()

    pfw = FakePfw()
    sync = make_sync(tmp_path, pfw, journal_path=tmp_path / "journal.jsonl", clock=Time(datetime(2026, 10, 1, 14, 0, tzinfo=UTC)),
                     status=lambda: {**status_ok(), "policy_sha256": fixed_sha,
                                     "journal": {"ok": True, "entries": 10, "head": "x", "anchored": True, "reason": None}})
    await sync.run_once()
    produced = {"journal_body": request_body(pfw, "journal"), "report_body": request_body(pfw, "report")}

    if os.getenv("UPDATE_PFW_FIXTURE"):
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(produced, indent=1, sort_keys=True) + "\n")
    assert FIXTURE.exists(), "run once with UPDATE_PFW_FIXTURE=1 to create it"
    assert produced == json.loads(FIXTURE.read_text())

    journal = json.loads(produced["journal_body"])
    events = [json.loads(e["raw"])["event"] for e in journal["entries"]]
    assert events == ["started", "plan_accepted", "plan_approved"] + ["order_submitted"] * 6 + ["plan_done"]


def request_body(pfw: FakePfw, kind: str) -> str:
    return next(r.content.decode() for r in pfw.requests if json.loads(r.content)["kind"] == kind)
