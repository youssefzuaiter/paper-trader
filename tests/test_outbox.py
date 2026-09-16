"""outbox.py — the durable retry queue for receipts PFW never acknowledged.

Every test drives a real ``Outbox`` against a temporary directory with a
scripted ``send_once`` — the file format, the atomic rewrite, the
backoff arithmetic and the dead-letter rule are what's under test, not
HTTP (``webhook.send_once`` is a thin httpx wrapper exercised live).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import outbox
from outbox import Outbox, SendOutcome, backoff_seconds, entry_id


def _receipt(key: str = "order-1") -> dict[str, Any]:
    return {"schema_version": 1, "idempotency_key": key, "status": "pending", "symbol": "GOOGL"}


class ScriptedSender:
    """Returns the scripted outcomes in order; the last one repeats."""

    def __init__(self, *outcomes: SendOutcome) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, url: str, receipt: dict[str, Any]) -> SendOutcome:
        self.calls.append((url, receipt))
        if len(self.outcomes) > 1:
            return self.outcomes.pop(0)
        return self.outcomes[0]


DELIVERED = SendOutcome(delivered=True, permanent=False, status_code=201)
TRANSIENT = SendOutcome(delivered=False, permanent=False, error="ConnectError: refused")
PERMANENT = SendOutcome(delivered=False, permanent=True, error="HTTP 400: bad schema", status_code=400)


@pytest.fixture
def box(tmp_path: Path) -> Outbox:
    return Outbox(tmp_path / "outbox")


@pytest.mark.asyncio
async def test_enqueue_persists_to_disk_and_is_idempotent_per_receipt(box: Outbox) -> None:
    await box.enqueue("trade", "http://pfw/api/webhooks/trades", _receipt(), "ConnectError", now=1000.0)
    await box.enqueue("trade", "http://pfw/api/webhooks/trades", _receipt(), "HTTP 503", now=1001.0)

    assert box.pending_count() == 1
    lines = box.pending_path.read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["id"] == entry_id("trade", _receipt())
    assert stored["last_error"] == "HTTP 503"  # updated in place, not duplicated
    assert stored["attempts"] == 0
    assert stored["next_attempt_at"] == 1000.0 + outbox.BACKOFF_BASE_SECONDS


@pytest.mark.asyncio
async def test_survives_a_restart(box: Outbox) -> None:
    await box.enqueue("metrics", "http://pfw/api/webhooks/metrics", _receipt("m-1"), "timeout", now=1000.0)
    reopened = Outbox(box.directory)  # a fresh process reading the same file
    assert [entry.id for entry in reopened.pending_entries()] == ["metrics:m-1"]


@pytest.mark.asyncio
async def test_fails_twice_then_delivers_exactly_once_and_is_removed(box: Outbox) -> None:
    await box.enqueue("trade", "http://pfw/t", _receipt(), "refused", now=0.0)
    sender = ScriptedSender(TRANSIENT, TRANSIENT, DELIVERED)

    # Not due yet: first replay is scheduled BACKOFF_BASE_SECONDS after enqueue.
    stats = await box.replay_pending(sender, now=1.0)
    assert stats.attempted == 0 and stats.still_pending == 1

    stats = await box.replay_pending(sender, now=outbox.BACKOFF_BASE_SECONDS)
    assert stats.attempted == 1 and stats.delivered == 0 and stats.still_pending == 1
    entry = box.pending_entries()[0]
    assert entry.attempts == 1
    assert entry.next_attempt_at == outbox.BACKOFF_BASE_SECONDS + backoff_seconds(1)

    stats = await box.replay_pending(sender, now=entry.next_attempt_at)
    assert stats.attempted == 1 and stats.still_pending == 1
    entry = box.pending_entries()[0]
    assert entry.attempts == 2

    stats = await box.replay_pending(sender, now=entry.next_attempt_at)
    assert stats.delivered == 1 and stats.still_pending == 0
    assert box.pending_count() == 0
    assert len(sender.calls) == 3
    assert all(call[1] == _receipt() for call in sender.calls)  # identical body every replay
    assert not box.dead_letter_path.exists()


@pytest.mark.asyncio
async def test_permanent_rejection_goes_to_dead_letter_not_retried(box: Outbox) -> None:
    await box.enqueue("settlement", "http://pfw/t", _receipt("s-1"), "refused", now=0.0)
    sender = ScriptedSender(PERMANENT)

    stats = await box.replay_pending(sender, now=outbox.BACKOFF_BASE_SECONDS)
    assert stats.dead_lettered == 1 and stats.still_pending == 0
    assert box.pending_count() == 0
    dead = [json.loads(line) for line in box.dead_letter_path.read_text().strip().splitlines()]
    assert dead[0]["id"] == "settlement:s-1"
    assert dead[0]["reason"] == "HTTP 400: bad schema"
    assert len(sender.calls) == 1


@pytest.mark.asyncio
async def test_entries_older_than_max_age_are_dead_lettered_without_a_send(box: Outbox) -> None:
    await box.enqueue("trade", "http://pfw/t", _receipt("old"), "refused", now=0.0)
    sender = ScriptedSender(DELIVERED)

    stats = await box.replay_pending(sender, now=outbox.MAX_AGE_SECONDS + 1)
    assert stats.dead_lettered == 1 and stats.attempted == 0
    assert sender.calls == []
    assert box.pending_count() == 0


@pytest.mark.asyncio
async def test_a_sender_exception_counts_as_transient_and_never_kills_the_pass(box: Outbox) -> None:
    await box.enqueue("trade", "http://pfw/t", _receipt("boom"), "refused", now=0.0)

    async def exploding_sender(url: str, receipt: dict[str, Any]) -> SendOutcome:
        raise RuntimeError("socket on fire")

    stats = await box.replay_pending(exploding_sender, now=outbox.BACKOFF_BASE_SECONDS)
    assert stats.attempted == 1 and stats.still_pending == 1
    entry = box.pending_entries()[0]
    assert entry.attempts == 1
    assert "socket on fire" in entry.last_error


@pytest.mark.asyncio
async def test_unreadable_line_is_skipped_not_fatal(box: Outbox) -> None:
    await box.enqueue("trade", "http://pfw/t", _receipt("good"), "refused", now=0.0)
    with box.pending_path.open("a", encoding="utf-8") as handle:
        handle.write("{this is not json\n")
    assert [entry.id for entry in box.pending_entries()] == ["trade:good"]


def test_backoff_is_capped() -> None:
    assert [backoff_seconds(n) for n in (0, 1, 2, 3, 4, 5, 20)] == [30.0, 30.0, 60.0, 120.0, 240.0, 300.0, 300.0]
