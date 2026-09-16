"""Durable outbox for signed receipts PFW never acknowledged.

Why this exists
---------------
``webhook._deliver`` retries a failed POST three times over ~1.5s and
then gives up. That was the entire story: a receipt for an order that
had ALREADY been submitted to Alpaca was dropped for good if PFW happened
to be down for two seconds — found live, not hypothetically, when a
GOOGL paper order landed at the broker during a PFW restart and never
reached PFW's ledger. For a system whose whole point is a tamper-evident
ledger of what actually executed, silently losing a fill is the worst
available failure mode.

What it does
------------
Any delivery that ends in a *retryable* failure (transport error, 5xx,
429) is appended here and replayed by ``replay_loop`` every
``REPLAY_INTERVAL_SECONDS`` with capped exponential backoff, until PFW
returns 2xx. A *permanent* failure (any other 4xx — a contract problem
that will not fix itself) goes straight to the dead-letter file, exactly
the rule the inline retry loop already applied.

Two properties the replay must have, both non-obvious:

1. **Re-sign on every attempt.** PFW rejects a signature timestamp older
   than its ``REPLAY_WINDOW_SECONDS`` (300s), so replaying the original
   headers byte-for-byte hours later would fail with 403 forever. The
   BODY is replayed byte-for-byte (``canonical_json`` is deterministic);
   only the timestamp and therefore the MAC are fresh. The receipt's
   ``idempotency_key`` is what lets PFW dedupe a replay of a delivery
   whose 2xx was lost in flight.
2. **Survive a restart.** The queue is a JSONL file, rewritten atomically
   (write a temp file, ``os.replace``) under a lock — ``agent_telemetry``
   is deliberately in-memory and resets on restart; this must not.

Stored receipts contain no secrets (the MAC is recomputed at send time,
never persisted) — but they are the agent's own record of real orders,
so ``outbox/`` is gitignored.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Final, Literal

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_OUTBOX_DIR: Final[Path] = PROJECT_ROOT / "outbox"

REPLAY_INTERVAL_SECONDS: Final[float] = 30.0
BACKOFF_BASE_SECONDS: Final[float] = 30.0
BACKOFF_MAX_SECONDS: Final[float] = 300.0
#: An entry older than this is dead-lettered rather than retried forever —
#: a week of continuous PFW downtime is an outage to investigate by hand,
#: not something to keep hammering.
MAX_AGE_SECONDS: Final[float] = 7 * 24 * 3600.0

ReceiptKind = Literal["trade", "settlement", "metrics"]


@dataclass(frozen=True)
class SendOutcome:
    """The result of exactly one signed POST — see ``webhook.send_once``."""

    delivered: bool
    #: True for a failure that will not fix itself (a 4xx other than 429);
    #: such an entry is dead-lettered instead of retried.
    permanent: bool
    error: str = ""
    status_code: int | None = None


SendOnce = Callable[[str, dict[str, Any]], Awaitable[SendOutcome]]


@dataclass
class OutboxEntry:
    id: str
    kind: str
    url: str
    receipt: dict[str, Any]
    enqueued_at: float
    attempts: int
    next_attempt_at: float
    last_error: str

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> OutboxEntry:
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            url=str(raw["url"]),
            receipt=dict(raw["receipt"]),
            enqueued_at=float(raw["enqueued_at"]),
            attempts=int(raw["attempts"]),
            next_attempt_at=float(raw["next_attempt_at"]),
            last_error=str(raw.get("last_error", "")),
        )


@dataclass(frozen=True)
class ReplayStats:
    attempted: int = 0
    delivered: int = 0
    dead_lettered: int = 0
    still_pending: int = 0


def entry_id(kind: ReceiptKind, receipt: dict[str, Any]) -> str:
    """One entry per (kind, idempotency_key): a receipt that fails twice
    (e.g. queued, replayed, failed again) updates in place rather than
    appearing twice.
    """
    return f"{kind}:{receipt['idempotency_key']}"


def backoff_seconds(attempts: int) -> float:
    """30s, 60s, 120s, 240s, then capped at 300s."""
    return min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** max(0, attempts - 1)))


class Outbox:
    """A file-backed queue. Construct with a directory for tests; the
    process-wide instance is ``default_outbox`` below.
    """

    def __init__(self, directory: Path = DEFAULT_OUTBOX_DIR) -> None:
        self.directory = directory
        self.pending_path = directory / "pending.jsonl"
        self.dead_letter_path = directory / "dead-letter.jsonl"
        self._lock = asyncio.Lock()

    # -- persistence ---------------------------------------------------

    def _load(self) -> list[OutboxEntry]:
        if not self.pending_path.exists():
            return []
        entries: list[OutboxEntry] = []
        with self.pending_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(OutboxEntry.from_json(json.loads(line)))
                except (ValueError, KeyError, TypeError) as exc:
                    # One corrupt line must not take the whole queue down
                    # with it — skip it loudly and keep the rest.
                    logger.error("outbox: skipping unreadable line %d of %s: %s", line_number, self.pending_path, exc)
        return entries

    def _save(self, entries: list[OutboxEntry]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp_path = self.pending_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.pending_path)

    def _dead_letter(self, entry: OutboxEntry, reason: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = asdict(entry) | {"dead_lettered_at": time.time(), "reason": reason}
        with self.dead_letter_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        logger.error(
            "outbox: DEAD-LETTERED %s after %d attempt(s): %s (%s)",
            entry.id, entry.attempts, reason, self.dead_letter_path,
        )

    # -- public API ----------------------------------------------------

    def pending_count(self) -> int:
        return len(self._load())

    def pending_entries(self) -> list[OutboxEntry]:
        return self._load()

    async def enqueue(
        self,
        kind: ReceiptKind,
        url: str,
        receipt: dict[str, Any],
        error: str,
        *,
        now: float | None = None,
    ) -> OutboxEntry:
        """Record an undelivered receipt for replay. Idempotent per entry id."""
        now = time.time() if now is None else now
        async with self._lock:
            entries = self._load()
            target_id = entry_id(kind, receipt)
            for existing in entries:
                if existing.id == target_id:
                    existing.last_error = error
                    self._save(entries)
                    return existing
            entry = OutboxEntry(
                id=target_id,
                kind=kind,
                url=url,
                receipt=receipt,
                enqueued_at=now,
                attempts=0,
                next_attempt_at=now + BACKOFF_BASE_SECONDS,
                last_error=error,
            )
            entries.append(entry)
            self._save(entries)
        logger.warning("outbox: queued %s for replay (%s); %d pending", entry.id, error, len(entries))
        return entry

    async def replay_pending(self, send_once: SendOnce, *, now: float | None = None) -> ReplayStats:
        """One replay pass over every entry that is due. Never raises."""
        now = time.time() if now is None else now
        attempted = delivered = dead_lettered = 0
        async with self._lock:
            entries = self._load()
            remaining: list[OutboxEntry] = []
            for entry in entries:
                if now - entry.enqueued_at > MAX_AGE_SECONDS:
                    self._dead_letter(entry, f"older than {int(MAX_AGE_SECONDS)}s")
                    dead_lettered += 1
                    continue
                if entry.next_attempt_at > now:
                    remaining.append(entry)
                    continue

                attempted += 1
                try:
                    outcome = await send_once(entry.url, entry.receipt)
                except Exception as exc:  # noqa: BLE001 — a replay pass must never die on one entry
                    outcome = SendOutcome(delivered=False, permanent=False, error=f"{type(exc).__name__}: {exc}")

                if outcome.delivered:
                    delivered += 1
                    logger.info("outbox: replayed %s successfully after %d prior attempt(s)", entry.id, entry.attempts)
                    continue
                if outcome.permanent:
                    self._dead_letter(entry, outcome.error)
                    dead_lettered += 1
                    continue
                entry.attempts += 1
                entry.last_error = outcome.error
                entry.next_attempt_at = now + backoff_seconds(entry.attempts)
                remaining.append(entry)

            self._save(remaining)
        stats = ReplayStats(
            attempted=attempted, delivered=delivered, dead_lettered=dead_lettered, still_pending=len(remaining),
        )
        if attempted or dead_lettered:
            logger.info("outbox: replay pass — %s", stats)
        return stats

    async def replay_loop(self, send_once: SendOnce) -> None:
        """Background task: replay due entries every ``REPLAY_INTERVAL_SECONDS``."""
        logger.info("outbox: replay loop started (%d pending)", self.pending_count())
        while True:
            await asyncio.sleep(REPLAY_INTERVAL_SECONDS)
            try:
                await self.replay_pending(send_once)
            except Exception:  # noqa: BLE001 — see replay_pending; belt and braces
                logger.exception("outbox: replay pass failed unexpectedly")


default_outbox: Final[Outbox] = Outbox()
