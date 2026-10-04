"""Mirror the long-term core to PFW (the owner's website), read-only: the journal as its own outbox.

The core router keeps an append-only, hash-chained journal (``core-journal.jsonl``): one JSON line per step, each
carrying the sha256 of the line before it. This module copies it to PFW so the owner can watch the core from there,
and says, every few minutes, whether the router is alive, trading, halted or waiting on anyone.

What it is, and is not
----------------------
* **A copy, never a control.** It reads the journal file and the account, and it sends. It never writes the journal,
  the plan state or the kill switch, never places or cancels an order, and holds no credential PFW could use to do
  either: PFW can show the core, not steer it. (Approving a plan, raising cash and halting are signed commands to
  the router; ``risk_router.core_ctl`` makes them.)
* **Not a receipt feed.** ``receipts_to_pfw`` in the policy is still off: the core's fills are not booked in PFW's
  ledger, which has no notion of a second account. This sends a separate mirror that PFW keeps in tables nothing
  else reads, so the core's paper money cannot mix with the news agent's trades or reach PFW's net worth.
* **Optional, and fails soft.** Off unless ``CORE_PFW_SYNC_URL`` is set. Whatever goes wrong here — PFW down, a
  rejected request, a bug in this file — is logged, counted and retried, and never reaches the router's execution
  loop: it runs as its own task, and ``run_once`` cannot raise.

The journal is the outbox
-------------------------
There is no second queue to keep consistent. The router sends from a **cursor** (the index of the next entry PFW
has not acknowledged, with the id of the journal it belongs to, in ``core-sync-state.json``) and moves it only when
PFW answers that it holds the entries. PFW's answer carries ``next_index``, where *it* is, and the cursor is set from
that: so a PFW database restored from a backup (behind the cursor) is caught up rather than left with a hole, and a
cursor file that is lost or damaged costs one pass of duplicates, which PFW stores once. A journal entry is
identified by its place in its chain, so a replay is harmless by construction.

A journal's id is the sha256 of its first line. A router whose state directory was reset starts a new journal, which
is a new chain: it is sent from its first entry and PFW keeps the old one as history.

Signing
-------
Each request is signed exactly as the trade receipts are (``webhook.sign``: HMAC-SHA256 over ``"{timestamp}." +
body``, the timestamp fresh on every attempt so a replay hours later is not outside PFW's 300-second window) with
the shared ``WEBHOOK_SECRET``. That is the existing trust relationship between this repository and PFW; the mirror
is display data, and a forged request could only mislead what one page shows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import urlparse

import httpx

import webhook

logger = logging.getLogger("risk_router.core_sync")

SCHEMA_VERSION: Final[int] = 1
#: Entries per request. PFW accepts up to 100; half that keeps a request small enough to be quick on a cold start.
BATCH_SIZE: Final[int] = 50
#: A pass drains the journal in batches, but not without limit: a mirror that keeps answering "gap" must not spin.
MAX_BATCHES_PER_PASS: Final[int] = 20
#: How often the loop looks for new journal entries (a cheap file read when there are none).
SYNC_TICK_SECONDS: Final[float] = 30.0
#: How often the router's status is reported, with or without news.
REPORT_INTERVAL_SECONDS: Final[float] = 300.0
#: How often the account (equity, cash, positions) is added to a report; also after a plan closes.
ACCOUNT_INTERVAL_SECONDS: Final[float] = 3600.0
BACKOFF_MAX_SECONDS: Final[float] = 600.0
#: After a rejection that will not fix itself (a 4xx), try again this slowly: PFW may be redeployed with the fix.
PERMANENT_RETRY_SECONDS: Final[float] = 600.0
#: A sync that has been failing this long asks for attention; a PFW redeploy does not.
STUCK_AFTER_SECONDS: Final[float] = 1800.0
HTTP_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(10.0, connect=3.0)
MAX_TEXT: Final[int] = 500
MAX_ATTENTION: Final[int] = 20
#: Journal events that end a plan: the account has changed and is worth sending now rather than at the next hour.
PLAN_CLOSING_EVENTS: Final[frozenset[str]] = frozenset(
    {"plan_done", "plan_abandoned", "plan_expired", "plan_deferred", "plan_halted"})
LOCAL_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})
JOURNAL_FILE: Final[str] = "core-journal.jsonl"
STATE_FILE: Final[str] = "core-sync-state.json"


class SyncConfigError(ValueError):
    """``CORE_PFW_SYNC_URL`` is set but cannot be used."""


@dataclass(frozen=True)
class SyncSettings:
    url: str
    secret: str = field(repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SyncSettings | None:
        """``None`` when the sync is not configured (no URL). A URL with a secret that cannot sign is an error:
        silently not syncing would look exactly like a working sync that has nothing to say."""
        env = os.environ if env is None else env
        url = env.get("CORE_PFW_SYNC_URL", "").strip()
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SyncConfigError("CORE_PFW_SYNC_URL must be a full URL, such as https://your-site/api/webhooks/core")
        if parsed.scheme == "http" and parsed.hostname not in LOCAL_HOSTS:
            raise SyncConfigError("CORE_PFW_SYNC_URL must be https:// (plain http is allowed only for localhost)")
        secret = env.get("WEBHOOK_SECRET", "").strip()
        if len(secret) < 32:
            raise SyncConfigError("CORE_PFW_SYNC_URL is set but WEBHOOK_SECRET is missing or shorter than 32 characters")
        return cls(url=url, secret=secret)

    def display_url(self) -> str:
        """The URL without any credentials someone may have put in it."""
        parsed = urlparse(self.url)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}{parsed.path}"


# --- reading the journal ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class JournalEntry:
    index: int
    hash: str       # sha256 of ``raw``: the same function the router's own Journal uses
    prev: str
    raw: str        # the line exactly as the router wrote it


@dataclass(frozen=True)
class JournalRead:
    entries: list[JournalEntry]
    #: Why the entries after these cannot be trusted (the chain breaks, a line is not an entry); None if all is well.
    problem: str | None


def read_journal(path: Path) -> JournalRead:
    """Every complete line of the journal, in order, as far as it verifies as a hash chain.

    The router appends a line and its newline in one write, so a line without its newline is one still being
    written and is left for the next pass. The chain is checked as it is read: if an entry does not continue the one
    before it, the entries up to it are returned and the rest is a ``problem`` — a partial history that verifies is
    worth showing, and the router's own check is what stops it trading on a journal that does not."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return JournalRead(entries=[], problem=None)
    except OSError as exc:
        return JournalRead(entries=[], problem=f"the journal could not be read: {exc}")

    entries: list[JournalEntry] = []
    head = ""
    for raw in text.split("\n")[:-1]:      # the last piece is "" or a line not yet finished
        if not raw.strip():
            continue
        try:
            prev = json.loads(raw)["prev"]
        except (ValueError, KeyError, TypeError):
            return JournalRead(entries, f"entry {len(entries)} is not a journal entry")
        if prev != head:
            return JournalRead(entries, f"entry {len(entries)}: the hash chain is broken")
        head = hashlib.sha256(raw.encode()).hexdigest()
        entries.append(JournalEntry(index=len(entries), hash=head, prev=prev, raw=raw))
    return JournalRead(entries, None)


# --- the cursor ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cursor:
    chain_id: str | None
    next_index: int


class CursorStore:
    """Where the sync has got to. Losing it is cheap (PFW stores a repeated entry once and says where it is), so a
    damaged or unwritable file is a log line, never a stopped sync."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Cursor:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            chain_id, next_index = raw["chain_id"], raw["next_index"]
            if (chain_id is None or isinstance(chain_id, str)) and isinstance(next_index, int) \
                    and not isinstance(next_index, bool) and next_index >= 0:
                return Cursor(chain_id, next_index)
        except FileNotFoundError:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            logger.warning("core sync: the cursor file %s is unreadable; starting from the first entry", self.path)
        return Cursor(chain_id=None, next_index=0)

    def save(self, cursor: Cursor) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".core-sync-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"chain_id": cursor.chain_id, "next_index": cursor.next_index}, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("core sync: could not save the cursor to %s: %s", self.path, exc)


# --- the account and the status, in the wire shape -----------------------------------------------------------------

def _cents(value: Any) -> int | None:
    """A dollar amount Alpaca sent as a decimal string, as whole cents, half up. Never a float on the way."""
    if value is None or value == "":
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    if not amount.is_finite():
        return None
    return int((amount * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _quantity(value: Any) -> str | None:
    """A share count as a plain decimal string of at most nine places (Alpaca's own precision)."""
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    if not amount.is_finite():
        return None
    return format(amount.quantize(Decimal("0.000000001")).normalize(), "f")


def _clip(value: Any) -> str | None:
    return None if value is None else str(value)[:MAX_TEXT]


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AccountReads(Protocol):
    async def account(self) -> dict[str, Any]: ...
    async def positions(self) -> list[dict[str, Any]]: ...


async def build_account(broker: AccountReads, policy: Any, taken_at: datetime) -> dict[str, Any] | None:
    """The account as the wire contract has it: equity, cash and each position in USD cents, share counts as decimal
    strings, and the policy's target weights. ``None`` when it cannot be read — a report without an account is still
    a report, and the next one tries again."""
    try:
        account = await broker.account()
        positions = await broker.positions()
    except Exception as exc:  # noqa: BLE001 - the mirror must never be what fails
        logger.warning("core sync: could not read the account for the report: %s: %s", type(exc).__name__, exc)
        return None
    equity, cash = _cents(account.get("equity")), _cents(account.get("cash"))
    if equity is None or cash is None:
        return None
    rows = []
    for position in positions:
        symbol, qty = str(position.get("symbol", "")).strip(), _quantity(position.get("qty"))
        if not symbol or qty is None:
            continue
        rows.append({"symbol": symbol, "qty": qty,
                     "market_value_usd_cents": _cents(position.get("market_value")),
                     "avg_entry_price_usd_cents": _cents(position.get("avg_entry_price")),
                     "current_price_usd_cents": _cents(position.get("current_price"))})
    mix = getattr(policy, "mix", None) or {}
    return {"taken_at": _iso(taken_at), "equity_usd_cents": equity, "cash_usd_cents": cash,
            "last_equity_usd_cents": _cents(account.get("last_equity")), "positions": rows,
            "targets": {symbol: format(weight, "f") for symbol, weight in mix.items()}}


def wire_status(health: Mapping[str, Any]) -> dict[str, Any]:
    """The router's own ``/health`` answer in the shape PFW stores. One source: what the owner sees on the website is
    what ``GET /health`` says, not a second opinion computed here."""
    plan, journal, tick = health.get("plan"), health.get("journal"), health.get("tick")
    age = tick.get("last_age_seconds") if tick else None
    return {
        "trading_enabled": bool(health.get("trading_enabled")),
        "disabled_reason": _clip(health.get("disabled_reason")),
        "halted": bool(health.get("halted")),
        "halt_reason": _clip(health.get("halt_reason")),
        "policy_sha256": health.get("policy_sha256"),
        "policy_effective_from": health.get("policy_effective_from"),
        "plan": ({"id": str(plan["id"])[:64], "kind": str(plan["kind"])[:32], "status": str(plan["status"])[:32],
                  "execute_on": str(plan["execute_on"])} if plan else None),
        "journal": ({"ok": bool(journal["ok"]), "entries": int(journal["entries"]), "reason": _clip(journal.get("reason"))}
                    if journal else {"ok": False, "entries": 0, "reason": "the router has no journal: it did not start"}),
        "tick_age_seconds": None if age is None else float(age),
        "tick_failures": int(tick["failures"]) if tick else 0,
        "attention": [str(item)[:MAX_TEXT] for item in list(health.get("attention") or [])[:MAX_ATTENTION]],
    }


# --- the sync -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reply:
    status: int | None      # None: the request never got an answer
    body: dict[str, Any]
    error: str


def _classify(reply: Reply) -> str:
    """``ok`` (stored, or already stored), ``gap`` (PFW is behind: rewind), ``permanent`` (a rejection that will not
    fix itself) or ``transient`` (try again)."""
    status = reply.status
    if status is None:
        return "transient"
    if 200 <= status < 300:
        return "ok"
    if status == 409 and reply.body.get("error") == "gap" and isinstance(reply.body.get("next_index"), int):
        return "gap"
    if 400 <= status < 500 and status != 429:
        return "permanent"
    return "transient"


class CoreSync:
    def __init__(self, *, settings: SyncSettings, journal_path: Path, state_path: Path,
                 status: Callable[[], Mapping[str, Any]], broker: AccountReads, policy: Any,
                 transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._settings = settings
        self._journal_path = journal_path
        self._cursor_store = CursorStore(state_path)
        self._cursor = self._cursor_store.load()
        self._status = status
        self._broker = broker
        self._policy = policy
        self._transport = transport
        self._clock = clock
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._journal_len = 0
        self._journal_problem: str | None = None
        self._last_report_at: datetime | None = None
        self._last_account_at: datetime | None = None
        self._account_due = False
        self._last_success_at: datetime | None = None
        self._failing_since: datetime | None = None
        self.failures = 0
        self.stuck = False
        self.last_error: str | None = None

    @classmethod
    def from_env(cls, *, journal_path: Path, state_path: Path, status: Callable[[], Mapping[str, Any]],
                 broker: AccountReads, policy: Any, env: Mapping[str, str] | None = None) -> CoreSync | None:
        settings = SyncSettings.from_env(env)
        if settings is None:
            return None
        return cls(settings=settings, journal_path=journal_path, state_path=state_path, status=status,
                   broker=broker, policy=policy)

    # -- the loop -------------------------------------------------------------------------------------------------

    def next_delay(self) -> float:
        if self.stuck:
            return PERMANENT_RETRY_SECONDS
        if self.failures:
            return min(BACKOFF_MAX_SECONDS, SYNC_TICK_SECONDS * 2 ** self.failures)
        return SYNC_TICK_SECONDS

    async def run_forever(self) -> None:
        logger.info("core sync: mirroring to %s", self._settings.display_url())
        while True:
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001 - run_once never raises; this is belt and braces for the one thing that must not stop
                logger.exception("core sync: a pass failed unexpectedly")
            await self._sleep(self.next_delay())

    async def run_once(self) -> None:
        """One pass: send what the journal has that PFW has not acknowledged, then a report if one is due. Never
        raises: a failure is recorded and the next pass retries."""
        now = self._clock()
        try:
            delivered = await self._push_journal(now)
            if delivered is not None:
                await self._push_report(now, news=delivered)
        except Exception as exc:  # noqa: BLE001
            self._fail(now, f"{type(exc).__name__}: {exc}", permanent=False)

    # -- the journal ----------------------------------------------------------------------------------------------

    async def _push_journal(self, now: datetime) -> bool | None:
        """``True`` if entries were delivered this pass, ``False`` if there was nothing to send, ``None`` if a send failed."""
        read = read_journal(self._journal_path)
        self._journal_problem = read.problem
        entries = read.entries
        self._journal_len = len(entries)
        chain_id = entries[0].hash if entries else None

        if chain_id != self._cursor.chain_id:
            # A different journal (the state directory was reset) or none at all: a new chain, from its first entry.
            self._set_cursor(Cursor(chain_id, 0))
        elif self._cursor.next_index > len(entries):
            # The journal is shorter than the cursor says: an older state directory was restored. Nothing to send;
            # anything the router writes next either continues what PFW holds or is reported to PFW as a conflict.
            self._set_cursor(Cursor(chain_id, len(entries)))

        delivered = False
        for _ in range(MAX_BATCHES_PER_PASS):
            pending = entries[self._cursor.next_index:]
            if not pending or chain_id is None:
                break
            batch = pending[:BATCH_SIZE]
            reply = await self._post({
                "schema_version": SCHEMA_VERSION, "kind": "journal", "chain_id": chain_id,
                "idempotency_key": f"core-journal:{chain_id[:16]}:{batch[0].index}-{batch[-1].index}",
                "entries": [{"index": e.index, "hash": e.hash, "prev": e.prev, "raw": e.raw} for e in batch]})
            outcome = _classify(reply)
            if outcome == "ok":
                reported = reply.body.get("next_index")
                next_index = reported if isinstance(reported, int) and not isinstance(reported, bool) else batch[-1].index + 1
                self._set_cursor(Cursor(chain_id, max(0, min(next_index, len(entries)))))
                self._succeeded(now)
                delivered = True
                if any(self._event_of(e.raw) in PLAN_CLOSING_EVENTS for e in batch):
                    self._account_due = True
            elif outcome == "gap":
                # PFW holds fewer entries than the cursor says (a restored database): go back to where it is.
                self._set_cursor(Cursor(chain_id, max(0, min(int(reply.body["next_index"]), len(entries)))))
            else:
                self._fail(now, reply.error, permanent=outcome == "permanent")
                return None
        return delivered

    @staticmethod
    def _event_of(raw: str) -> str:
        try:
            return str(json.loads(raw).get("event", ""))
        except (ValueError, AttributeError):
            return ""

    def _set_cursor(self, cursor: Cursor) -> None:
        if cursor != self._cursor:
            self._cursor = cursor
            self._cursor_store.save(cursor)

    # -- the report -----------------------------------------------------------------------------------------------

    async def _push_report(self, now: datetime, *, news: bool) -> None:
        due = (self._last_report_at is None or news
               or (now - self._last_report_at).total_seconds() >= REPORT_INTERVAL_SECONDS)
        if not due:
            return
        try:
            status = wire_status(self._status())
        except Exception as exc:  # noqa: BLE001 - the router's health failing is worth saying, not worth stopping for
            self.last_error = f"could not read the router's status: {type(exc).__name__}: {exc}"
            logger.warning("core sync: %s", self.last_error)
            return
        include_account = (self._last_account_at is None or self._account_due
                           or (now - self._last_account_at).total_seconds() >= ACCOUNT_INTERVAL_SECONDS)
        account = await build_account(self._broker, self._policy, now) if include_account else None
        reported_at = _iso(now)
        reply = await self._post({"schema_version": SCHEMA_VERSION, "kind": "report", "idempotency_key": f"core-report:{reported_at}",
                                  "reported_at": reported_at, "status": status, "account": account})
        outcome = _classify(reply)
        if outcome in ("ok", "gap"):
            self._last_report_at = now
            if account is not None:
                self._last_account_at, self._account_due = now, False
            self._succeeded(now)
        else:
            self._fail(now, reply.error, permanent=outcome == "permanent")

    # -- one signed request ---------------------------------------------------------------------------------------

    async def _post(self, payload: dict[str, Any]) -> Reply:
        """One signed POST. The timestamp is the real time, taken now: PFW rejects one outside its replay window, so
        it must never be the injected scheduling clock's, and a retry hours later gets a fresh one."""
        body = webhook.canonical_json(payload)
        timestamp = str(int(time.time()))
        headers = {"Content-Type": "application/json", webhook.TIMESTAMP_HEADER: timestamp,
                   webhook.SIGNATURE_HEADER: f"{webhook.SIGNATURE_PREFIX}{webhook.sign(body, timestamp, self._settings.secret)}",
                   webhook.IDEMPOTENCY_HEADER: payload["idempotency_key"], "User-Agent": "core-router-sync/1.0"}
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, transport=self._transport) as client:
                response = await client.post(self._settings.url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            return Reply(status=None, body={}, error=f"{type(exc).__name__}: {exc}")
        try:
            parsed = response.json()
        except ValueError:
            parsed = {}
        data = parsed if isinstance(parsed, dict) else {}
        if response.is_success:
            return Reply(status=response.status_code, body=data, error="")
        code = data.get("error") if isinstance(data.get("error"), str) else ""
        detail = data.get("detail") if isinstance(data.get("detail"), str) else ""
        text = f"HTTP {response.status_code}" + (f" {code}" if code else "") + (f": {detail}" if detail else "")
        if not code and not detail:
            text += f": {response.text[:200]}"
        return Reply(status=response.status_code, body=data, error=text)

    # -- bookkeeping ----------------------------------------------------------------------------------------------

    def _succeeded(self, now: datetime) -> None:
        self.failures, self.stuck, self.last_error = 0, False, None
        self._last_success_at, self._failing_since = now, None

    def _fail(self, now: datetime, error: str, *, permanent: bool) -> None:
        self.failures += 1
        self.stuck = permanent
        self.last_error = error
        if self._failing_since is None:
            self._failing_since = now
        logger.warning("core sync: %s%s", error, " (will not be retried quickly: a person should look)" if permanent else "")

    def attention(self) -> str | None:
        """A line for ``/health``'s attention list once the mirror has been failing long enough to matter."""
        if self._failing_since is None:
            return None
        failing_for = (self._clock() - self._failing_since).total_seconds()
        if failing_for < STUCK_AFTER_SECONDS:
            return None
        return f"the website mirror has not been updated for {int(failing_for // 60)} min: {self.last_error}"

    def health(self) -> dict[str, Any]:
        # No URL: /health is unauthenticated, and where the owner's website lives is not for everyone who can reach it.
        return {"enabled": True,
                "last_success_at": None if self._last_success_at is None else _iso(self._last_success_at),
                "consecutive_failures": self.failures, "last_error": self.last_error, "stuck": self.stuck,
                "pending_entries": max(0, self._journal_len - self._cursor.next_index),
                "journal_problem": self._journal_problem,
                "chain": (self._cursor.chain_id or "")[:12] or None, "next_index": self._cursor.next_index}
