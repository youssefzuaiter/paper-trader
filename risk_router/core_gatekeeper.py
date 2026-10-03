"""The long-term core's gatekeeper (core design §8): the only component that places core orders.

A plan's life:

1. **Proposed** after the close by the Allocator, signed with its own secret.
2. **Verified** here: the router rebuilds the snapshot from its *own* reads of the account and the
   day's bars, recomputes the plan with ``core_paper.decide`` (the backtest's ``core_alloc``
   arithmetic) and rejects any difference (``plan_mismatch``); then the tier-0 core limits.
3. **Approved**: automatically if routine; the initial build needs the owner's signed ``fund``
   command (and the account's cash within the policy's funded cap); a plan trading more than
   ``approval_needed_above_usd``, and every raise-cash plan, needs the owner's signed approval.
4. **Executed** on the next session: in the 09:20-09:27 window, sells go in as collared limit orders
   (Alpaca fills them at the official opening price); once they fill, buys follow as marketable
   notional limits, scaled to the cash the account actually has. A plan with no sells sends its buys
   pre-open the same way.
5. **Closed**, always, one of these ways::

       awaiting_approval ─approve─▶ approved ─09:20-09:27─▶ selling ─sells done─▶ buying ─buys done─▶ done
              │                        │                      │                      │
              └─ window passed ────────┴─▶ expired            └─ 15:30 cutoff, or    └─▶ done, with unfilled
                 (re-decided at the close)                       the next day ───────▶ abandoned   orders: re-decided
                                                                 (re-decided)

   ``deferred`` (the breaker blocked a plan that buys) and ``halted`` (the kill switch) close a plan before any
   order exists. A plan is never left open: ``reconcile`` closes whatever can no longer finish, so a process
   that died between the sells and the buys cannot strand the cash. Anything that ends short of complete sets
   ``forced``, and the next close re-decides it from the account's actual positions.
6. **Recorded**: every step goes to an append-only journal, each entry carrying the policy's sha256
   and the previous entry's hash.

Guards, imported from the news router and unchanged: the kill switch blocks everything; the daily
loss breaker blocks any plan that buys (a sells-only raise-cash plan still passes). The breaker reads the
account's equity against the last close, and before the open that is still the last close, so it cannot see
a gap-down at the 09:20-09:27 check: it is re-checked when the buys go in, after the sells have filled, and a
block there abandons the rest of the plan for the next close to re-decide. The backtest defers the whole plan
at the open instead; that difference is documented in docs/core-paper-mode.md. There are no exits: core
positions are never stopped out.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import core_alloc
import core_paper
import tier0_core
from risk_router import core_alerts
from risk_router.alpaca_async import OrderRejected
from risk_router.guards import ExecutionGuard, GuardBlocked, Intent
from tier0_core import CorePolicy, Executed

logger = logging.getLogger("risk_router.core")

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")
#: How long after the open sells may take to fill before the rest is cancelled and buys proceed.
SELL_DEADLINE_MINUTES: Final[int] = 30
#: Buys not filled by then are cancelled; the drift waits for the next close's re-decision.
BUY_DEADLINE_MINUTES: Final[int] = 90
#: New York time. A plan still working after this (the process was down, an order never settled) is abandoned
#: rather than traded into the close; its remainder is re-decided at that evening's close.
EXECUTION_CUTOFF: Final[time] = time(15, 30)
FEE_ALLOWANCE: Final[Decimal] = Decimal("0.00001")  # equity buys pay only CAT on top of the notional
#: What a raise-cash sale is assumed to lose (half-spread + slippage + fees): the core's central level for
#: its widest instrument (VXUS, 3.9 bp + 2 bp) plus sell fees, rounded up. Grosses the sells up.
RAISE_CASH_COST_RATE: Final[Decimal] = Decimal("0.0007")
#: An order in one of these states will not change again.
TERMINAL: Final[tuple[str, ...]] = ("filled", "canceled", "expired", "rejected")
#: Consecutive failed ticks before the first ``tick_failed`` entry (45 s at the 15 s tick), then one an hour.
TICK_FAILURES_BEFORE_ALERT: Final[int] = 3
TICK_FAILURES_REPEAT_EVERY: Final[int] = 240


class Broker(Protocol):
    async def account(self) -> dict[str, Any]: ...
    async def positions(self) -> list[dict[str, Any]]: ...
    async def daily_bars(self, symbols: list[str], start: str, end: str) -> dict[str, list[dict[str, Any]]]: ...
    async def calendar(self, start: str, end: str) -> list[dict[str, Any]]: ...
    async def latest_quote(self, symbol: str) -> Any: ...
    async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None: ...
    async def cancel_all_orders(self) -> int: ...


class PlanRejected(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class CoreState:
    """What survives a restart, beside the shared kill-switch file. Alpaca stays the source of truth
    for positions and cash; this holds only what Alpaca cannot know."""
    plan: dict[str, Any] | None = None           # the current plan record
    history: list[dict[str, Any]] = field(default_factory=list)
    targets: dict[str, str] = field(default_factory=dict)
    forced: bool = False                         # the last plan was deferred, expired, abandoned or left unfilled
    initial_incomplete: bool = False             # the first build did not finish: completing it is still "the build"
    journal_head: str = ""                       # an anchor: a copy of the journal's last hash (the file is the truth)


class CoreStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> CoreState:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return CoreState()
        return CoreState(**{k: raw[k] for k in CoreState.__dataclass_fields__ if k in raw})

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".core-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data.__dict__, handle, indent=1, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)

    def executed(self) -> list[Executed]:
        return [Executed(date.fromisoformat(h["day"]), h["orders"], Decimal(h["traded_usd"]), h.get("redecision", False),
                         h.get("kind") == "initial") for h in self.data.history]


class Journal:
    """Append-only JSON lines; each entry carries the previous entry's sha256, so an edit or a deletion
    anywhere breaks the chain (``verify``).

    The *file* is the source of truth for the chain's head, so losing the state file cannot fork the chain.
    The state file keeps a copy as an anchor: a journal cut short is still internally consistent, but it no
    longer ends where the anchor says it should (``status``)."""

    def __init__(self, path: Path, store: CoreStore, policy_sha: str,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.path, self.store, self.policy_sha, self.on_event = path, store, policy_sha, on_event
        self._head = self._tail_hash()

    def _lines(self) -> list[str]:
        try:
            return [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except FileNotFoundError:
            return []

    def _tail_hash(self) -> str:
        lines = self._lines()
        return hashlib.sha256(lines[-1].encode()).hexdigest() if lines else ""

    def write(self, event: str, **detail: Any) -> dict[str, Any]:
        entry = {"at": datetime.now(UTC).isoformat(), "event": event, "policy_sha256": self.policy_sha,
                 "prev": self._head, **detail}
        line = json.dumps(entry, sort_keys=True, default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._head = hashlib.sha256(line.encode()).hexdigest()
        self.store.data.journal_head = self._head
        self.store.save()
        logger.info("core journal: %s %s", event, {k: v for k, v in detail.items() if k != "plan"})
        if self.on_event is not None:
            try:
                self.on_event(event, entry)
            except Exception:  # noqa: BLE001 - a failing alert must never lose or delay a journal entry
                logger.exception("core journal: the event hook failed for %s", event)
        return entry

    @staticmethod
    def verify(path: Path) -> bool:
        head = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if json.loads(line)["prev"] != head:
                return False
            head = hashlib.sha256(line.encode()).hexdigest()
        return True

    def status(self, *, repair: bool = True) -> dict[str, Any]:
        """Whether the journal can be trusted: its hash chain intact, and ending where the state file says.
        A crash between appending an entry and saving the anchor leaves the file exactly one entry ahead,
        which is not tampering: the anchor is then the last entry's ``prev``, and (unless ``repair`` is False,
        for read-only callers) the anchor is caught up."""
        lines = self._lines()
        anchor = self.store.data.journal_head
        try:
            chain_ok = Journal.verify(self.path) if lines else True
            last_prev = json.loads(lines[-1])["prev"] if lines else ""
        except (ValueError, KeyError):
            chain_ok, last_prev = False, ""
        if not chain_ok:
            ok, reason = False, "an entry was altered or removed: the hash chain is broken"
        elif anchor and anchor not in (self._head, last_prev):
            ok, reason = False, ("the journal and the state file disagree about the last entry "
                                 "(entries removed, or the files are from different times)")
        else:
            ok, reason = True, None
            if repair and anchor != self._head:      # the benign crash window above: catch the anchor up
                self.store.data.journal_head = self._head
                self.store.save()
        return {"ok": ok, "entries": len(lines), "head": self._head[:12], "anchored": bool(anchor), "reason": reason}


def _traded_usd(orders: dict[str, dict[str, Any]]) -> Decimal:
    """Dollars actually filled across a plan's orders (partial fills count, whatever the order ended as)."""
    total = Decimal(0)
    for order in orders.values():
        qty, price = order.get("filled_qty"), order.get("filled_avg_price")
        if qty and price and Decimal(str(qty)) > 0:
            total += Decimal(str(qty)) * Decimal(str(price))
    return total


def _in_flight(plan: dict[str, Any] | None) -> bool:
    """Orders may be live at Alpaca: the status says so, or some have already been sent (a submission part-way
    through still reads 'approved')."""
    return plan is not None and (plan["status"] in ("selling", "buying") or bool(plan["orders"]))


class CoreRouter:
    def __init__(self, broker: Broker, guard: ExecutionGuard, policy: CorePolicy, store: CoreStore, journal: Journal,
                 *, now: Callable[[], datetime] = lambda: datetime.now(UTC),
                 alerter: core_alerts.Alerter | None = None) -> None:
        self.broker, self.guard, self.policy, self.store, self.journal = broker, guard, policy, store, journal
        self._now = now
        self.alerter = alerter
        self.last_tick_at: datetime | None = None
        self.tick_failures = 0
        # One caller at a time through anything that changes the plan. The background tick and the HTTP routes
        # (the Allocator's state call, a proposal, an approval) share this object and interleave at every await:
        # without the lock two callers could both abandon one stale plan, or a proposal could supersede a plan
        # whose sells are half submitted.
        self._lock = asyncio.Lock()
        journal.on_event = self._on_journal_event

    def _on_journal_event(self, event: str, entry: dict[str, Any]) -> None:
        if self.alerter is not None and core_alerts.should_alert(event, entry):
            self.alerter.notify(*core_alerts.render(event, entry))

    # --- start-up ---------------------------------------------------------------------------------------

    async def check_account(self) -> None:
        """Refuse to run against any account but the policy's (design §8.1)."""
        problems = tier0_core.violations(self.policy)
        if problems:
            raise PlanRejected("policy_invalid", "; ".join(problems))
        number = str((await self.broker.account()).get("account_number", ""))
        if number != self.policy.account:
            raise PlanRejected("wrong_account", f"connected to {number!r}, the policy names {self.policy.account!r}")

    def health(self) -> dict[str, Any]:
        age = None if self.last_tick_at is None else (self._now() - self.last_tick_at).total_seconds()
        return {"last_tick_age_seconds": age, "tick_failures": self.tick_failures, "journal": self.journal.status()}

    # --- the snapshot --------------------------------------------------------------------------------------

    async def snapshot(self, session: date) -> core_paper.Snapshot:
        """Rebuild what the Allocator saw, from this router's own reads (the same shared reader)."""
        state = self.store.data
        try:
            return await core_paper.read_snapshot(self.broker, self.policy, session, forced=state.forced,
                                                  targets={s: Decimal(w) for s, w in state.targets.items()})
        except core_paper.SnapshotError as exc:
            raise PlanRejected(exc.code, exc.detail) from exc

    # --- 2. verify -------------------------------------------------------------------------------------------

    async def receive_plan(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """The Allocator's proposal: ``{"session": ..., "plan": {...} | None, "inputs": digest}``."""
        async with self._lock:
            return await self._receive_plan(proposal)

    async def _receive_plan(self, proposal: dict[str, Any]) -> dict[str, Any]:
        await self._reconcile()    # a plan the process could not finish must be closed before this one is judged
        session = date.fromisoformat(proposal["session"])
        snap = await self.snapshot(session)
        mine = core_paper.decide(self.policy, snap)
        theirs = core_paper.plan_from_json(proposal["plan"]) if proposal.get("plan") else None
        if snap.digest() != proposal.get("inputs") or (mine is None) != (theirs is None) or (
                mine is not None and core_paper.plan_to_json(mine) != core_paper.plan_to_json(theirs)):  # type: ignore[arg-type]
            self.journal.write("plan_mismatch", session=session.isoformat(), proposed=proposal.get("plan"),
                               recomputed=core_paper.plan_to_json(mine) if mine else None,
                               inputs_match=snap.digest() == proposal.get("inputs"))
            raise PlanRejected("plan_mismatch", "the router's recomputation differs from the proposal")
        if mine is None:
            if snap.forced:
                self.store.data.forced = False
                self.store.save()
            reason = core_paper.skip_reason(self.policy, snap)
            self.journal.write("no_plan", session=session.isoformat(), reason=reason)
            return {"status": "no_plan", **({"reason": reason} if reason else {})}
        pid = core_paper.plan_id(mine, snap)
        current = self.store.data.plan
        if current is not None:
            if _in_flight(current):
                raise PlanRejected("plan_in_flight", f"plan {current['id']} is trading now; it must finish or be "
                                                     f"abandoned before another is accepted")
            if current["id"] == pid and current["status"] in ("awaiting_approval", "approved"):
                # The same decision again (a cron retry): keep the plan, and the owner's approval of it, as they are.
                return {"status": current["status"], "plan_id": pid, "execute_on": current["execute_on"]}
        kind = "initial" if (snap.initial or self.store.data.initial_incomplete) else "rebalance"
        return self._admit(mine, snap, kind=kind)

    def _admit(self, plan: core_alloc.Plan, snap: core_paper.Snapshot, *, kind: str) -> dict[str, Any]:
        if kind == "initial" and snap.cash > self.policy.funded_amount_cap_usd:
            self.journal.write("plan_rejected", problems=[f"account cash ${snap.cash} exceeds the funded cap"])
            raise PlanRejected("over_funded", f"account cash ${snap.cash} exceeds the policy's funded cap "
                                              f"${self.policy.funded_amount_cap_usd}: reset the paper account to it")
        account_value = snap.cash + sum((q * snap.closes[s] for s, q in snap.qty.items()), Decimal(0))
        problems = tier0_core.plan_violations(plan, self.policy, prices=snap.closes, equity=account_value,
                                              history=self.store.executed(), today=snap.session,
                                              initial=kind == "initial", funded=kind == "initial")
        pid = core_paper.plan_id(plan, snap)
        if problems:
            self.journal.write("plan_rejected", plan_id=pid, problems=problems, plan=core_paper.plan_to_json(plan))
            raise PlanRejected("limits", "; ".join(problems))
        needs = kind in ("initial", "raise_cash") or tier0_core.needs_approval(plan, self.policy, snap.closes)
        record = {"id": pid, "kind": kind, "status": "awaiting_approval" if needs else "approved",
                  "plan": core_paper.plan_to_json(plan), "decided_on": snap.session.isoformat(),
                  "execute_on": snap.next_session.isoformat(), "closes": {s: str(c) for s, c in snap.closes.items()},
                  "redecision": snap.forced, "orders": {}}
        replaced = self.store.data.plan
        if replaced is not None and replaced["id"] != pid:
            self.journal.write("plan_superseded", plan_id=replaced["id"], by=pid)
        self.store.data.plan = record
        self.store.data.forced = False
        self.store.save()
        self.journal.write("plan_accepted", plan_id=pid, kind=kind, status=record["status"],
                           plan=record["plan"], execute_on=record["execute_on"], redecision=record["redecision"])
        return {"status": record["status"], "plan_id": pid, "execute_on": record["execute_on"]}

    # --- 3. approvals (owner-signed control calls) ---------------------------------------------------------

    async def approve(self, pid: str, *, fund: bool = False) -> dict[str, Any]:
        async with self._lock:
            return await self._approve(pid, fund=fund)

    async def _approve(self, pid: str, *, fund: bool) -> dict[str, Any]:
        await self._reconcile()    # an approval that arrives after the plan's window finds the plan already closed
        record = self.store.data.plan
        if not record or record["id"] != pid or record["status"] != "awaiting_approval":
            raise PlanRejected("no_such_plan", f"no plan {pid} awaiting approval")
        if record["kind"] == "initial":
            if not fund:
                raise PlanRejected("fund_required", "the initial build needs the signed fund command")
            cash = Decimal(str((await self.broker.account())["cash"]))
            if cash > self.policy.funded_amount_cap_usd:
                raise PlanRejected("over_funded", f"account cash ${cash} exceeds the policy's funded cap "
                                                  f"${self.policy.funded_amount_cap_usd}")
        record["status"] = "approved"
        self.store.save()
        self.journal.write("plan_approved", plan_id=pid, fund=fund)
        return {"status": "approved", "plan_id": pid}

    async def raise_cash(self, amount: Decimal, need_by: date) -> dict[str, Any]:
        """The owner's request (system design §3): a sells-only plan, executed only after approval."""
        async with self._lock:
            return await self._raise_cash(amount, need_by)

    async def _raise_cash(self, amount: Decimal, need_by: date) -> dict[str, Any]:
        await self._reconcile()
        now_ny = self._now().astimezone(NEW_YORK)
        sessions = await self.broker.calendar((now_ny.date() - timedelta(days=10)).isoformat(), now_ny.date().isoformat())
        last = core_paper.last_published([date.fromisoformat(s["date"]) for s in sessions], self._now())
        if last is None:
            raise PlanRejected("not_a_session", "no session close has been published in the last ten days")
        snap = await self.snapshot(last)
        reason = core_paper.skip_reason(self.policy, snap)
        if reason:
            raise PlanRejected("not_effective", reason)
        if snap.next_session < now_ny.date() or (snap.next_session == now_ny.date()
                                                 and now_ny.time() > tier0_core.SUBMIT_UNTIL):
            # Asked during the session, the last published close is yesterday's and the plan it would make
            # executes at *today's* open, which is over: it would be born expired. Decide on today's close instead.
            raise PlanRejected("ask_after_the_close", f"a plan decided on the {snap.session} close would execute at "
                                                      f"the {snap.next_session} open, which has passed; ask again "
                                                      f"after {tier0_core.DECIDE_AFTER:%H:%M} New York")
        if _in_flight(self.store.data.plan):
            raise PlanRejected("plan_in_flight", f"plan {self.store.data.plan['id']} is trading now")
        if snap.next_session > need_by:
            raise PlanRejected("too_late", f"the next session {snap.next_session} is after {need_by}")
        held = {s: q for s, q in snap.qty.items() if q}
        rates = dict.fromkeys(held, RAISE_CASH_COST_RATE)
        try:
            plan = core_alloc.plan_raise_cash(snap.session, held, {s: snap.closes[s] for s in held},
                                              snap.cash - self.policy.cash_reserve_usd, amount, self.policy.mix,
                                              cost_rate=rates, buffer_symbol=self.policy.buffer_symbol,
                                              min_order_usd=self.policy.min_order_usd)
        except ValueError as exc:      # "exceeds_portfolio": more than the account could raise
            raise PlanRejected("exceeds_portfolio", str(exc).removeprefix("exceeds_portfolio: ")) from exc
        return self._admit(plan, snap, kind="raise_cash")

    # --- 4. execution ---------------------------------------------------------------------------------------

    async def run_tick(self) -> None:
        """One pass of the app's loop: ``tick`` plus the bookkeeping the health check and alerts read. A failure
        is counted and journaled, never raised: the next tick retries, and orders are at-most-once by id."""
        try:
            await self.tick()
        except Exception as exc:  # noqa: BLE001
            self.tick_failures += 1
            logger.exception("core tick failed (%d in a row)", self.tick_failures)
            if self.tick_failures == TICK_FAILURES_BEFORE_ALERT or self.tick_failures % TICK_FAILURES_REPEAT_EVERY == 0:
                self.journal.write("tick_failed", error=f"{type(exc).__name__}: {exc}"[:300],
                                   consecutive=self.tick_failures)
            return
        self.tick_failures = 0
        self.last_tick_at = self._now()

    async def tick(self) -> None:
        """Moves the current plan along; called every few seconds by the app's loop."""
        async with self._lock:
            await self._tick()

    async def _tick(self) -> None:
        await self._reconcile()
        record = self.store.data.plan
        if not record:
            return
        now_ny = self._now().astimezone(NEW_YORK)
        if now_ny.date() != date.fromisoformat(record["execute_on"]):
            return
        if record["status"] == "approved" and tier0_core.SUBMIT_FROM <= now_ny.time() <= tier0_core.SUBMIT_UNTIL:
            await self._start(record)
        elif record["status"] == "selling":
            await self._after_sells(record, now_ny)
        elif record["status"] == "buying":
            await self._after_buys(record, now_ny)

    async def reconcile(self) -> None:
        """Close a plan that can no longer finish as intended (see ``_reconcile``); safe to call from anywhere."""
        async with self._lock:
            await self._reconcile()

    async def _reconcile(self) -> None:
        """Close a plan that can no longer finish as intended, so none is ever left open.

        * ``approved`` / ``awaiting_approval``: its submit window has passed. With no order sent it ``expired``;
          with some sent (a submission that failed part-way and never resumed) it is abandoned.
        * ``selling`` / ``buying``: the 15:30 cutoff, or the next day, has come. Orders still working are
          cancelled; if every buy had filled the plan is simply done, otherwise it is abandoned.

        Either way ``forced`` is set, so the next close re-decides what is left from the account's real
        positions. Idempotent: the app's tick, the Allocator's state call and every approval run it first.
        """
        record = self.store.data.plan
        if not record:
            return
        now_ny = self._now().astimezone(NEW_YORK)
        execute_on = date.fromisoformat(record["execute_on"])
        today, clock = now_ny.date(), now_ny.time()
        if record["status"] in ("approved", "awaiting_approval"):
            if today > execute_on or (today == execute_on and clock > tier0_core.SUBMIT_UNTIL):
                if record["orders"]:
                    await self._abandon(record, "the submit window closed with the plan only partly submitted")
                else:
                    self._close(record, "expired", forced=True, reason="its session passed before it could start")
        elif record["status"] in ("selling", "buying"):
            if today > execute_on or (today == execute_on and clock >= EXECUTION_CUTOFF):
                await self._abandon(record, f"still {record['status']} at {now_ny:%Y-%m-%d %H:%M} New York: "
                                            f"the process was down, or an order never settled")

    async def _abandon(self, record: dict[str, Any], reason: str, *, forced: bool = True) -> None:
        """Cancel what is still working, read the final state of every order, and close the plan."""
        try:
            await self.broker.cancel_all_orders()
            await self._refresh(record, "sell")
            await self._refresh(record, "buy")
        except Exception as exc:  # noqa: BLE001 - closing the plan matters more than a perfect final reading
            logger.exception("core: could not read the final state of an abandoned plan's orders")
            reason = f"{reason} (the final order states could not be read: {exc})"
        complete = (record["status"] == "buying" and not record.get("blocked") and bool(record["orders"])
                    and all(o.get("status") == "filled" for o in record["orders"].values()))
        if complete:
            self._close(record, "done")
        else:
            self._close(record, "abandoned", forced=forced, reason=reason)

    async def _start(self, record: dict[str, Any]) -> None:
        plan = core_paper.plan_from_json(record["plan"])
        try:
            await self.guard.check(Intent.OPEN if plan.buys else Intent.REDUCE)
        except GuardBlocked as exc:
            # The breaker blocks any plan that buys, whole: deferred, re-decided at the next close (§3.1 a).
            # The kill switch blocks everything. If a resumed submission already has orders live, cancel them.
            if record["orders"]:
                await self._abandon(record, f"{exc.code}: {exc.detail}", forced=exc.code != "halted")
            else:
                self._close(record, "deferred" if exc.code != "halted" else "halted", forced=exc.code != "halted",
                            reason=f"{exc.code}: {exc.detail}")
            return
        closes = {s: Decimal(c) for s, c in record["closes"].items()}
        try:
            for s, q in sorted(plan.sells.items()):
                await self._submit(record, core_paper.sell_payload(record["id"], s, q, closes[s]))
        except GuardBlocked as exc:     # the kill switch, pressed between two sells
            await self._abandon(record, f"{exc.code}: {exc.detail}", forced=exc.code != "halted")
            return
        if plan.sells:
            record["status"] = "selling"
        else:
            await self._buy(record, plan, closes, preopen=True)
        self.store.save()

    async def _submit(self, record: dict[str, Any], payload: dict[str, Any]) -> None:
        """Send one order, at most once. Re-entrant: an order already recorded under its deterministic client id
        is skipped, so a submission that failed part-way resumes on the next tick without duplicating anything
        (and ``AsyncAlpaca.submit_order`` looks an unrecorded duplicate up by that id before it re-sends)."""
        cid = payload["client_order_id"]
        if cid in record["orders"]:
            return
        intent = Intent.OPEN if payload["side"] == "buy" else Intent.REDUCE
        await self.guard.check(intent)  # again at the one place orders leave (as the news router does)
        try:
            order = await self.broker.submit_order(payload)
        except OrderRejected as exc:
            # Alpaca refused it for good (a 4xx): note it and carry on. Retrying every tick for seven minutes
            # would change nothing, and the rest of the plan should not be held hostage by one order.
            record["orders"][cid] = {"symbol": payload["symbol"], "side": payload["side"], "status": "rejected",
                                     "payload": payload, "error": str(exc)[:300]}
            self.store.save()
            self.journal.write("order_rejected", plan_id=record["id"], symbol=payload["symbol"], side=payload["side"],
                               error=str(exc)[:300])
            return
        record["orders"][cid] = {"symbol": payload["symbol"], "side": payload["side"],
                                 "status": str(order.get("status")), "payload": payload}
        self.store.save()
        self.journal.write("order_submitted", plan_id=record["id"], payload=payload, alpaca_status=order.get("status"))

    async def _refresh(self, record: dict[str, Any], side: str) -> list[dict[str, Any]]:
        out = []
        for cid, o in record["orders"].items():
            if o["side"] != side:
                continue
            if o["status"] != "rejected":        # never reached Alpaca: there is nothing to look up
                latest = await self.broker.order_by_client_id(cid)
                if latest is not None:
                    o["status"] = str(latest.get("status"))
                    o["filled_qty"] = str(latest.get("filled_qty") or "0")
                    o["filled_avg_price"] = str(latest.get("filled_avg_price") or "")
            out.append(o)
        return out

    async def _after_sells(self, record: dict[str, Any], now_ny: datetime) -> None:
        sells = await self._refresh(record, "sell")
        done = all(o["status"] in TERMINAL for o in sells)
        open_at = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        if not done and now_ny < open_at + timedelta(minutes=SELL_DEADLINE_MINUTES):
            self.store.save()
            return
        if not done:
            await self.broker.cancel_all_orders()
            await self._refresh(record, "sell")
            self.journal.write("sells_cancelled_at_deadline", plan_id=record["id"])
        plan = core_paper.plan_from_json(record["plan"])
        await self._buy(record, plan, {s: Decimal(c) for s, c in record["closes"].items()}, preopen=False)
        self.store.save()

    async def _buy(self, record: dict[str, Any], plan: core_alloc.Plan, closes: dict[str, Decimal], *,
                   preopen: bool) -> None:
        if not plan.buys:
            self._close(record, "done")
            return
        try:
            await self.guard.check(Intent.OPEN)
        except GuardBlocked as exc:
            # Before the open the breaker cannot see the day's gap (equity still equals the last close); it can
            # only trip now, with the sells already in. No exposure is added; the rest is re-decided at the close.
            self.journal.write("buy_blocked", plan_id=record["id"], code=exc.code, detail=exc.detail)
            self._close(record, "abandoned" if record["orders"] else "deferred", forced=exc.code != "halted",
                        reason=f"buys blocked: {exc.code}: {exc.detail}")
            return
        scaled = record.get("buys_scaled")
        if scaled is None:    # sized once, from the cash the account has now; a resumed attempt reuses these
            cash = Decimal(str((await self.broker.account())["cash"]))
            allowance = core_alloc.CENT * (4 * len(plan.buys) + 4)
            buys = core_alloc.scale_buys(plan.buys, cash - self.policy.cash_reserve_usd - allowance,
                                         dict.fromkeys(plan.buys, FEE_ALLOWANCE))
            record["buys_scaled"] = {s: str(n) for s, n in buys.items()}
        else:
            buys = {s: Decimal(n) for s, n in scaled.items()}
        for s, notional in sorted(buys.items()):
            if notional < self.policy.min_order_usd:
                continue
            limit = core_paper.preopen_limit("buy", closes[s]) if preopen else \
                core_paper.marketable_buy_limit(Decimal(str((await self.broker.latest_quote(s)).ask)))
            payload = core_paper.buy_payload(record["id"], s, notional, limit)
            if payload is None:
                continue
            try:
                await self._submit(record, payload)
            except GuardBlocked as exc:     # the breaker tripped between two buys: add no more exposure today
                record["blocked"] = True
                self.journal.write("buy_blocked", plan_id=record["id"], code=exc.code, detail=exc.detail)
                break
        record["status"] = "buying"

    async def _after_buys(self, record: dict[str, Any], now_ny: datetime) -> None:
        buys = await self._refresh(record, "buy")
        done = all(o["status"] in TERMINAL for o in buys)
        open_at = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        if not done and now_ny < open_at + timedelta(minutes=BUY_DEADLINE_MINUTES):
            self.store.save()
            return
        if not done:
            await self.broker.cancel_all_orders()
            await self._refresh(record, "buy")
            self.journal.write("buys_cancelled_at_deadline", plan_id=record["id"])
        self._close(record, "done")

    def _close(self, record: dict[str, Any], status: str, *, forced: bool | None = None,
               reason: str | None = None) -> None:
        """End a plan. ``forced`` (the next close re-decides it) defaults to *the plan did not fully fill*."""
        record["status"] = status
        state = self.store.data
        orders = record["orders"]
        incomplete = bool(record.get("blocked")) or any(o.get("status") != "filled" for o in orders.values())
        if forced is None:
            forced = incomplete if status == "done" else False
        if status in ("done", "abandoned"):
            state.history.append({"day": record["execute_on"], "orders": len(orders), "kind": record["kind"],
                                  "traded_usd": str(_traded_usd(orders).quantize(core_alloc.CENT, ROUND_DOWN)),
                                  "redecision": record.get("redecision", False), "status": status})
            if record["kind"] == "initial":
                state.initial_incomplete = incomplete or status == "abandoned"
        if status == "done" and not incomplete and record["kind"] != "raise_cash":
            state.targets = {s: str(w) for s, w in core_paper.plan_from_json(record["plan"]).targets.items()}
        state.forced = forced
        state.plan = None
        self.store.save()
        self.journal.write(f"plan_{status}", plan_id=record["id"], kind=record["kind"], reason=reason, orders=orders)

    # --- the shadow line (design §8.4.5), after the day's bars are published -------------------------------------

    def shortfall(self, order: dict[str, Any], open_price: Decimal, half_spread: Decimal,
                  slippage: Decimal) -> dict[str, Decimal] | None:
        if order.get("status") != "filled" or not order.get("filled_avg_price"):
            return None
        return core_paper.shortfall_bps(order["side"], Decimal(order["filled_avg_price"]), open_price, half_spread,
                                        slippage)
