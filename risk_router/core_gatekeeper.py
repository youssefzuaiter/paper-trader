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
5. **Recorded**: every step goes to an append-only journal, each entry carrying the policy's sha256
   and the previous entry's hash.

Guards, imported from the news router and unchanged: the kill switch blocks everything; the daily
loss breaker blocks any plan that buys, **whole** (the plan is deferred and re-decided at the next
close, exactly as the backtest modelled), while a sells-only raise-cash plan still passes. There are
no exits: core positions are never stopped out.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import core_alloc
import core_paper
import tier0_core
from risk_router.guards import ExecutionGuard, GuardBlocked, Intent
from tier0_core import CorePolicy, Executed

logger = logging.getLogger("risk_router.core")

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")
#: How long after the open sells may take to fill before the rest is cancelled and buys proceed.
SELL_DEADLINE_MINUTES: Final[int] = 30
#: Buys not filled by then are cancelled; the drift waits for the next rebalance.
BUY_DEADLINE_MINUTES: Final[int] = 90
FEE_ALLOWANCE: Final[Decimal] = Decimal("0.00001")  # equity buys pay only CAT on top of the notional
#: What a raise-cash sale is assumed to lose (half-spread + slippage + fees): the core's central level for
#: its widest instrument (VXUS, 3.9 bp + 2 bp) plus sell fees, rounded up. Grosses the sells up.
RAISE_CASH_COST_RATE: Final[Decimal] = Decimal("0.0007")


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
    forced: bool = False                         # the last plan was deferred or expired: re-decide at the close
    journal_head: str = ""


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
    anywhere breaks the chain (``verify``)."""

    def __init__(self, path: Path, store: CoreStore, policy_sha: str) -> None:
        self.path, self.store, self.policy_sha = path, store, policy_sha

    def write(self, event: str, **detail: Any) -> dict[str, Any]:
        entry = {"at": datetime.now(UTC).isoformat(), "event": event, "policy_sha256": self.policy_sha,
                 "prev": self.store.data.journal_head, **detail}
        line = json.dumps(entry, sort_keys=True, default=str)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.store.data.journal_head = hashlib.sha256(line.encode()).hexdigest()
        self.store.save()
        logger.info("core journal: %s %s", event, {k: v for k, v in detail.items() if k != "plan"})
        return entry

    @staticmethod
    def verify(path: Path) -> bool:
        head = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if json.loads(line)["prev"] != head:
                return False
            head = hashlib.sha256(line.encode()).hexdigest()
        return True


class CoreRouter:
    def __init__(self, broker: Broker, guard: ExecutionGuard, policy: CorePolicy, store: CoreStore, journal: Journal,
                 *, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.broker, self.guard, self.policy, self.store, self.journal = broker, guard, policy, store, journal
        self._now = now

    # --- start-up ---------------------------------------------------------------------------------------

    async def check_account(self) -> None:
        """Refuse to run against any account but the policy's (design §8.1)."""
        problems = tier0_core.violations(self.policy)
        if problems:
            raise PlanRejected("policy_invalid", "; ".join(problems))
        number = str((await self.broker.account()).get("account_number", ""))
        if number != self.policy.account:
            raise PlanRejected("wrong_account", f"connected to {number!r}, the policy names {self.policy.account!r}")

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
            self.journal.write("no_plan", session=session.isoformat())
            return {"status": "no_plan"}
        return self._admit(mine, snap, kind="initial" if snap.initial else "rebalance")

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
        self.store.data.plan = record
        self.store.data.forced = False
        self.store.save()
        self.journal.write("plan_accepted", plan_id=pid, kind=kind, status=record["status"],
                           plan=record["plan"], execute_on=record["execute_on"])
        return {"status": record["status"], "plan_id": pid, "execute_on": record["execute_on"]}

    # --- 3. approvals (owner-signed control calls) ---------------------------------------------------------

    async def approve(self, pid: str, *, fund: bool = False) -> dict[str, Any]:
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
        today = self._now().astimezone(NEW_YORK).date()
        sessions = await self.broker.calendar((today - timedelta(days=10)).isoformat(), today.isoformat())
        last = max(date.fromisoformat(s["date"]) for s in sessions if date.fromisoformat(s["date"]) <= today)
        snap = await self.snapshot(last)
        if snap.next_session > need_by:
            raise PlanRejected("too_late", f"the next session {snap.next_session} is after {need_by}")
        held = {s: q for s, q in snap.qty.items() if q}
        rates = dict.fromkeys(held, RAISE_CASH_COST_RATE)
        plan = core_alloc.plan_raise_cash(snap.session, held, {s: snap.closes[s] for s in held},
                                          snap.cash - self.policy.cash_reserve_usd, amount, self.policy.mix,
                                          cost_rate=rates, buffer_symbol=self.policy.buffer_symbol,
                                          min_order_usd=self.policy.min_order_usd)
        return self._admit(plan, snap, kind="raise_cash")

    # --- 4. execution ---------------------------------------------------------------------------------------

    async def tick(self) -> None:
        """Called every few seconds by the app's loop: moves the current plan along."""
        record = self.store.data.plan
        if not record:
            return
        now_ny = self._now().astimezone(NEW_YORK)
        execute_on = date.fromisoformat(record["execute_on"])
        if record["status"] in ("approved", "awaiting_approval") and now_ny.date() > execute_on:
            self._close(record, "expired", forced=True)  # missed its session: re-decided at the next close
            return
        if now_ny.date() != execute_on:
            return
        if record["status"] == "approved" and tier0_core.SUBMIT_FROM <= now_ny.time() <= tier0_core.SUBMIT_UNTIL:
            await self._start(record)
        elif record["status"] == "selling":
            await self._after_sells(record, now_ny)
        elif record["status"] == "buying":
            await self._after_buys(record, now_ny)

    async def _start(self, record: dict[str, Any]) -> None:
        plan = core_paper.plan_from_json(record["plan"])
        try:
            await self.guard.check(Intent.OPEN if plan.buys else Intent.REDUCE)
        except GuardBlocked as exc:
            # The breaker blocks any plan that buys, whole: deferred, re-decided at the next close (§3.1 a).
            # The kill switch blocks everything.
            self._close(record, "deferred" if exc.code != "halted" else "halted", forced=exc.code != "halted",
                        reason=f"{exc.code}: {exc.detail}")
            return
        closes = {s: Decimal(c) for s, c in record["closes"].items()}
        for s, q in sorted(plan.sells.items()):
            await self._submit(record, core_paper.sell_payload(record["id"], s, q, closes[s]))
        if plan.sells:
            record["status"] = "selling"
        else:
            await self._buy(record, plan, closes, preopen=True)
        self.store.save()

    async def _submit(self, record: dict[str, Any], payload: dict[str, Any]) -> None:
        intent = Intent.OPEN if payload["side"] == "buy" else Intent.REDUCE
        await self.guard.check(intent)  # again at the one place orders leave (as the news router does)
        order = await self.broker.submit_order(payload)
        record["orders"][payload["client_order_id"]] = {"symbol": payload["symbol"], "side": payload["side"],
                                                        "status": str(order.get("status")), "payload": payload}
        self.journal.write("order_submitted", plan_id=record["id"], payload=payload, alpaca_status=order.get("status"))

    async def _refresh(self, record: dict[str, Any], side: str) -> list[dict[str, Any]]:
        out = []
        for cid, o in record["orders"].items():
            if o["side"] != side:
                continue
            latest = await self.broker.order_by_client_id(cid)
            if latest is not None:
                o["status"] = str(latest.get("status"))
                o["filled_qty"] = str(latest.get("filled_qty") or "0")
                o["filled_avg_price"] = str(latest.get("filled_avg_price") or "")
            out.append(o)
        return out

    async def _after_sells(self, record: dict[str, Any], now_ny: datetime) -> None:
        sells = await self._refresh(record, "sell")
        done = all(o["status"] in ("filled", "canceled", "expired", "rejected") for o in sells)
        open_at = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        if not done and now_ny < open_at + timedelta(minutes=SELL_DEADLINE_MINUTES):
            self.store.save()
            return
        if not done:
            await self.broker.cancel_all_orders()
            self.journal.write("sells_cancelled_at_deadline", plan_id=record["id"])
        plan = core_paper.plan_from_json(record["plan"])
        await self._buy(record, plan, {s: Decimal(c) for s, c in record["closes"].items()}, preopen=False)
        self.store.save()

    async def _buy(self, record: dict[str, Any], plan: core_alloc.Plan, closes: dict[str, Decimal], *,
                   preopen: bool) -> None:
        if not plan.buys:
            self._close(record, "done")
            return
        cash = Decimal(str((await self.broker.account())["cash"]))
        allowance = core_alloc.CENT * (4 * len(plan.buys) + 4)
        buys = core_alloc.scale_buys(plan.buys, cash - self.policy.cash_reserve_usd - allowance,
                                     dict.fromkeys(plan.buys, FEE_ALLOWANCE))
        for s, notional in sorted(buys.items()):
            if notional < self.policy.min_order_usd:
                continue
            limit = core_paper.preopen_limit("buy", closes[s]) if preopen else \
                core_paper.marketable_buy_limit(Decimal(str((await self.broker.latest_quote(s)).ask)))
            payload = core_paper.buy_payload(record["id"], s, notional, limit)
            if payload is not None:
                await self._submit(record, payload)
        record["status"] = "buying"

    async def _after_buys(self, record: dict[str, Any], now_ny: datetime) -> None:
        buys = await self._refresh(record, "buy")
        done = all(o["status"] in ("filled", "canceled", "expired", "rejected") for o in buys)
        open_at = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        if not done and now_ny < open_at + timedelta(minutes=BUY_DEADLINE_MINUTES):
            self.store.save()
            return
        if not done:
            await self.broker.cancel_all_orders()
            self.journal.write("buys_cancelled_at_deadline", plan_id=record["id"])
        self._close(record, "done")

    def _close(self, record: dict[str, Any], status: str, *, forced: bool = False, reason: str | None = None) -> None:
        record["status"] = status
        state = self.store.data
        if status == "done":
            plan = core_paper.plan_from_json(record["plan"])
            filled = [o for o in record["orders"].values() if o.get("status") == "filled"]
            traded = sum((Decimal(o["filled_qty"]) * Decimal(o["filled_avg_price"]) for o in filled
                          if o.get("filled_avg_price")), Decimal(0))
            state.history.append({"day": record["execute_on"], "orders": len(record["orders"]), "kind": record["kind"],
                                  "traded_usd": str(traded.quantize(core_alloc.CENT, ROUND_DOWN)),
                                  "redecision": record.get("redecision", False)})
            if record["kind"] != "raise_cash":
                state.targets = {s: str(w) for s, w in plan.targets.items()}
        state.forced = forced
        state.plan = None
        self.store.save()
        self.journal.write(f"plan_{status}", plan_id=record["id"], reason=reason, orders=record["orders"])

    # --- the shadow line (design §8.4.5), after the day's bars are published -------------------------------------

    def shortfall(self, order: dict[str, Any], open_price: Decimal, half_spread: Decimal,
                  slippage: Decimal) -> dict[str, Decimal] | None:
        if order.get("status") != "filled" or not order.get("filled_avg_price"):
            return None
        return core_paper.shortfall_bps(order["side"], Decimal(order["filled_avg_price"]), open_price, half_spread,
                                        slippage)
