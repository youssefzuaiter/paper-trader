"""The long-term core's paper-mode decision, shared by the Allocator and the core router (design §8).

The Allocator proposes a plan after the close; the router recomputes it from its *own* reads of the
account and the day's bars and rejects any difference (``plan_mismatch``). Both call ``decide`` here,
and ``decide`` calls ``core_alloc``'s functions, the same ones the backtest engine calls (check C12),
so paper trading runs the evidence's arithmetic, not a re-implementation of it.

Also here: the order payloads (limit orders only, per the system design §7.4: pre-open collared
limits that fill at the opening print, then marketable limits for buys), deterministic client order
ids for at-most-once submission, and the shadow line's implementation shortfall.

Pure standard library: the router image imports it without numpy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Final, Protocol
from zoneinfo import ZoneInfo

import core_alloc
import tier0_core
from tier0_core import CorePolicy

CENT: Final[Decimal] = Decimal("0.01")
QTY: Final[Decimal] = core_alloc.QTY_QUANTUM
NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Snapshot:
    """Everything a decision reads, as of one session's close."""
    session: date
    next_session: date
    qty: dict[str, Decimal]          # units held (Alpaca positions)
    cash: Decimal                    # account cash
    closes: dict[str, Decimal]       # the session's closes, every policy symbol
    forced: bool = False             # the last plan was deferred by the breaker: re-decide now
    initial: bool = False            # nothing held yet: the initial build
    targets: dict[str, Decimal] = field(default_factory=dict)  # the last rebalance's targets ({} = the policy mix)

    def digest(self) -> str:
        body = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(self).items()}
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class SnapshotError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class Reads(Protocol):
    async def account(self) -> dict[str, Any]: ...
    async def positions(self) -> list[dict[str, Any]]: ...
    async def daily_bars(self, symbols: list[str], start: str, end: str) -> dict[str, list[dict[str, Any]]]: ...
    async def calendar(self, start: str, end: str) -> list[dict[str, Any]]: ...


async def read_snapshot(broker: Reads, policy: CorePolicy, session: date, *, forced: bool,
                        targets: dict[str, Decimal]) -> Snapshot:
    """The snapshot for ``session``'s close, from the account and the published daily bars. The Allocator
    and the router each call this with their own broker connection; ``forced`` and ``targets`` are the
    router's facts (it knows whether the last plan was deferred), which the Allocator asks it for."""
    sessions = await broker.calendar(session.isoformat(), (session + timedelta(days=10)).isoformat())
    days = [date.fromisoformat(s["date"]) for s in sessions]
    later = [d for d in days if d > session]
    if session not in days or not later:
        raise SnapshotError("not_a_session", f"{session} is not a session, or the next one is unknown")
    account = await broker.account()
    qty = dict.fromkeys(policy.symbols, Decimal(0))
    for p in await broker.positions():
        held = Decimal(str(p["qty"]))
        if p["symbol"] in qty:
            qty[p["symbol"]] = held
        elif held != 0:
            raise SnapshotError("foreign_position", f"{p['symbol']} is held but is not in the policy")
    bars = await broker.daily_bars(list(policy.symbols), session.isoformat(), session.isoformat())
    closes: dict[str, Decimal] = {}
    for s in policy.symbols:
        day = [b for b in bars.get(s, []) if str(b["t"])[:10] == session.isoformat()]
        if not day:
            raise SnapshotError("missing_close", f"no {session} close for {s}")
        closes[s] = Decimal(str(day[-1]["c"]))
    return Snapshot(session, min(later), qty, Decimal(str(account["cash"])), closes, forced=forced,
                    initial=not any(q for q in qty.values()), targets=dict(targets))


def last_published(sessions: list[date], now: datetime) -> date | None:
    """The most recent session whose close the Allocator may read: past, or today once it is 16:20 in New York
    (the backtest's close + 15 minutes, rounded up to a clock time the Allocator can be scheduled at)."""
    now_ny = now.astimezone(NEW_YORK)
    today = now_ny.date()
    published = [d for d in sorted(sessions) if d < today or (d == today and now_ny.time() >= tier0_core.DECIDE_AFTER)]
    return published[-1] if published else None


def skip_reason(policy: CorePolicy, snap: Snapshot) -> str | None:
    """Why there can be no plan at all for this close, or None. ``effective_from`` is the first session on
    which the policy may place an order: a plan that would execute earlier is not made, on either side of the
    router/Allocator split (both call ``decide``)."""
    if snap.next_session < policy.effective_from:
        return f"the policy takes effect {policy.effective_from}; the next session is {snap.next_session}"
    return None


def decide(policy: CorePolicy, snap: Snapshot) -> core_alloc.Plan | None:
    """The plan for this close, or None. The initial build plans fully to the policy mix; afterwards
    ``core_alloc.rebalance_due`` decides, exactly as in the backtest (drift measured with cash in the total)."""
    if skip_reason(policy, snap) is not None:
        return None
    targets = dict(policy.mix)
    usable_cash = snap.cash - policy.cash_reserve_usd
    if snap.initial:
        plan = core_alloc.plan_rebalance(snap.session, snap.qty, snap.closes, usable_cash, targets,
                                         min_order_usd=policy.min_order_usd, reason="initial")
        return None if plan.empty else plan
    weights = core_alloc.current_weights(snap.qty, snap.closes, snap.cash)
    ends = core_alloc.period_ends(snap.session, snap.next_session)
    if not core_alloc.rebalance_due(policy.rule, ends, weights, snap.targets or targets, forced=snap.forced):
        return None
    plan = core_alloc.plan_rebalance(snap.session, snap.qty, snap.closes, usable_cash, targets,
                                     min_order_usd=policy.min_order_usd)
    return None if plan.empty else plan


def plan_to_json(plan: core_alloc.Plan) -> dict[str, Any]:
    return {"decided_on": plan.decided_on.isoformat(), "reason": plan.reason,
            "targets": {s: str(w) for s, w in sorted(plan.targets.items())},
            "sells": {s: str(q) for s, q in sorted(plan.sells.items())},
            "buys": {s: str(n) for s, n in sorted(plan.buys.items())}}


def plan_from_json(raw: Mapping[str, Any]) -> core_alloc.Plan:
    return core_alloc.Plan(date.fromisoformat(raw["decided_on"]), {s: Decimal(w) for s, w in raw["targets"].items()},
                           {s: Decimal(q) for s, q in raw["sells"].items()},
                           {s: Decimal(n) for s, n in raw["buys"].items()}, raw["reason"])


def plan_id(plan: core_alloc.Plan, snap: Snapshot) -> str:
    """Content-addressed: the same decision from the same inputs always has the same id."""
    body = json.dumps({"plan": plan_to_json(plan), "inputs": snap.digest()}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


# --- orders --------------------------------------------------------------------------------------------

def client_order_id(pid: str, symbol: str, side: str) -> str:
    """Deterministic, so a lost response is looked up rather than re-sent (alpaca_async's at-most-once)."""
    return f"core-{pid}-{symbol.replace('/', '')}-{side[0]}"


def preopen_limit(side: str, prev_close: Decimal) -> Decimal:
    """Previous close ± the collar, rounded away from fill-blocking: buys up, sells down, to the cent."""
    collar = tier0_core.LIMIT_COLLAR_PCT / 100
    if side == "buy":
        return (prev_close * (1 + collar)).quantize(CENT, ROUND_UP)
    return (prev_close * (1 - collar)).quantize(CENT, ROUND_DOWN)


def marketable_buy_limit(ask: Decimal) -> Decimal:
    return (ask * (1 + tier0_core.MARKETABLE_BUFFER_PCT / 100)).quantize(CENT, ROUND_UP)


def sell_payload(pid: str, symbol: str, qty: Decimal, prev_close: Decimal) -> dict[str, Any]:
    return {"symbol": symbol, "side": "sell", "type": "limit", "time_in_force": _tif(symbol),
            "qty": str(qty.quantize(QTY, ROUND_DOWN)), "limit_price": str(preopen_limit("sell", prev_close)),
            "client_order_id": client_order_id(pid, symbol, "sell"), "extended_hours": False}


def buy_payload(pid: str, symbol: str, notional: Decimal, limit: Decimal) -> dict[str, Any] | None:
    """A buy of ``notional`` dollars (rounded down to the cent, never oversized) at no more than ``limit``.
    Alpaca takes notional amounts on limit orders, so the dollars spent are exactly the plan's."""
    dollars = notional.quantize(CENT, ROUND_DOWN)
    if dollars < tier0_core.MIN_ORDER_USD:
        return None
    return {"symbol": symbol, "side": "buy", "type": "limit", "time_in_force": _tif(symbol),
            "notional": str(dollars), "limit_price": str(limit),
            "client_order_id": client_order_id(pid, symbol, "buy"), "extended_hours": False}


def _tif(symbol: str) -> str:
    """Fractional equity orders must be day orders; Alpaca's crypto orders take gtc or ioc only."""
    return "ioc" if symbol in tier0_core.CRYPTO_SYMBOLS else "day"


# --- the shadow line (design §8.4.5) ----------------------------------------------------------------------

def shortfall_bps(side: str, fill_price: Decimal, open_price: Decimal, half_spread: Decimal,
                  slippage: Decimal) -> dict[str, Decimal]:
    """A fill against the open (the frictionless price) and against the backtest's modelled fill
    (open ± h + σ). Positive = the fill cost more than that benchmark. Measured over months, this
    checks the registered cost levels against evidence."""
    sign = Decimal(1) if side == "buy" else Decimal(-1)
    modelled = open_price * (1 + sign * (half_spread + slippage))
    return {"vs_open_bps": sign * (fill_price / open_price - 1) * 10000,
            "vs_modelled_bps": sign * (fill_price / modelled - 1) * 10000}
