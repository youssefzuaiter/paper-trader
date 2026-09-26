"""The replay (design §4): each session minute by minute through the production router.

At each minute boundary *t*, with the router's clock at *t*:

1. ``SimAlpaca.advance(t)`` resolves the bar that just ended for every
   working order (fills are stamped with the bar's end) and marks positions;
2. exits: the production ``run_exit_pass`` at the bar's low, then at its
   close (stops fire on any touch, take-profits on a close through);
3. entries: signals decided at *t* (created in (t - 1 min, t]) go to
   ``handle_signal``, highest ``prob_up`` first, then by symbol.

**Minute skipping** (L12) only skips work that provably does nothing: a
boundary with no working order, no position and no signal is skipped
outright; with positions, the exit passes are skipped when the production
``exit_reason`` finds nothing at either mark and the session-close window
has not begun. Every ``Decision``, rejections included, is recorded.

The router is the production class, constructed as production does except
for three arguments: the broker (the production client over ``SimAlpaca``),
the clock, and the gate threshold (D2). One router per session, because the
walk-forward's threshold is per fold; the broker, the guard and its breaker
latch persist across sessions like the live pod's.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import tier0
from backtest.calendar import Session
from backtest.costs import CostLevel, FeeTable
from backtest.data import MarketData
from backtest.sim_broker import Fill, SimAlpaca, SimOrder
from risk_router.gatekeeper import RiskRouter, exit_reason
from risk_router.guards import CircuitBreaker, ExecutionGuard, GuardBlocked
from risk_router.schemas import Decision, TradeSignal
from risk_router.state import StateStore

logger = logging.getLogger("backtest.engine")

_EXIT_ID = re.compile(r"^rx-[0-9a-f]{32}$")


@dataclass(frozen=True)
class PendingSignal:
    """A signal and the instant the strategy hands it to the router."""

    emit_at: datetime
    signal: TradeSignal
    prediction_id: str | None = None
    sources: tuple[str, ...] = ()   # the events behind it (an S2 signal aggregates a night)


class Strategy(Protocol):
    name: str

    def min_prob_up(self, session: Session) -> Decimal:
        """The router's gate for this session (the fold's threshold)."""

    def signals_for(self, session: Session, market: MarketData) -> list[PendingSignal]:
        """Every signal of ``session``, each built from a ``PitView`` at its ``emit_at``."""


@dataclass(frozen=True)
class EngineParams:
    latency_bars: int = 1
    skip_idle: bool = True
    capital_base: Decimal = Decimal("50")     # tier0.MAX_GROSS_EXPOSURE_USD: the most S2/S3 can hold


@dataclass
class DayResult:
    date: date
    equity: Decimal
    pnl: Decimal
    fees: Decimal
    entries: int
    carried: list[str]


@dataclass
class RunResult:
    run_id: str
    strategy: str
    level: str
    days: list[DayResult]
    rows: list[dict[str, Any]]
    rejections: Counter[str]
    fees_paid: dict[str, Decimal]
    requests: int
    params: dict[str, Any] = field(default_factory=dict)

    def daily_returns(self, capital_base: Decimal) -> list[float]:
        return [float(d.pnl / capital_base) for d in self.days]


class Recorder:
    """The trade state machine as ``trade_event`` rows (design §7):
    ``signal → rejected(code)`` or ``signal → submitted → filled | expired →
    exit_submitted(reason) → exit_filled``. The live router's exit ids are
    random (``rx-`` + uuid4); they are recorded by sequence number instead,
    so a reproduction compares every field."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.rows: list[dict[str, Any]] = []
        self.rejections: Counter[str] = Counter()
        self._seq: dict[str, int] = {}
        self._trade_of_order: dict[str, str] = {}
        self._trade_of_client: dict[str, str] = {}
        self._open_trade: dict[str, str] = {}
        self._exit_ids: dict[str, str] = {}
        self.entries_today = 0

    def _add(self, trade_id: str, at: datetime, state: str, symbol: str, **fields: Any) -> None:
        seq = self._seq.get(trade_id, 0)
        self._seq[trade_id] = seq + 1
        self.rows.append({"run_id": self.run_id, "trade_id": trade_id, "seq": seq, "occurred_at": at,
                          "state": state, "symbol": symbol, **fields})

    def _canonical(self, client_order_id: str | None) -> str | None:
        if client_order_id is None or not _EXIT_ID.match(client_order_id):
            return client_order_id
        return self._exit_ids.setdefault(client_order_id, f"rx-{len(self._exit_ids) + 1:08d}")

    def _decision_detail(self, decision: Decision) -> dict[str, Any]:
        detail = decision.model_dump(mode="json")
        if detail.get("order") and detail["order"].get("client_order_id"):
            detail["order"]["client_order_id"] = self._canonical(detail["order"]["client_order_id"])
        return detail

    def signal(self, pending: PendingSignal, at: datetime) -> None:
        s = pending.signal
        self._add(s.signal_id, at, "signal", s.symbol, prediction_id=pending.prediction_id, detail={
            "prob_up": s.prob_up, "predicted_move_pct": s.predicted_move_pct,
            "atr": None if s.atr is None else str(s.atr), "created_at": s.created_at.isoformat(),
            "emit_at": pending.emit_at.isoformat(), "sources": list(pending.sources)})

    def entry_decision(self, pending: PendingSignal, decision: Decision) -> None:
        if decision.decision == "rejected":
            self.rejections[decision.code] += 1
            self._add(pending.signal.signal_id, decision.decided_at, "rejected", decision.symbol,
                      reason_code=decision.code, detail=self._decision_detail(decision))

    def blocked(self, pending: PendingSignal, at: datetime, exc: GuardBlocked) -> None:
        self.rejections[exc.code] += 1
        self._add(pending.signal.signal_id, at, "rejected", pending.signal.symbol, reason_code=exc.code,
                  detail={"guard": exc.detail})

    async def on_submitted(self, plan: tier0.ExecutionPlan, signal: TradeSignal, order: dict[str, Any]) -> None:
        """The router's own post-submit hook: the same seam the live writers will use."""
        trade_id = signal.signal_id
        self._trade_of_order[order["id"]] = trade_id
        self._trade_of_client[order["client_order_id"]] = trade_id
        self._open_trade[signal.symbol] = trade_id
        self.entries_today += 1
        self._add(trade_id, datetime.fromisoformat(order["submitted_at"]), "submitted", signal.symbol, side="buy",
                  qty=plan.quantity, price=plan.limit_price, order_id=order["id"],
                  client_order_id=order["client_order_id"], detail={"plan": plan.as_dict()})

    def exit_decision(self, decision: Decision, at: datetime) -> None:
        trade_id = self._open_trade.get(decision.symbol, f"untracked:{decision.symbol}")
        detail = self._decision_detail(decision)
        if decision.decision == "accepted":
            order = decision.order or {}
            self._trade_of_order[order["id"]] = trade_id
            self._add(trade_id, at, "exit_submitted", decision.symbol, reason_code=decision.code, side="sell",
                      qty=Decimal(str(order["qty"])), order_id=order["id"],
                      client_order_id=self._canonical(order.get("client_order_id")), detail=detail)
        else:
            self._add(trade_id, at, "exit_rejected", decision.symbol, reason_code=decision.code, detail=detail)

    def fill(self, fill: Fill) -> None:
        order = fill.order
        trade_id = self._trade_of_order.get(order.id, f"untracked:{order.symbol}")
        qty = order.qty
        self._add(trade_id, fill.at, "filled" if order.side == "buy" else "exit_filled", order.symbol,
                  side=order.side, qty=qty, price=fill.price.price, ref_price=fill.price.ref, cash=fill.cash,
                  fees=sum(fill.fees.values(), Decimal(0)), spread_cost=qty * fill.price.spread_cost,
                  slippage_cost=qty * fill.price.slippage_cost, order_id=order.id,
                  client_order_id=self._canonical(order.client_order_id),
                  detail={"bar_start": fill.bar_start.isoformat(), "fees": {k: str(v) for k, v in fill.fees.items()}})
        if order.side == "sell" and self._open_trade.get(order.symbol) == trade_id:
            del self._open_trade[order.symbol]

    def expire(self, order: SimOrder) -> None:
        trade_id = self._trade_of_order.get(order.id, f"untracked:{order.symbol}")
        self._add(trade_id, order.expired_at, "expired" if order.side == "buy" else "exit_expired", order.symbol,
                  side=order.side, qty=order.qty, order_id=order.id,
                  client_order_id=self._canonical(order.client_order_id))
        if order.side == "buy" and self._open_trade.get(order.symbol) == trade_id:
            del self._open_trade[order.symbol]

    def end_session(self, session: Session, summary: dict[str, Any]) -> None:
        if summary["fee_rounding"]:
            self._add("fees", session.close_at, "fee_rounding", "*", fees=summary["fee_rounding"])
        for symbol in summary["carried"]:
            self._add(self._open_trade.get(symbol, f"untracked:{symbol}"), session.close_at, "carried_overnight",
                      symbol)


class Engine:
    def __init__(self, market: MarketData, strategy: Strategy, level: CostLevel, fees: FeeTable,
                 params: EngineParams | None = None, *, run_id: str = "run") -> None:
        self.market = market
        self.strategy = strategy
        self.level = level
        self.fees = fees
        self.params = params or EngineParams()
        self.run_id = run_id

    async def run(self, sessions: Sequence[Session]) -> RunResult:
        recorder = Recorder(self.run_id)
        sim = SimAlpaca(self.market, self.level, self.fees, latency_bars=self.params.latency_bars,
                        on_fill=recorder.fill, on_expire=recorder.expire)
        client = sim.client()
        days: list[DayResult] = []
        with tempfile.TemporaryDirectory(prefix="backtest-router-") as tmp:
            state = StateStore(Path(tmp) / "router-state.json")
            breaker = CircuitBreaker(client, state, now=lambda: sim.now, monotonic=lambda: sim.now.timestamp())
            guard = ExecutionGuard(state, breaker)
            try:
                for session in sessions:
                    router = RiskRouter(client, guard, on_submitted=recorder.on_submitted, now=lambda: sim.now,
                                        min_prob_up=self.strategy.min_prob_up(session))
                    equity_before = sim.last_equity
                    fees_before = sum(sim.fees_paid.values(), Decimal(0))
                    recorder.entries_today = 0
                    await self._session(session, sim, router, recorder)
                    summary = sim.end_session(session)
                    recorder.end_session(session, summary)
                    if summary["carried"]:
                        logger.warning("%s: positions carried past the close: %s", session.date, summary["carried"])
                    days.append(DayResult(session.date, sim.last_equity, sim.last_equity - equity_before,
                                          sum(sim.fees_paid.values(), Decimal(0)) - fees_before,
                                          recorder.entries_today, summary["carried"]))
            finally:
                await client.aclose()
        return RunResult(self.run_id, self.strategy.name, self.level.name, days, recorder.rows, recorder.rejections,
                         dict(sim.fees_paid), sim.requests,
                         {"latency_bars": self.params.latency_bars, "skip_idle": self.params.skip_idle})

    async def _session(self, session: Session, sim: SimAlpaca, router: RiskRouter, recorder: Recorder) -> None:
        by_boundary: dict[datetime, list[PendingSignal]] = {}
        for pending in self.strategy.signals_for(session, self.market):
            boundary = _ceil_minute(pending.emit_at)
            if not session.open_at <= boundary <= session.close_at:
                raise ValueError(f"{pending.signal.signal_id} is decided at {boundary}, outside {session.date}")
            by_boundary.setdefault(boundary, []).append(pending)
        closing_from = session.close_at - tier0.SESSION_EXIT_BEFORE_CLOSE
        t = session.open_at
        while t <= session.close_at:
            due = by_boundary.pop(t, [])
            if not (self.params.skip_idle and not due and not sim._working and not sim.positions):
                sim.advance(t)
                if sim.positions and self._exits_needed(sim, t, closing_from):
                    for mode in ("low", "close"):
                        sim.mark(mode)
                        for decision in await router.run_exit_pass():
                            recorder.exit_decision(decision, t)
                for pending in sorted(due, key=lambda p: (-p.signal.prob_up, p.signal.symbol)):
                    recorder.signal(pending, t)
                    try:
                        decision = await router.handle_signal(pending.signal)
                    except GuardBlocked as exc:
                        recorder.blocked(pending, t, exc)
                        continue
                    recorder.entry_decision(pending, decision)
            t += timedelta(minutes=1)

    def _exits_needed(self, sim: SimAlpaca, t: datetime, closing_from: datetime) -> bool:
        """Skip the exit passes only when they would provably do nothing (L12)."""
        return not self.params.skip_idle or t >= closing_from or _exit_due(sim)


def _exit_due(sim: SimAlpaca) -> bool:
    """Would the production ``exit_reason`` fire at the last bar's low or close?"""
    for symbol, position in sim.positions.items():
        j = sim._last_bar.get(symbol)
        if j is None:
            continue
        series = sim.bars[symbol]
        entry = str(position.avg_entry_price)
        for price in (series.l[j], series.c[j]):
            if exit_reason({"avg_entry_price": entry, "current_price": repr(float(price))}) is not None:
                return True
    return False


def _ceil_minute(ts: datetime) -> datetime:
    floor = ts.replace(second=0, microsecond=0)
    return floor if floor == ts else floor + timedelta(minutes=1)
