"""RiskRouter: the one component in the swarm that places orders.

Open path (``handle_signal``)::

    guard(OPEN) → entry gate → market open? → portfolio snapshot + quote
    (fetched concurrently) → portfolio limits → tier0.plan_buy → _submit

Exit path (``run_exit_pass`` every 15 s, ``close_position`` on demand)::

    guard(REDUCE) → stop / take-profit / final 10 minutes → market SELL → _submit

Holding period is intraday by design: the return model predicts the move
to the entry session's close, so the router opens nothing in a session's
last 30 minutes and sells everything in its last 10 (``tier0``).

Two concurrency rules keep this correct:

* **Every order goes through ``_submit``,** which re-runs the guard
  immediately before the network call. The route-level guard is the fast
  refusal; this one is the invariant.
* **Decisions are serialised by one ``asyncio.Lock``.** ``async`` is not
  race-free: two signals for TSLA arriving together would both read "no
  TSLA position" and both buy. Holding the lock from the exposure read to
  the order's acknowledgement makes check-then-act atomic. That is also
  why the router runs as exactly one replica (see deploy/k8s): a lock
  cannot serialise two processes.

The lock serialises decisions, not the service. While a decision awaits
Alpaca, the event loop keeps answering health checks and accepting (and
queueing) further requests, because every broker call is a non-blocking
``await`` on httpx.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Final

import tier0
from risk_router.alpaca_async import AlpacaError, AsyncAlpaca
from risk_router.guards import NEW_YORK, ExecutionGuard, GuardBlocked, Intent
from risk_router.policy import PolicyRejection, PortfolioSnapshot, check_can_open
from risk_router.schemas import Decision, TradeSignal

logger = logging.getLogger("risk_router.gatekeeper")

#: A signal older than this describes a market that has moved on.
MAX_SIGNAL_AGE: Final[timedelta] = timedelta(seconds=120)
#: IEX quotes on liquid names update many times a second in session; a
#: minute-old one means the feed is stale, not that the market is quiet.
MAX_QUOTE_AGE: Final[timedelta] = timedelta(seconds=60)
EXIT_INTERVAL_SECONDS: Final[float] = 15.0

OnSubmitted = Callable[[tier0.ExecutionPlan, TradeSignal, dict[str, Any]], Awaitable[None]]


def buy_client_order_id(signal_id: str) -> str:
    """Deterministic in the signal: a redelivered signal maps to the same
    id, which Alpaca refuses to accept twice."""
    return "rr-" + hashlib.sha256(f"open:{signal_id}".encode()).hexdigest()[:32]


def _fmt(value: Decimal) -> str:
    # Trailing zeros dropped ("2.000000000" → "2", so a whole-share OTO is
    # unambiguously whole); "f" so it is never scientific notation.
    return format(value.normalize(), "f")


def buy_order_payload(plan: tier0.ExecutionPlan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": plan.symbol,
        "qty": _fmt(plan.quantity),
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",  # the only TIF Alpaca allows on fractional qty
        "limit_price": _fmt(plan.limit_price),
        "client_order_id": plan.client_order_id,
    }
    if plan.stop_loss_kind is tier0.StopLossKind.NATIVE_OTO:
        payload["order_class"] = "oto"
        payload["stop_loss"] = {"stop_price": _fmt(plan.stop_price)}
    return payload


def exit_order_payload(symbol: str, qty: Decimal) -> dict[str, Any]:
    # Market, not limit: an exit that doesn't fill is not an exit.
    return {
        "symbol": symbol,
        "qty": _fmt(qty),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": "rx-" + uuid.uuid4().hex,
    }


def _order_summary(order: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "client_order_id", "symbol", "side", "type", "qty", "limit_price", "status", "submitted_at")
    return {k: (str(order[k]) if isinstance(order.get(k), Decimal) else order.get(k)) for k in keys if k in order}


class RiskRouter:
    def __init__(
        self,
        alpaca: AsyncAlpaca,
        guard: ExecutionGuard,
        *,
        on_submitted: OnSubmitted | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.alpaca = alpaca
        self.guard = guard
        self._on_submitted = on_submitted
        self._now = now
        self._lock = asyncio.Lock()

    def _decision(self, decision: str, code: str, detail: str, symbol: str, intent: Intent, **extra: Any) -> Decision:
        return Decision(decision=decision, code=code, detail=detail, symbol=symbol,
                        intent=intent.value, decided_at=self._now(), **extra)

    async def _submit(self, payload: dict[str, Any], intent: Intent) -> dict[str, Any]:
        """The only call site that sends an order to Alpaca."""
        await self.guard.check(intent)
        return await self.alpaca.submit_order(payload)

    # --- open --------------------------------------------------------------

    async def handle_signal(self, signal: TradeSignal) -> Decision:
        """Decide on one signal. Raises ``GuardBlocked`` when halted or
        when the breaker forbids new positions; every other refusal is a
        normal ``rejected`` decision."""
        symbol = signal.symbol
        async with self._lock:
            await self.guard.check(Intent.OPEN)
            now = self._now()

            rejection = self._entry_gate(signal, now)
            if rejection is not None:
                return self._decision("rejected", *rejection, symbol, Intent.OPEN)

            clock = await self.alpaca.clock()
            if not clock.get("is_open"):
                return self._decision("rejected", "market_closed",
                                      f"market is closed; next open {clock.get('next_open')}", symbol, Intent.OPEN)
            closes_at = session_close(clock)
            if closes_at is not None and now >= closes_at - tier0.LAST_ENTRY_BEFORE_CLOSE:
                # The model predicts the move to this session's close; a
                # position opened now would be sold before it could happen.
                return self._decision("rejected", "entry_window_closed",
                                      f"no new positions after {closes_at - tier0.LAST_ENTRY_BEFORE_CLOSE:%H:%M %Z}",
                                      symbol, Intent.OPEN)

            # Independent reads, issued together: four round-trips cost one.
            day_start = datetime.combine(now.astimezone(NEW_YORK).date(), time(0), NEW_YORK)
            since = min(day_start, now - timedelta(seconds=tier0.REENTRY_COOLDOWN_SECONDS))
            positions, open_orders, recent_orders, quote = await asyncio.gather(
                self.alpaca.positions(),
                self.alpaca.orders(status="open"),
                self.alpaca.orders(status="all", after=since),
                self.alpaca.latest_quote(symbol),
            )

            snapshot = PortfolioSnapshot.from_alpaca(
                positions=positions, open_orders=open_orders,
                recent_orders=recent_orders, day_start=day_start,
            )
            try:
                check_can_open(snapshot, symbol, now)
            except PolicyRejection as exc:
                return self._decision("rejected", exc.code, exc.detail, symbol, Intent.OPEN)

            if now - quote.timestamp > MAX_QUOTE_AGE:
                return self._decision("rejected", "stale_quote",
                                      f"latest quote is from {quote.timestamp.isoformat()}", symbol, Intent.OPEN)

            try:
                plan = tier0.plan_buy(symbol=symbol, bid=quote.bid, ask=quote.ask, atr=signal.atr,
                                      client_order_id=buy_client_order_id(signal.signal_id))
            except tier0.Tier0Rejection as exc:
                return self._decision("rejected", exc.code.value, exc.detail, symbol, Intent.OPEN)

            order = await self._submit(buy_order_payload(plan), Intent.OPEN)

        logger.info("OPEN %s qty=%s @ %s (order %s, signal %s from %s)",
                    symbol, plan.quantity, plan.limit_price, order.get("id"), signal.signal_id, signal.source)
        # Outside the lock: a slow PFW must not hold up the next decision.
        if self._on_submitted is not None:
            await self._on_submitted(plan, signal, order)
        return self._decision("accepted", "submitted", f"limit buy {plan.quantity} @ {plan.limit_price}",
                              symbol, Intent.OPEN, order=_order_summary(order), plan=plan.as_dict())

    @staticmethod
    def _entry_gate(signal: TradeSignal, now: datetime) -> tuple[str, str] | None:
        if now - signal.created_at > MAX_SIGNAL_AGE:
            return "stale_signal", f"signal created at {signal.created_at.isoformat()}"
        prob_up = Decimal(str(signal.prob_up))
        if prob_up < tier0.ROUTER_MIN_PROB_UP:
            return "below_min_prob", f"prob_up {prob_up} is below {tier0.ROUTER_MIN_PROB_UP}"
        if signal.predicted_move_pct <= 0:
            return "non_positive_edge", f"predicted move {signal.predicted_move_pct}% is not positive"
        return None

    # --- reduce ------------------------------------------------------------

    async def close_position(self, symbol: str, *, reason: str) -> Decision:
        """Market-sell everything available in ``symbol``."""
        symbol = symbol.strip().upper()
        async with self._lock:
            await self.guard.check(Intent.REDUCE)
            clock, positions, open_orders = await asyncio.gather(
                self.alpaca.clock(), self.alpaca.positions(), self.alpaca.orders(status="open"),
            )
            if not clock.get("is_open"):
                return self._decision("rejected", "market_closed", "market is closed", symbol, Intent.REDUCE)
            position = next((p for p in positions if p["symbol"] == symbol), None)
            if position is None:
                return self._decision("rejected", "no_position", f"no open {symbol} position", symbol, Intent.REDUCE)
            return await self._exit_locked(position, open_orders, reason)

    async def run_exit_pass(self) -> list[Decision]:
        """One sweep: sell any long position at or past its stop-loss or
        take-profit, and everything in the session's final minutes. This is
        what finally enforces the ``engine_tracked`` stops the monolith only
        wrote on its receipts."""
        if self.guard.state.halted:
            return []
        clock = await self.alpaca.clock()
        if not clock.get("is_open"):
            return []
        closes_at = session_close(clock)
        closing = closes_at is not None and self._now() >= closes_at - tier0.SESSION_EXIT_BEFORE_CLOSE

        decisions: list[Decision] = []
        async with self._lock:
            positions, open_orders = await asyncio.gather(self.alpaca.positions(), self.alpaca.orders(status="open"))
            for position in positions:
                reason = exit_reason(position) or ("session_close" if closing else None)
                if reason is None:
                    continue
                try:
                    decisions.append(await self._exit_locked(position, open_orders, reason))
                except GuardBlocked:
                    raise
                except AlpacaError as exc:
                    logger.error("Exit of %s (%s) failed: %s", position.get("symbol"), reason, exc)
        return decisions

    async def _exit_locked(self, position: dict[str, Any], open_orders: list[dict[str, Any]], reason: str) -> Decision:
        symbol = position["symbol"]
        if position.get("side", "long") != "long":
            return self._decision("rejected", "not_long", f"{symbol} is not a long position", symbol, Intent.REDUCE)
        if any(o["symbol"] == symbol and o.get("side") == "sell" for o in open_orders):
            return self._decision("rejected", "exit_in_flight", f"a sell for {symbol} is already working",
                                  symbol, Intent.REDUCE)
        # qty_available excludes shares already held for other orders, so
        # this can never sell more than the account owns.
        qty = Decimal(str(position.get("qty_available", position.get("qty", 0))))
        if qty <= 0:
            return self._decision("rejected", "nothing_available", f"no {symbol} shares available to sell",
                                  symbol, Intent.REDUCE)
        order = await self._submit(exit_order_payload(symbol, qty), Intent.REDUCE)
        logger.warning("EXIT %s qty=%s (%s), order %s", symbol, qty, reason, order.get("id"))
        return self._decision("accepted", reason, f"market sell {qty} {symbol}", symbol, Intent.REDUCE,
                              order=_order_summary(order))

    async def exit_loop(self) -> None:
        """Runs until cancelled. One failed pass never stops the loop."""
        while True:
            try:
                await self.run_exit_pass()
            except asyncio.CancelledError:
                raise
            except GuardBlocked as exc:
                logger.warning("Exit pass blocked: %s", exc)
            except Exception:
                logger.exception("Exit pass failed; retrying next interval")
            await asyncio.sleep(EXIT_INTERVAL_SECONDS)


def session_close(clock: dict[str, Any]) -> datetime | None:
    """While the market is open, Alpaca's ``next_close`` is today's close —
    16:00, or 13:00 on a half-day."""
    raw = clock.get("next_close")
    return datetime.fromisoformat(raw) if raw else None


def exit_reason(position: dict[str, Any]) -> str | None:
    """``stop_loss`` / ``take_profit`` / ``None`` for one Alpaca position,
    by the move from average entry to the current price."""
    entry = Decimal(str(position["avg_entry_price"]))
    current = Decimal(str(position["current_price"]))
    if entry <= 0:
        return None
    change_pct = (current - entry) / entry * Decimal(100)
    if change_pct <= -tier0.STOP_LOSS_PCT:
        return "stop_loss"
    if change_pct >= tier0.TAKE_PROFIT_PCT:
        return "take_profit"
    return None
