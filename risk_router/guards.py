"""The execution guard: kill switch + daily circuit breaker, per order intent.

Every order the router places — from ``POST /v1/signals``, from a manual
``POST /v1/positions/{symbol}/close``, or from the background exit monitor
— passes ``ExecutionGuard.check`` twice: once as a FastAPI dependency on
the route (so a blocked request is refused before any work is done), and
again inside ``RiskRouter._submit``, the one function that sends orders to
Alpaca (so a code path that never went through HTTP is covered too).

The guard distinguishes what an order *does*:

==========  =================  ==================================
Intent      Kill switch        Circuit breaker
==========  =================  ==================================
``OPEN``    blocks             blocks (and fails closed)
``REDUCE``  blocks             allows
==========  =================  ==================================

A breaker that blocked exits would stop the stop-losses from firing on the
very day they matter most. The kill switch blocks everything: it means a
human has taken over.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

import tier0
from risk_router.alpaca_async import AsyncAlpaca
from risk_router.state import StateStore

logger = logging.getLogger("risk_router.guards")

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")

#: How long one account read serves breaker checks. Short enough that a
#: fast drawdown is seen within seconds; long enough that a burst of
#: signals doesn't hit /v2/account once each.
BREAKER_CACHE_SECONDS: Final[float] = 10.0


class Intent(StrEnum):
    OPEN = "open"      # adds exposure: any buy
    REDUCE = "reduce"  # removes exposure: selling a long position


class GuardBlocked(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def trading_day(now: datetime) -> date:
    return now.astimezone(NEW_YORK).date()


class CircuitBreaker:
    """Daily P&L breaker over the whole account, latched for the day.

    Unlike the monolith's loop, which carries on when the account can't be
    read, this fails CLOSED for new positions: a gatekeeper that cannot
    see today's P&L cannot know the breaker hasn't tripped.
    """

    def __init__(
        self,
        alpaca: AsyncAlpaca,
        state: StateStore,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._alpaca = alpaca
        self._state = state
        self._now = now
        self._monotonic = monotonic
        self._cached: tuple[float, Decimal] | None = None

    async def daily_pnl_pct(self) -> Decimal:
        if self._cached is not None and self._monotonic() - self._cached[0] < BREAKER_CACHE_SECONDS:
            return self._cached[1]
        account = await self._alpaca.account()
        equity = Decimal(str(account["equity"]))
        last_equity = Decimal(str(account["last_equity"]))
        if last_equity <= 0:
            raise ValueError(f"last_equity={last_equity} is not positive")
        pnl = (equity - last_equity) / last_equity * Decimal(100)
        self._cached = (self._monotonic(), pnl)
        return pnl

    def latched_today(self) -> bool:
        return self._state.breaker_tripped_on() == trading_day(self._now())

    async def check_open(self) -> None:
        today = trading_day(self._now())
        if self._state.breaker_tripped_on() == today:
            raise GuardBlocked("circuit_breaker", self._state.breaker_detail or "tripped earlier today")
        try:
            pnl = await self.daily_pnl_pct()
        except Exception as exc:
            raise GuardBlocked("breaker_unavailable", f"cannot read today's P&L: {exc}") from exc
        if pnl <= tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT:
            detail = (f"daily P&L {pnl:.2f}% hit the {tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT}% "
                      f"breaker; no new positions until the next trading day")
            self._state.trip_breaker(today, detail)
            logger.critical("CIRCUIT BREAKER TRIPPED: %s", detail)
            raise GuardBlocked("circuit_breaker", detail)


class ExecutionGuard:
    def __init__(self, state: StateStore, breaker: CircuitBreaker) -> None:
        self.state = state
        self.breaker = breaker

    async def check(self, intent: Intent) -> None:
        """Raise ``GuardBlocked`` if an order with this intent may not go out."""
        if self.state.halted:
            raise GuardBlocked("halted", self.state.halt_reason or "emergency halt is active")
        if intent is Intent.OPEN:
            await self.breaker.check_open()

    def halt(self, reason: str, *, now: datetime | None = None) -> None:
        when = (now or datetime.now(UTC)).isoformat()
        self.state.set_halted(when, reason)
        logger.critical("EMERGENCY HALT: %s", reason)
