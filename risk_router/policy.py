"""Portfolio limits: what the monolith never checked before buying.

``tier0.MAX_NOTIONAL_USD`` bounds one order. Nothing bounded how many of
them piled up, which is how TSLA — the one built-in scenario that always
clears the gate — got bought every ~15 minutes. These checks run over a
``PortfolioSnapshot`` built from Alpaca's own positions and order history
at decision time, so they need no local database, survive restarts, and
count orders placed by any service trading the account.

Pure functions over plain data: no I/O here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import tier0


class PolicyRejection(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _dec(value: Any) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal(0)


def _parse_ts(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class PortfolioSnapshot:
    #: symbol → absolute market value of the open position, USD
    positions: dict[str, Decimal]
    #: symbol → remaining notional of working (unfilled) BUY orders, USD
    working_buys: dict[str, Decimal]
    #: symbols with a working SELL order (an exit already in flight)
    working_sells: frozenset[str]
    #: BUY orders submitted since the start of the trading day
    buys_today: int
    #: symbol → most recent fill time of a SELL order
    last_sell_at: dict[str, datetime] = field(default_factory=dict)

    @classmethod
    def from_alpaca(
        cls,
        *,
        positions: list[dict[str, Any]],
        open_orders: list[dict[str, Any]],
        recent_orders: list[dict[str, Any]],
        day_start: datetime,
    ) -> PortfolioSnapshot:
        held = {p["symbol"]: abs(_dec(p.get("market_value"))) for p in positions}

        working_buys: dict[str, Decimal] = {}
        working_sells: set[str] = set()
        for order in open_orders:
            symbol = order["symbol"]
            if order.get("side") == "sell":
                working_sells.add(symbol)
                continue
            if order.get("notional") is not None:
                remaining = _dec(order["notional"])
            elif order.get("limit_price") is not None:
                remaining = (_dec(order.get("qty")) - _dec(order.get("filled_qty"))) * _dec(order["limit_price"])
            else:
                # A market buy has no price to multiply by. Assume the
                # worst an order may be rather than zero.
                remaining = tier0.MAX_NOTIONAL_USD
            working_buys[symbol] = working_buys.get(symbol, Decimal(0)) + remaining

        buys_today = 0
        last_sell_at: dict[str, datetime] = {}
        for order in recent_orders:
            submitted = _parse_ts(order.get("submitted_at"))
            if order.get("side") == "buy":
                if order.get("status") != "rejected" and submitted is not None and submitted >= day_start:
                    buys_today += 1
                continue
            filled_at = _parse_ts(order.get("filled_at"))
            if filled_at is not None:
                symbol = order["symbol"]
                if symbol not in last_sell_at or filled_at > last_sell_at[symbol]:
                    last_sell_at[symbol] = filled_at

        return cls(
            positions=held,
            working_buys=working_buys,
            working_sells=frozenset(working_sells),
            buys_today=buys_today,
            last_sell_at=last_sell_at,
        )

    @property
    def gross_exposure(self) -> Decimal:
        return sum(self.positions.values(), Decimal(0)) + sum(self.working_buys.values(), Decimal(0))

    @property
    def symbols_exposed(self) -> set[str]:
        return set(self.positions) | set(self.working_buys)


def check_can_open(snapshot: PortfolioSnapshot, symbol: str, now: datetime) -> None:
    """Raise ``PolicyRejection`` if a new ``MAX_NOTIONAL_USD`` buy of
    ``symbol`` would breach a portfolio limit.

    Checked against the worst case (a full ``MAX_NOTIONAL_USD`` order)
    before the quote is even fetched: sizing can only make the order
    smaller, so passing here with the maximum means passing with any size.
    """
    new_notional = tier0.MAX_NOTIONAL_USD

    if not tier0.ALLOW_PYRAMIDING and symbol in snapshot.symbols_exposed:
        held = snapshot.positions.get(symbol, Decimal(0))
        working = snapshot.working_buys.get(symbol, Decimal(0))
        raise PolicyRejection(
            "duplicate_accumulation",
            f"{symbol} already has exposure (position ${held:.2f}, working buys ${working:.2f}); "
            "no second entry until it is fully exited",
        )

    if len(snapshot.symbols_exposed | {symbol}) > tier0.MAX_OPEN_POSITIONS:
        raise PolicyRejection(
            "max_open_positions",
            f"{len(snapshot.symbols_exposed)} symbols already held or being bought "
            f"(limit {tier0.MAX_OPEN_POSITIONS})",
        )

    if snapshot.gross_exposure + new_notional > tier0.MAX_GROSS_EXPOSURE_USD:
        raise PolicyRejection(
            "gross_exposure",
            f"gross exposure ${snapshot.gross_exposure:.2f} + ${new_notional} would exceed "
            f"${tier0.MAX_GROSS_EXPOSURE_USD}",
        )

    if snapshot.buys_today >= tier0.MAX_NEW_BUYS_PER_DAY:
        raise PolicyRejection(
            "daily_buy_limit",
            f"{snapshot.buys_today} buy orders already today (limit {tier0.MAX_NEW_BUYS_PER_DAY})",
        )

    last_sell = snapshot.last_sell_at.get(symbol)
    if last_sell is not None:
        ready_at = last_sell + timedelta(seconds=tier0.REENTRY_COOLDOWN_SECONDS)
        if now < ready_at:
            raise PolicyRejection(
                "reentry_cooldown",
                f"{symbol} was sold at {last_sell.isoformat()}; re-entry allowed from {ready_at.isoformat()}",
            )
