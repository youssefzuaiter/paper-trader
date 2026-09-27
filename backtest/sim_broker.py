"""A simulated Alpaca paper account (design §3, §4).

``SimAlpaca`` is the *server*. The production client, ``AsyncAlpaca``,
talks to it through ``httpx.MockTransport``, so everything from the router
down to the client's ``Decimal`` parsing and ``LatestQuote`` runs as in
production; only Alpaca itself is simulated. It answers the calls the
router and its circuit breaker make, in Alpaca's JSON shapes (numbers as
strings, timestamps ISO with nanoseconds or offsets as Alpaca writes them;
``tests/backtest`` holds them against recorded paper responses):

    GET  /v2/clock   /v2/account   /v2/positions   /v2/orders   /v2/orders:by_client_order_id
    POST /v2/orders
    GET  /v2/stocks/{symbol}/quotes/latest   (data host)

The engine drives time: ``set_time``, ``advance`` (resolve every bar that
ended, fill what can fill, mark positions), ``mark("low" | "close")`` and
``end_session``. Fills follow ``costs``: limit buys only through the
limit, market orders at the next bar's open ± costs, whole fills, day
orders expire at the close. P4 is asserted on every fill: never on a bar
starting less than one bar after the decision (legacy mode excepted).
"""

from __future__ import annotations

import itertools
import json
import uuid
from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

import httpx

from backtest import costs
from backtest.calendar import Session
from backtest.costs import CostLevel, DailyFees, FeeTable
from backtest.data import BarSeries, MarketData
from config import PAPER_BASE_URL
from risk_router.alpaca_async import AsyncAlpaca
from swarm.common import NEW_YORK

INITIAL_CASH: Final[Decimal] = Decimal("100000")
MIN_LATENCY_BARS: Final[int] = 1
_PAPER_HOST: Final[str] = httpx.URL(PAPER_BASE_URL).host
_DATA_HOST: Final[str] = "data.alpaca.markets"
_WORKING: Final[frozenset[str]] = frozenset({"new", "accepted", "partially_filled"})


class FillTimingError(AssertionError):
    """P4: an order filled on a bar that started too soon after its decision."""


class SimOrderRejected(RuntimeError):
    pass


def _iso_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _num(value: Decimal) -> str:
    return format(value.normalize(), "f") if value != 0 else "0"


@dataclass
class SimOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    type: str
    time_in_force: str
    qty: Decimal
    limit_price: Decimal | None
    submitted_at: datetime
    decided_at: datetime
    eligible_from: float          # epoch: the first bar a fill may use starts here or later
    session: int                  # calendar index of the session the order lives in
    next_bar: int                 # the next bar index to examine
    status: str = "accepted"
    filled_qty: Decimal = Decimal(0)
    filled_avg_price: Decimal | None = None
    filled_at: datetime | None = None
    expired_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def working(self) -> bool:
        return self.status in _WORKING


@dataclass
class SimPosition:
    symbol: str
    qty: Decimal = Decimal(0)
    cost_basis: Decimal = Decimal(0)
    mark: Decimal = Decimal(0)
    lastday_price: Decimal = Decimal(0)

    @property
    def avg_entry_price(self) -> Decimal:
        return self.cost_basis / self.qty if self.qty else Decimal(0)


@dataclass(frozen=True)
class Fill:
    order: SimOrder
    at: datetime                  # the fill bar's end
    bar_start: datetime
    price: costs.FillPrice
    cash: Decimal                 # signed
    fees: dict[str, Decimal]


FillHook = Callable[[Fill], None]
ExpireHook = Callable[[SimOrder], None]


class SimAlpaca:
    def __init__(self, market: MarketData, level: CostLevel, fees: FeeTable, *, latency_bars: int = 1,
                 bars: Mapping[str, BarSeries] | None = None, legacy: bool = False,
                 initial_cash: Decimal = INITIAL_CASH, on_fill: FillHook | None = None,
                 on_expire: ExpireHook | None = None) -> None:
        self.market = market
        self.calendar = market.calendar
        self.level = level
        self.fees = fees
        self.latency_bars = latency_bars
        self.bars = dict(bars) if bars is not None else dict(market.minute)
        self.legacy = legacy
        self.cash = initial_cash
        self.last_equity = initial_cash
        self.positions: dict[str, SimPosition] = {}
        self.orders: list[SimOrder] = []          # in submission order: time only moves forward
        self._submitted: list[datetime] = []      # their submission times, for bisect
        self._working: list[SimOrder] = []
        self._by_client_id: dict[str, SimOrder] = {}
        self._ids = itertools.count(1)
        self.now = datetime.fromtimestamp(0, UTC)
        self.daily_fees = DailyFees()
        self.fees_paid: dict[str, Decimal] = {}
        self._on_fill = on_fill
        self._on_expire = on_expire
        self._last_bar: dict[str, int] = {}
        self.requests = 0

    # --- the production client, pointed here ---------------------------------------------

    def client(self) -> AsyncAlpaca:
        async def no_sleep(_: float) -> None:
            return None

        return AsyncAlpaca("sim-key", "sim-secret", transport=httpx.MockTransport(self.handle), sleep=no_sleep)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        host, path, method = request.url.host, request.url.path, request.method
        params = request.url.params
        if host == _DATA_HOST and path.startswith("/v2/stocks/") and path.endswith("/quotes/latest"):
            return _json(200, self.quote_body(path.split("/")[3]))
        if host != _PAPER_HOST:
            return _json(404, {"message": f"unknown host {host}"})
        if path == "/v2/clock":
            return _json(200, self.clock_body())
        if path == "/v2/account":
            return _json(200, self.account_body())
        if path == "/v2/positions":
            return _json(200, self.positions_body())
        if path == "/v2/orders" and method == "GET":
            after = datetime.fromisoformat(params["after"]) if "after" in params else None
            return _json(200, self.orders_body(params["status"], after, int(params.get("limit", 50)),
                                               params.get("direction", "desc")))
        if path == "/v2/orders" and method == "POST":
            status, body = self.submit(json.loads(request.content))
            return _json(status, body)
        if path == "/v2/orders:by_client_order_id":
            order = self._by_client_id.get(params["client_order_id"])
            return _json(200, self._order_body(order)) if order else _json(404, {"message": "order not found"})
        return _json(404, {"message": f"unhandled {method} {path}"})

    # --- the Alpaca bodies ------------------------------------------------------------------

    def clock_body(self) -> dict[str, Any]:
        return self.calendar.clock(self.now)

    def equity(self) -> Decimal:
        return self.cash + sum((p.qty * p.mark for p in self.positions.values()), Decimal(0))

    def account_body(self) -> dict[str, Any]:
        equity, long_value = self.equity(), sum((p.qty * p.mark for p in self.positions.values()), Decimal(0))
        cash = _num(self.cash.quantize(Decimal("0.01")))
        return {
            "id": "sim-account", "admin_configurations": {}, "user_configurations": None,
            "account_number": "SIM0000000", "status": "ACTIVE", "crypto_status": "ACTIVE",
            "options_approved_level": 0, "options_trading_level": 0, "currency": "USD", "buying_power": cash,
            "regt_buying_power": cash, "effective_buying_power": cash, "non_marginable_buying_power": cash,
            "options_buying_power": cash, "cash": cash, "accrued_fees": "0", "portfolio_value": _num(equity),
            "trading_blocked": False, "transfers_blocked": False, "account_blocked": False,
            "created_at": "2024-01-02T14:30:00.000000Z", "trade_suspended_by_user": False, "multiplier": "1",
            "shorting_enabled": False, "equity": _num(equity), "last_equity": _num(self.last_equity),
            "long_market_value": _num(long_value), "short_market_value": "0", "position_market_value": _num(long_value),
            "initial_margin": "0", "maintenance_margin": "0", "last_maintenance_margin": "0", "sma": "0",
            "balance_asof": self.now.astimezone(NEW_YORK).date().isoformat(), "crypto_tier": 0,
            "intraday_adjustments": "0", "pending_reg_taf_fees": "0",
        }

    def positions_body(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.positions.values(), key=lambda p: p.symbol):
            held = sum((o.qty for o in self._working if o.side == "sell" and o.symbol == p.symbol), Decimal(0))
            value, basis = p.qty * p.mark, p.cost_basis
            intraday_base = p.qty * p.lastday_price if p.lastday_price else basis
            out.append({
                "asset_id": f"asset-{p.symbol}", "symbol": p.symbol, "exchange": "NASDAQ", "asset_class": "us_equity",
                "asset_marginable": True, "qty": _num(p.qty), "qty_available": _num(p.qty - held),
                "avg_entry_price": _num(p.avg_entry_price), "side": "long", "market_value": _num(value),
                "cost_basis": _num(basis), "unrealized_pl": _num(value - basis),
                "unrealized_plpc": _num((value - basis) / basis) if basis else "0",
                "unrealized_intraday_pl": _num(value - intraday_base),
                "unrealized_intraday_plpc": _num((value - intraday_base) / intraday_base) if intraday_base else "0",
                "current_price": _num(p.mark), "lastday_price": _num(p.lastday_price),
                "change_today": _num((p.mark - p.lastday_price) / p.lastday_price) if p.lastday_price else "0",
            })
        return out

    def orders_body(self, status: str, after: datetime | None, limit: int, direction: str) -> list[dict[str, Any]]:
        if status == "open":
            candidates = self._working
        else:  # only orders submitted after ``after``: a bisect, not a scan of the whole history
            candidates = self.orders[bisect_right(self._submitted, after):] if after is not None else self.orders
        chosen = [o for o in candidates
                  if (status == "all" or (status == "open") == o.working)
                  and (after is None or o.submitted_at > after)]
        chosen.sort(key=lambda o: (o.submitted_at, o.id), reverse=direction == "desc")
        return [self._order_body(o) for o in chosen[:limit]]

    def _order_body(self, o: SimOrder) -> dict[str, Any]:
        session = self.calendar.sessions[o.session]
        return {
            "id": o.id, "client_order_id": o.client_order_id, "created_at": _iso_utc(o.submitted_at),
            "updated_at": _iso_utc(o.updated_at or o.submitted_at), "submitted_at": _iso_utc(o.submitted_at),
            "filled_at": _iso_utc(o.filled_at) if o.filled_at else None,
            "expired_at": _iso_utc(o.expired_at) if o.expired_at else None, "canceled_at": None, "failed_at": None,
            "replaced_at": None, "replaced_by": None, "replaces": None, "asset_id": f"asset-{o.symbol}",
            "symbol": o.symbol, "asset_class": "us_equity", "notional": None, "qty": _num(o.qty),
            "filled_qty": _num(o.filled_qty),
            "filled_avg_price": _num(o.filled_avg_price) if o.filled_avg_price is not None else None,
            "order_class": "", "order_type": o.type, "type": o.type, "side": o.side,
            "position_intent": "buy_to_open" if o.side == "buy" else "sell_to_close",
            "time_in_force": o.time_in_force,
            "limit_price": _num(o.limit_price) if o.limit_price is not None else None, "stop_price": None,
            "status": o.status, "extended_hours": False, "legs": None, "trail_percent": None, "trail_price": None,
            "hwm": None, "subtag": None, "source": "access_key", "expires_at": _iso_utc(session.close_at),
        }

    def quote_body(self, symbol: str) -> dict[str, Any]:
        bar = self.market.as_of(self.now).quote_bar(symbol) if symbol in self.bars else None
        if bar is None:
            return {"symbol": symbol, "quote": {"t": _iso_utc(self.now - timedelta(days=1)), "ax": " ", "ap": 0,
                                                "as": 0, "bx": " ", "bp": 0, "bs": 0, "c": ["R"], "z": "C"}}
        session = self.calendar.session_at(bar.start)
        m = costs.multiplier(self.level, bar.start, session.open_at) if session else Decimal(1)
        q = costs.quote(costs.to_decimal(bar.c), symbol, self.level, m)
        return {"symbol": symbol, "quote": {"t": _iso_utc(bar.end), "ax": "V", "ap": float(q.ask), "as": 1,
                                            "bx": "V", "bp": float(q.bid), "bs": 1, "c": ["R"], "z": "C"}}

    # --- orders ----------------------------------------------------------------------------------

    def submit(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        client_id = payload["client_order_id"]
        if client_id in self._by_client_id:
            return 422, {"code": 40010001, "message": "client_order_id must be unique"}
        if payload.get("order_class") == "oto" or "stop_loss" in payload:
            # tier0 attaches a native stop only to whole-share orders; at $10 a share that never
            # happens on these symbols. A backtest that meets one has left its assumptions: stop.
            raise SimOrderRejected(f"order_class oto is not simulated: {dict(payload)}")
        symbol, side, kind = payload["symbol"], payload["side"], payload["type"]
        qty = Decimal(payload["qty"])
        if symbol not in self.bars or side not in {"buy", "sell"} or kind not in {"limit", "market"} or qty <= 0:
            return 422, {"code": 40010001, "message": f"invalid order {dict(payload)}"}
        tif = payload.get("time_in_force", "day")
        if tif not in {"day", "cls"}:
            return 422, {"code": 40010001, "message": f"time_in_force {tif} is not simulated"}
        if side == "sell" and not self.legacy:
            # Legacy mode (the L4b label check) places each sample's exit with its entry.
            position = self.positions.get(symbol)
            held = sum((o.qty for o in self._working if o.side == "sell" and o.symbol == symbol), Decimal(0))
            if position is None or qty > position.qty - held:
                return 403, {"code": 40310000, "message": "insufficient qty available for order"}
        limit = Decimal(payload["limit_price"]) if kind == "limit" else None
        eligible = self.now if self.legacy else _ceil_minute(self.now) + timedelta(minutes=self.latency_bars)
        series = self.bars[symbol]
        first = self._first_bar(series, eligible.timestamp())
        if self.legacy and first < len(series):
            session_index = int(series.session[first])  # legacy labels roll to the next session's bars
        else:
            session = self.calendar.session_at(self.now) or self.calendar.next_session(self.now)
            session_index = self.calendar.index[session.date]
        order = SimOrder(
            id=str(uuid.UUID(int=next(self._ids))), client_order_id=client_id, symbol=symbol, side=side,
            type=kind, time_in_force=tif, qty=qty, limit_price=limit, submitted_at=self.now, decided_at=self.now,
            eligible_from=eligible.timestamp(), session=session_index, next_bar=first)
        self.orders.append(order)
        self._submitted.append(order.submitted_at)
        self._working.append(order)
        self._by_client_id[client_id] = order
        return 200, self._order_body(order)

    def _first_bar(self, series: BarSeries, eligible_epoch: float) -> int:
        """The first bar an order may fill on: the first starting at or after its eligible time."""
        return series.first_at_or_after(eligible_epoch)

    # --- time ---------------------------------------------------------------------------------------

    def set_time(self, t: datetime) -> None:
        if t < self.now:
            raise ValueError(f"time runs forward: {t} < {self.now}")
        self.now = t

    def advance(self, t: datetime) -> list[Fill]:
        """Resolve every bar that ended by ``t`` for each working order, in
        submission order, then mark positions at the latest bar's close."""
        self.set_time(t)
        epoch = t.timestamp()
        fills = []
        for order in list(self._working):
            series = self.bars[order.symbol]
            i = order.next_bar
            while i < len(series) and series.start[i] + series.seconds <= epoch and series.session[i] == order.session:
                fill = self._try_fill(order, series, i)
                i += 1
                if fill is not None:
                    fills.append(fill)
                    break
            order.next_bar = i
        for symbol in self.positions:
            series = self.bars[symbol]
            j = series.last_ended_by(epoch)
            if j >= 0:
                self._last_bar[symbol] = j
        self.mark("close")
        return fills

    def mark(self, mode: str) -> None:
        """Positions' ``current_price``: the last resolved bar's low (stops fire
        on any touch) or its close (take-profits need a close through)."""
        for symbol, position in self.positions.items():
            j = self._last_bar.get(symbol)
            if j is None:
                continue
            series = self.bars[symbol]
            value = series.l[j] if mode == "low" else series.c[j]
            position.mark = costs.to_decimal(float(value))

    def _try_fill(self, order: SimOrder, series: BarSeries, i: int) -> Fill | None:
        start = float(series.start[i])
        if not self.legacy and start < order.decided_at.timestamp() + MIN_LATENCY_BARS * 60:
            raise FillTimingError(f"{order.symbol} {order.side} decided {order.decided_at.isoformat()} "
                                  f"would fill on the bar starting {datetime.fromtimestamp(start, UTC).isoformat()}")
        bar_start = datetime.fromtimestamp(start, UTC)
        session = self.calendar.sessions[int(series.session[i])]
        m = costs.multiplier(self.level, bar_start, session.open_at)
        open_ = costs.to_decimal(float(series.o[i]))
        if order.time_in_force == "cls":
            if i != series.session_last(i):
                return None
            close = costs.to_decimal(float(series.c[i]))
            price = costs.FillPrice(close, close, Decimal(0), Decimal(0))
        elif order.type == "market":
            price = costs.market_fill(order.side, open_, order.symbol, self.level, m)
        else:
            if order.side != "buy":
                raise SimOrderRejected("limit sells are not simulated")
            price = costs.limit_buy_fill(order.limit_price, open_, costs.to_decimal(float(series.l[i])),
                                         order.symbol, self.level, m)
            if price is None:
                return None
        return self._apply(order, price, bar_start, bar_start + timedelta(seconds=series.seconds), session.date)

    def _apply(self, order: SimOrder, price: costs.FillPrice, bar_start: datetime, at: datetime,
               day: date) -> Fill:
        qty = order.qty
        fees = self.fees.order_fees(day, order.side, qty, price.price, self.level.fee_rounding)
        if self.level.fee_rounding == "daily":
            self.daily_fees.add(day, fees)
        if self.level.fee_rounding == "none":
            fees = {}
        fee_total = sum(fees.values(), Decimal(0))
        for k, v in fees.items():
            self.fees_paid[k] = self.fees_paid.get(k, Decimal(0)) + v
        position = self.positions.setdefault(order.symbol, SimPosition(order.symbol))
        if order.side == "buy":
            cash = -costs.cash_debit(qty, price.price, self.level)
            if not position.lastday_price:
                position.lastday_price = price.ref
            position.qty += qty
            position.cost_basis += qty * price.price
        else:
            cash = costs.cash_credit(qty, price.price, self.level)
            position.cost_basis -= position.avg_entry_price * qty
            position.qty -= qty
        self.cash += cash - fee_total
        position.mark = price.price
        if position.qty == 0:
            del self.positions[order.symbol]
            self._last_bar.pop(order.symbol, None)
        order.status, order.filled_qty, order.filled_avg_price = "filled", qty, price.price
        order.filled_at = order.updated_at = at
        self._working.remove(order)
        fill = Fill(order, at, bar_start, price, cash, fees)
        if self._on_fill is not None:
            self._on_fill(fill)
        return fill

    def end_session(self, session: Session) -> dict[str, Any]:
        """At the close: expire day orders, charge the day's fee rounding,
        roll ``last_equity``. Returns what happened, for the recorder."""
        self.advance(max(self.now, session.close_at))
        expired = []
        for order in list(self._working):
            if order.session == self.calendar.index[session.date]:
                order.status, order.expired_at = "expired", session.close_at
                order.updated_at = session.close_at
                self._working.remove(order)
                expired.append(order)
                if self._on_expire is not None:
                    self._on_expire(order)
        rounding = Decimal(0)
        if self.level.fee_rounding == "daily":
            rounding = self.daily_fees.rounding_charge(session.date)
            self.cash -= rounding
            self.fees_paid["daily_rounding"] = self.fees_paid.get("daily_rounding", Decimal(0)) + rounding
        for position in self.positions.values():
            position.lastday_price = position.mark
        self.last_equity = self.equity()
        return {"expired": expired, "fee_rounding": rounding, "carried": sorted(self.positions)}


def _ceil_minute(ts: datetime) -> datetime:
    floor = ts.replace(second=0, microsecond=0)
    return floor if floor == ts else floor + timedelta(minutes=1)


def _json(status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(body).encode(), headers={"content-type": "application/json"})
