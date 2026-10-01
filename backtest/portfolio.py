"""The long-term core's one portfolio engine (core design §3).

For each session *k* of a window:

1. **Open (fill side).** A pending plan fills at this open: sells first, then
   buys capped by the cash the sells left (``CORE``). Under ``CORE`` the
   router-faithful breaker applies first: if the portfolio's value at the open
   is down ``tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT`` or more on the previous
   close and the plan buys anything, the whole plan is deferred and re-decided
   at this close, as the live router would block it (§3.1 a, §8.3).
2. **Close.** Mark at adjusted closes; record value, cash, weights, costs.
3. **Decision**, as of the close + 15 minutes, from a ``DailyView`` only:
   tax due at a year's end, registered cash needs, then ``rule.due``. A due
   rule recomputes the targets (M4's volatilities) and plans fully back to
   them; a plan fills at the next open.

Two profiles (§3.4). ``PHASE1`` exists only so phase 1's ``run_hold`` runs
through this engine and reproduces its golden file exactly (check C7):
every asset sized ``value(prev close) × w / open`` (cash may go negative),
fees per order on adjusted units, the band measured without cash, no breaker.
``CORE`` is every registered core run: sells by quantity and buys by notional
from the decision close, buys scaled to the cash available, per-share fees on
raw shares, the level's fee rounding, the breaker on.

Money and quantities are ``Decimal`` throughout; there is no second engine
(C7's AST check: ``run_hold`` only calls ``simulate``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_DOWN, Decimal
from typing import Final

import core_alloc
import tier0
from backtest import costs
from backtest.costs import CostLevel, DailyFees, FeeTable
from backtest.daily import DailyView, Prices

ZERO: Final[Decimal] = Decimal(0)
QTY: Final[Decimal] = core_alloc.QTY_QUANTUM
#: A sale's fees may each round up by a cent (SEC, TAF, CAT): a raise-cash plan sells this much extra per order.
SELL_ROUNDING_USD: Final[Decimal] = Decimal("0.03")


@dataclass(frozen=True)
class Profile:
    name: str
    sizing: str                  # "phase1" | "core"
    fee_rounding: str | None     # None: the cost level's own (D12)
    open_multiplier: bool        # phase 1: h·m at the open; core: h is measured at the open, m = 1
    breaker: bool
    drift_includes_cash: bool
    raw_fees: bool               # per-share fees on raw shares (adjusted units × factor)


PHASE1: Final[Profile] = Profile("PHASE1", "phase1", "per_order", True, False, False, False)
CORE: Final[Profile] = Profile("CORE", "core", None, False, True, True, True)


@dataclass(frozen=True)
class Spec:
    """One configuration: what to hold, how to weight it, when to rebalance."""
    name: str
    symbols: tuple[str, ...]
    weights: Callable[[DailyView | None], dict[str, Decimal]]
    rule: core_alloc.Rule
    capital: Decimal = Decimal(100000)
    min_order_usd: Decimal = Decimal(1)
    cash_reserve_usd: Decimal = ZERO
    tax_rate: Decimal = ZERO
    lot_method: str = "average"
    #: (decision date, amount): cash the owner needs, raised by the next open (design §6).
    cash_needs: tuple[tuple[date, Decimal], ...] = ()
    needs_view: bool = False     # M4: weights read trailing volatility through the view
    buffer_symbol: str | None = None  # the cash-buffer sleeve a raise-cash plan drains first


@dataclass
class Day:
    date: date
    value: Decimal               # mark-to-market at the adjusted close, after every cost
    cash: Decimal
    weights: dict[str, Decimal]
    traded: Decimal = ZERO       # notional bought and sold at the open's reference price
    exec_cost: Decimal = ZERO    # spread + slippage + tick rounding, at the fills
    spread: Decimal = ZERO
    slippage: Decimal = ZERO
    fees: Decimal = ZERO
    tax: Decimal = ZERO          # paid this day (illustrative)
    flow: Decimal = ZERO         # cash the owner withdrew this day (not a return)
    realised: Decimal = ZERO     # realised gain on this day's sales
    rebalanced: bool = False
    deferred: bool = False       # a plan with buys was blocked by the breaker
    decided: str | None = None   # what this close decided: "rebalance", "raise_cash", ...
    orders: list[dict[str, str]] = field(default_factory=list)

    @property
    def costs(self) -> Decimal:
        return self.exec_cost + self.fees


@dataclass
class _Pending:
    plan: core_alloc.Plan
    value: Decimal               # portfolio value at the decision close (phase 1 sizes from it)
    need: Decimal = ZERO         # cash to set aside at the fill: tax and withdrawals
    tax: Decimal = ZERO
    initial: bool = False


def simulate(prices: Prices, spec: Spec, start: int, end: int, level: CostLevel, fees: FeeTable,
             profile: Profile = CORE) -> list[Day]:
    """Sessions ``start`` … ``end`` (indices, inclusive). The initial build is decided at the close
    of ``start - 1`` (sized on ``capital``) and fills at the open of ``start``."""
    if profile.sizing == "core" and fees.covered_from() > prices.sessions[start].date:
        raise ValueError(f"fees.json covers equities only from {fees.covered_from()}")
    cash = spec.capital
    qty: dict[str, Decimal] = {s: ZERO for s in spec.symbols}
    holdings = {s: core_alloc.Holding() for s in spec.symbols}
    tax = core_alloc.TaxYear(spec.tax_rate)
    view0 = prices.view(start - 1) if spec.needs_view else None
    targets = spec.weights(view0)
    pending: _Pending | None = _Pending(
        core_alloc.Plan(prices.sessions[start - 1].date if start else prices.sessions[0].date, targets,
                        buys={s: (w * spec.capital) for s, w in targets.items()}, reason="initial"),
        spec.capital, initial=True)
    forced = False
    carried = carried_tax = ZERO   # cash a deferred or short fill still owes, and the tax part of it
    prev_value = spec.capital
    needs = dict(spec.cash_needs)
    daily_fees = DailyFees()
    rounding = profile.fee_rounding or level.fee_rounding
    out: list[Day] = []
    for k in range(start, end + 1):
        day_date = prices.sessions[k].date
        day = Day(day_date, ZERO, ZERO, {})
        # --- 1. open ---------------------------------------------------------------------------------
        if pending is not None:
            value_at_open = cash + sum((q * prices.open(s, k) for s, q in qty.items() if q), ZERO)
            blocked = (profile.breaker and not pending.initial and pending.plan.buys
                       and (value_at_open / prev_value - 1) * 100 <= tier0.CIRCUIT_BREAKER_DAILY_PNL_PCT)
            if blocked:
                day.deferred = True
                forced = True
                carried, carried_tax = pending.need, pending.tax
            elif profile.sizing == "phase1":
                cash = _fill_phase1(prices, spec, k, level, fees, pending, qty, cash, day)
            else:
                cash, carried = _fill_core(prices, spec, k, level, fees, rounding, pending, qty, holdings, tax, cash,
                                           day, daily_fees)
                carried_tax = min(carried, pending.tax)
            pending = None
        if rounding == "daily" and profile.sizing == "core":
            charge = daily_fees.rounding_charge(day_date)
            cash -= charge
            day.fees += charge
        # --- 2. close --------------------------------------------------------------------------------
        closes = {s: prices.close(s, k) for s, q in qty.items() if q}
        value = cash + sum((q * closes[s] for s, q in qty.items() if q), ZERO)
        day.value, day.cash = value, cash
        day.weights = core_alloc.current_weights(qty, closes, cash) if value > 0 else {}
        out.append(day)
        prev_value = value
        # --- 3. decision, as of close + 15 minutes ----------------------------------------------------------
        if k == end:
            break
        ends = core_alloc.period_ends(day_date, prices.sessions[k + 1].date)
        tax_due = tax.due(day_date.year) if "year" in ends and spec.tax_rate > 0 else ZERO
        need = carried + tax_due + needs.pop(day_date, ZERO)
        carried_tax += tax_due
        carried = ZERO
        drift_basis = day.weights if profile.drift_includes_cash else _weights_ex_cash(qty, closes)
        rule = spec.rule
        if forced:  # a plan the breaker deferred is re-decided at this close (§3.1 a)
            due = rule.never or rule.band is None or core_alloc.drift(drift_basis, targets) > rule.band
        else:
            due = rule.due(ends, drift_basis, targets)
        forced = False
        if due:
            view = prices.view(k) if spec.needs_view else None
            targets = spec.weights(view)
            prices_k = {s: prices.close(s, k) for s in set(qty) | set(targets)}
            plan = core_alloc.plan_rebalance(day_date, qty, prices_k, cash - need - spec.cash_reserve_usd, targets,
                                             min_order_usd=spec.min_order_usd)
            if profile.sizing == "phase1" or not plan.empty or need > 0:
                pending = _Pending(plan, value, need=need, tax=carried_tax)
                day.decided = "rebalance"
        elif need > 0:
            prices_k = {s: prices.close(s, k) for s in qty if qty[s]}
            # Each sale may round up to a cent per fee type (pessimistic level): allow for it in the need.
            allowance = SELL_ROUNDING_USD * len(prices_k) if level.fee_rounding != "none" else ZERO
            plan = core_alloc.plan_raise_cash(day_date, {s: q for s, q in qty.items() if q}, prices_k,
                                              cash - spec.cash_reserve_usd, need + allowance, targets,
                                              cost_rate={s: _cost_rate(level, s, prices, "sell") for s in prices_k},
                                              buffer_symbol=spec.buffer_symbol, min_order_usd=spec.min_order_usd)
            pending = _Pending(plan, value, need=need, tax=carried_tax)
            day.decided = "raise_cash"
        if pending is not None:
            carried_tax = ZERO
    return out


def qty_held(qty: dict[str, Decimal]) -> bool:
    return any(q for q in qty.values())


def _weights_ex_cash(qty: dict[str, Decimal], closes: dict[str, Decimal]) -> dict[str, Decimal]:
    """Phase 1's band basis: each holding's share of the holdings' total, cash left out."""
    values = {s: q * closes[s] for s, q in qty.items() if q}
    total = sum(values.values(), ZERO)
    return {s: v / total for s, v in values.items()} if total > 0 else {}


def _cost_rate(level: CostLevel, symbol: str, prices: Prices, side: str = "buy") -> Decimal:
    """Expected cost of one order as a fraction of notional: spread, slippage and a fee allowance,
    so buys scaled to the cash available leave it non-negative (design §3.1 c). Equity buys pay
    only CAT (a fraction of a basis point); crypto pays its 0.25% on both sides."""
    if level.fee_rounding == "none":
        fee = ZERO
    elif prices.asset_class(symbol) == "crypto":
        fee = Decimal("0.0025")
    else:
        fee = Decimal("0.00001") if side == "buy" else Decimal("0.0001")  # sells add SEC and TAF
    return level.half_spread(symbol) + level.slippage_bps * costs.BPS + fee


def _fill_phase1(prices: Prices, spec: Spec, k: int, level: CostLevel, fees: FeeTable, pending: _Pending,
                 qty: dict[str, Decimal], cash: Decimal, day: Day) -> Decimal:
    """``run_hold``'s arithmetic, unchanged: every symbol to ``value × w / open``, in ``spec.symbols`` order."""
    day_date = prices.sessions[k].date
    for s in spec.symbols:
        open_ = prices.open(s, k)
        want = (pending.value * pending.plan.targets[s] / open_).quantize(QTY, rounding=ROUND_DOWN)
        delta = want - qty[s]
        if delta == 0:
            continue
        side = "buy" if delta > 0 else "sell"
        fill = costs.market_fill(side, open_, s, level, level.open_multiplier)
        amount = abs(delta)
        charged = sum(fees.order_fees(day_date, side, amount, fill.price, "per_order").values(), ZERO)
        if side == "buy":
            cash -= costs.cash_debit(amount, fill.price, level) + charged
        else:
            cash += costs.cash_credit(amount, fill.price, level) - charged
        qty[s] = want
        day.traded += amount * open_
        day.exec_cost += amount * abs(fill.price - open_)
        day.fees += charged
    day.rebalanced = True
    return cash


def _fill_core(prices: Prices, spec: Spec, k: int, level: CostLevel, fees: FeeTable, rounding: str,
               pending: _Pending, qty: dict[str, Decimal], holdings: dict[str, core_alloc.Holding],
               tax: core_alloc.TaxYear, cash: Decimal, day: Day, daily_fees: DailyFees) -> tuple[Decimal, Decimal]:
    """Returns the cash after the fills and any need the sells could not cover (carried to the next close)."""
    day_date = prices.sessions[k].date
    h_sigma = {s: (level.half_spread(s), level.slippage_bps * costs.BPS) for s in set(qty) | set(pending.plan.buys)}

    def charge(symbol: str, side: str, units: Decimal, price: Decimal) -> Decimal:
        if rounding == "none":
            return ZERO  # the frictionless consistency level (C9)
        factor = prices.factor(symbol, k)
        raw_units, raw_price = units * factor, price / factor
        exact = fees.order_fees(day_date, side, raw_units, raw_price, rounding, prices.asset_class(symbol))
        if rounding == "daily":
            daily_fees.add(day_date, {f"{prices.asset_class(symbol)}:{name}": v for name, v in exact.items()})
        return sum(exact.values(), ZERO)

    for s in sorted(pending.plan.sells):
        units = min(pending.plan.sells[s], qty[s])
        if units <= 0:
            continue
        open_ = prices.open(s, k)
        fill = costs.market_fill("sell", open_, s, level, Decimal(1))
        proceeds = costs.cash_credit(units, fill.price, level)
        fee = charge(s, "sell", units, fill.price)
        basis = holdings[s].sell(units, spec.lot_method)
        gain = proceeds - fee - basis
        tax.realise(day_date.year, gain)
        cash += proceeds - fee
        qty[s] -= units
        _book(day, s, "sell", units, open_, fill, fee, h_sigma[s])
        day.realised += gain
    short = ZERO
    if pending.need:
        # Sells were sized at the decision close; an overnight gap can leave them short. Pay what the
        # cash covers (tax first), carry the rest: cash never goes negative (C8).
        paid = min(pending.need, max(cash - spec.cash_reserve_usd, ZERO)).quantize(core_alloc.CENT, ROUND_DOWN)
        short = pending.need - paid
        tax_paid = min(pending.tax, paid)
        cash -= paid
        day.tax += tax_paid
        day.flow += paid - tax_paid
    available = cash - spec.cash_reserve_usd
    buys = core_alloc.scale_buys(pending.plan.buys, available,
                                 {s: _cost_rate(level, s, prices) for s in pending.plan.buys})
    for s in sorted(buys):
        notional = buys[s]
        if notional < spec.min_order_usd:
            continue
        open_ = prices.open(s, k)
        fill = costs.market_fill("buy", open_, s, level, Decimal(1))
        units = (notional / fill.price).quantize(QTY, rounding=ROUND_DOWN)
        if units <= 0:
            continue
        debit = costs.cash_debit(units, fill.price, level)
        fee = charge(s, "buy", units, fill.price)
        cash -= debit + fee
        qty[s] = qty.get(s, ZERO) + units
        holdings.setdefault(s, core_alloc.Holding()).buy(units, debit + fee, day_date)
        _book(day, s, "buy", units, open_, fill, fee, h_sigma[s])
    day.rebalanced = True
    return cash, short


def _book(day: Day, symbol: str, side: str, units: Decimal, open_: Decimal, fill: costs.FillPrice, fee: Decimal,
          h_sigma: tuple[Decimal, Decimal]) -> None:
    notional = units * open_
    day.traded += notional
    day.exec_cost += units * abs(fill.price - open_)
    day.spread += notional * h_sigma[0]
    day.slippage += notional * h_sigma[1]
    day.fees += fee
    day.orders.append({"symbol": symbol, "side": side, "units": str(units), "ref": str(open_),
                       "price": str(fill.price), "fees": str(fee)})


def hold_spec(symbols: Sequence[str], *, capital: Decimal, band: Decimal | None) -> Spec:
    """Phase 1's S0 (``band`` None: equal weight, held) and S1 (each month's last close: back to
    equal weight if any weight, cash left out, is outside 1/n ± ``band``)."""
    weight = Decimal(1) / len(symbols)
    rule = core_alloc.RULES["none"] if band is None else core_alloc.Rule(f"monthly_band{band}", "month", band)
    return Spec("S0" if band is None else "S1", tuple(symbols), lambda _view: dict.fromkeys(symbols, weight), rule,
                capital=capital)
