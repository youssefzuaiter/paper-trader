"""The long-term core's allocation rules: the one copy the backtest and paper trading share.

Core design §3 and §6. Like ``tier0.py``: pure standard library, no I/O, so
the router image can import it without numpy, and the engine and the paper
Allocator compute the *same bits* from the same inputs (check C12 asserts
both import these functions rather than copies).

Weights are ``Decimal``, quantised to 10⁻⁶ and summing to exactly 1 (the
rounding residual goes to the largest weight). Money and quantities are
``Decimal``; only volatility estimates are floats, and they are quantised
into weights before anything is decided from them.

Nothing here reads a price after the decision instant: every function takes
the history it may use as an argument, and the engine's point-in-time view
is what supplies it (check C2).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, Decimal
from typing import Final

WEIGHT_QUANTUM: Final[Decimal] = Decimal("0.000001")
QTY_QUANTUM: Final[Decimal] = Decimal("0.000000001")  # Alpaca's fractional precision
CENT: Final[Decimal] = Decimal("0.01")
ONE: Final[Decimal] = Decimal(1)
DONE: Final[Decimal] = Decimal("0.005")  # less than half a cent left to raise: done
ROUND_UP_QTY: Final[str] = ROUND_UP  # a raise-cash sale rounds up: never short of the need


# --- weights ------------------------------------------------------------------------------------------

def quantise_weights(raw: Mapping[str, Decimal | float]) -> dict[str, Decimal]:
    """Normalise to sum 1, quantise each to 10⁻⁶, give the residual to the largest
    (ties: the first symbol in sorted order), so the result sums to exactly 1."""
    if not raw or any(float(v) < 0 for v in raw.values()):
        raise ValueError(f"weights must be non-negative and non-empty: {dict(raw)}")
    total = sum((Decimal(repr(v)) if isinstance(v, float) else v for v in raw.values()), Decimal(0))
    if total <= 0:
        raise ValueError("weights sum to zero")
    out = {s: ((Decimal(repr(v)) if isinstance(v, float) else v) / total).quantize(WEIGHT_QUANTUM, ROUND_HALF_EVEN)
           for s, v in sorted(raw.items())}
    largest = max(sorted(out), key=lambda s: out[s])
    out[largest] += ONE - sum(out.values(), Decimal(0))
    return out


def log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(b / a) for a, b in itertools.pairwise(closes)]


def sample_std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two returns")
    mean = math.fsum(xs) / n
    return math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (n - 1))


def inverse_vol_weights(closes: Mapping[str, Sequence[float]], lookback: int,
                        fixed: Mapping[str, Decimal] | None = None) -> dict[str, Decimal]:
    """M4 (design §3.3): wᵢ ∝ 1/σᵢ, σᵢ the sample std of the last ``lookback`` log returns,
    so each series needs exactly ``lookback + 1`` closes ending at the decision close.
    ``fixed`` sleeves (BTC's 5% in M8, a BIL buffer) take their weight first; the
    inverse-volatility assets share the rest."""
    fixed = dict(fixed or {})
    for symbol, series in closes.items():
        if len(series) != lookback + 1:
            raise ValueError(f"{symbol}: {len(series)} closes, need exactly {lookback + 1}")
    inv = {s: 1.0 / sample_std(log_returns(c)) for s, c in closes.items()}
    rest = ONE - sum(fixed.values(), Decimal(0))
    if rest <= 0:
        raise ValueError("fixed sleeves leave nothing to risk-balance")
    total = math.fsum(inv.values())
    raw: dict[str, Decimal] = {s: rest * Decimal(repr(v / total)) for s, v in inv.items()}
    return quantise_weights({**raw, **fixed})


def with_sleeves(base: Mapping[str, Decimal], sleeves: Mapping[str, Decimal]) -> dict[str, Decimal]:
    """Scale ``base`` down to make room for ``sleeves`` (M5-M8's 5% BTC, a BIL buffer)."""
    rest = ONE - sum(sleeves.values(), Decimal(0))
    return quantise_weights({**{s: w * rest for s, w in base.items()}, **sleeves})


# --- rules --------------------------------------------------------------------------------------------

PERIODS: Final[tuple[str, ...]] = ("month", "quarter", "year")


def period_ends(session: date, next_session: date) -> frozenset[str]:
    """Which periods ``session`` is the last session of. The exchange calendar is published in
    advance, so knowing the next session's date at the close is not look-ahead."""
    ends = set()
    if (session.year, session.month) != (next_session.year, next_session.month):
        ends.add("month")
        if (session.month - 1) // 3 != (next_session.month - 1) // 3 or session.year != next_session.year:
            ends.add("quarter")
    if session.year != next_session.year:
        ends.update({"quarter", "year"})
    return frozenset(ends)


@dataclass(frozen=True)
class Rule:
    """One of the design's eight rules (§3.2). ``period`` None = every close; ``band`` None = always
    rebalance when checked. ``none`` never checks after the initial build."""
    name: str
    period: str | None
    band: Decimal | None
    never: bool = False

    def checked(self, ends: frozenset[str]) -> bool:
        return not self.never and (self.period is None or self.period in ends)

    def due(self, ends: frozenset[str], weights: Mapping[str, Decimal], targets: Mapping[str, Decimal]) -> bool:
        if not self.checked(ends):
            return False
        return self.band is None or drift(weights, targets) > self.band


RULES: Final[dict[str, Rule]] = {r.name: r for r in (
    Rule("none", None, None, never=True),
    Rule("monthly", "month", None),
    Rule("quarterly", "quarter", None),
    Rule("yearly", "year", None),
    Rule("band5", None, Decimal("0.05")),
    Rule("band10", None, Decimal("0.10")),
    Rule("monthly_band5", "month", Decimal("0.05")),     # phase 1's S1
    Rule("quarterly_band5", "quarter", Decimal("0.05")),
)}


def drift(weights: Mapping[str, Decimal], targets: Mapping[str, Decimal]) -> Decimal:
    """max |wᵢ − targetᵢ| in absolute weight (D6), over every symbol either side holds."""
    return max((abs(weights.get(s, Decimal(0)) - targets.get(s, Decimal(0))) for s in set(weights) | set(targets)),
               default=Decimal(0))


def current_weights(qty: Mapping[str, Decimal], prices: Mapping[str, Decimal], cash: Decimal) -> dict[str, Decimal]:
    """Each holding's share of total value (cash included in the total, not in the result)."""
    values = {s: q * prices[s] for s, q in qty.items() if q}
    total = cash + sum(values.values(), Decimal(0))
    if total <= 0:
        raise ValueError("portfolio value is not positive")
    return {s: v / total for s, v in values.items()}


# --- the rebalance plan --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """Sells by quantity, buys by notional (design §3.1), both decided at one close."""
    decided_on: date
    targets: dict[str, Decimal]
    sells: dict[str, Decimal] = field(default_factory=dict)   # symbol -> quantity
    buys: dict[str, Decimal] = field(default_factory=dict)    # symbol -> USD notional
    reason: str = "rebalance"

    @property
    def empty(self) -> bool:
        return not self.sells and not self.buys


def plan_rebalance(decided_on: date, qty: Mapping[str, Decimal], prices: Mapping[str, Decimal], cash: Decimal,
                   targets: Mapping[str, Decimal], *, min_order_usd: Decimal, reason: str = "rebalance") -> Plan:
    """Fully back to ``targets`` at the decision close's ``prices``. An order below
    ``min_order_usd`` is skipped and its drift stays (Alpaca's fractional minimum)."""
    value = cash + sum((q * prices[s] for s, q in qty.items()), Decimal(0))
    if value <= 0:
        raise ValueError("portfolio value is not positive")
    sells: dict[str, Decimal] = {}
    buys: dict[str, Decimal] = {}
    for symbol in sorted(set(qty) | set(targets)):
        held = qty.get(symbol, Decimal(0))
        price = prices[symbol]
        delta = targets.get(symbol, Decimal(0)) * value - held * price
        if abs(delta) < min_order_usd:
            continue
        if delta < 0:
            sell = min(held, (-delta / price).quantize(QTY_QUANTUM, ROUND_DOWN))
            if targets.get(symbol, Decimal(0)) == 0:
                sell = held  # leaving the mix entirely: no dust
            if sell > 0:
                sells[symbol] = sell
        else:
            buys[symbol] = delta.quantize(CENT, ROUND_DOWN)
    return Plan(decided_on, dict(targets), sells, buys, reason)


def scale_buys(buys: Mapping[str, Decimal], available: Decimal, cost_rate: Mapping[str, Decimal]) -> dict[str, Decimal]:
    """Each buy's notional scaled pro-rata so notional·(1 + cost) fits in ``available`` cash
    (design §3.1 c: buys capped by cash, never an overdraft)."""
    need = sum((n * (ONE + cost_rate.get(s, Decimal(0))) for s, n in buys.items()), Decimal(0))
    if need <= available or need == 0:
        return dict(buys)
    k = max(available, Decimal(0)) / need
    return {s: (n * k).quantize(CENT, ROUND_DOWN) for s, n in buys.items()}


# --- tax lots (design §3.5) ------------------------------------------------------------------------------

@dataclass
class Lot:
    qty: Decimal
    cost: Decimal        # dollars paid, fees included
    opened: date


@dataclass
class Holding:
    """Lots of one asset. ``average`` cost by default (O7); ``hifo`` sells the highest
    cost per unit first, ``fifo`` the oldest."""
    lots: list[Lot] = field(default_factory=list)

    @property
    def qty(self) -> Decimal:
        return sum((lot.qty for lot in self.lots), Decimal(0))

    @property
    def cost(self) -> Decimal:
        return sum((lot.cost for lot in self.lots), Decimal(0))

    def buy(self, qty: Decimal, cost: Decimal, day: date) -> None:
        if qty > 0:
            self.lots.append(Lot(qty, cost, day))

    def sell(self, qty: Decimal, method: str) -> Decimal:
        """Remove ``qty``; return the cost basis removed."""
        if qty > self.qty:
            raise ValueError(f"selling {qty} of {self.qty}")
        if method == "average":
            held, cost = self.qty, self.cost
            basis = cost * qty / held if held else Decimal(0)
            remaining = held - qty
            self.lots = [Lot(remaining, cost - basis, self.lots[0].opened)] if remaining > 0 else []
            return basis
        order = {"fifo": lambda lot: lot.opened,
                 "hifo": lambda lot: -(lot.cost / lot.qty)}[method]
        basis = Decimal(0)
        left = qty
        for lot in sorted(self.lots, key=order):
            if left <= 0:
                break
            take = min(lot.qty, left)
            part = lot.cost * take / lot.qty
            lot.cost -= part
            lot.qty -= take
            basis += part
            left -= take
        self.lots = [lot for lot in self.lots if lot.qty > 0]
        return basis


@dataclass
class TaxYear:
    """Net realised gains per calendar year, losses carried forward (design §3.5)."""
    rate: Decimal
    carried_loss: Decimal = Decimal(0)
    realised: dict[int, Decimal] = field(default_factory=dict)

    def realise(self, year: int, gain: Decimal) -> None:
        self.realised[year] = self.realised.get(year, Decimal(0)) + gain

    def due(self, year: int) -> Decimal:
        """Tax for ``year``, decided at its last close; updates the carried loss."""
        net = self.realised.get(year, Decimal(0)) - self.carried_loss
        if net <= 0:
            self.carried_loss = -net
            return Decimal(0)
        self.carried_loss = Decimal(0)
        return (net * self.rate).quantize(CENT, ROUND_DOWN)


# --- raising cash (design §6) -------------------------------------------------------------------------

def plan_raise_cash(decided_on: date, qty: Mapping[str, Decimal], prices: Mapping[str, Decimal], cash_free: Decimal,
                    need: Decimal, targets: Mapping[str, Decimal], *, cost_rate: Mapping[str, Decimal],
                    buffer_symbol: str | None = None, min_order_usd: Decimal = Decimal(1)) -> Plan:
    """Sells (by quantity, at the decision close's ``prices``) that raise ``need`` net of costs.

    Sources in order: free cash; the buffer sleeve, down to zero; then water-filling on the
    over-weights *after* the withdrawal (each asset's excess eᵢ = valueᵢ − targetᵢ·(V − need):
    sell from the largest down to a common level, so every dollar sold reduces drift); when no
    excess is left, pro-rata to the targets, which keeps the mix on target. Proceeds are grossed
    up by each asset's ``cost_rate``. Tax on the gains is settled at the year's end (§3.5), not
    here; *which lots* a sale removes is the holding's lot method.
    """
    values = {s: q * prices[s] for s, q in qty.items() if q > 0}
    liquidation = cash_free + sum((v * (ONE - cost_rate.get(s, Decimal(0))) for s, v in values.items()), Decimal(0))
    if need > liquidation:
        raise ValueError(f"exceeds_portfolio: need {need}, liquidation value {liquidation}")
    remaining = need - min(max(cash_free, Decimal(0)), need)
    sells_usd: dict[str, Decimal] = {}

    def take(symbol: str, gross: Decimal) -> Decimal:
        """Sell ``gross`` dollars of ``symbol`` (at most what is held); return the net raised."""
        gross = min(gross, values[symbol] - sells_usd.get(symbol, Decimal(0)))
        if gross <= 0:
            return Decimal(0)
        sells_usd[symbol] = sells_usd.get(symbol, Decimal(0)) + gross
        return gross * (ONE - cost_rate.get(symbol, Decimal(0)))

    if remaining > DONE and buffer_symbol in values:
        remaining -= take(buffer_symbol, remaining / (ONE - cost_rate.get(buffer_symbol, Decimal(0))))
    if remaining > DONE:
        after = cash_free + sum(values.values(), Decimal(0)) - need
        excess = {s: values[s] - sells_usd.get(s, Decimal(0)) - targets.get(s, Decimal(0)) * after
                  for s in values}
        remaining -= _water_fill(excess, remaining, cost_rate, take)
    if remaining > DONE:
        held = {s: values[s] - sells_usd.get(s, Decimal(0)) for s in values}
        weights = {s: targets.get(s, Decimal(0)) for s in held if held[s] > 0 and targets.get(s, Decimal(0)) > 0}
        if not weights:
            weights = {s: v for s, v in held.items() if v > 0}
        total = sum(weights.values(), Decimal(0))
        net_per_dollar = sum((w / total * (ONE - cost_rate.get(s, Decimal(0))) for s, w in weights.items()), Decimal(0))
        gross = remaining / net_per_dollar
        for s in sorted(weights):
            remaining -= take(s, gross * weights[s] / total)
    if remaining > DONE:
        # What a single asset could not cover (pro-rata hit a holding's size): from the largest left.
        for s in sorted(values, key=lambda x: -(values[x] - sells_usd.get(x, Decimal(0)))):
            if remaining <= DONE:
                break
            remaining -= take(s, remaining / (ONE - cost_rate.get(s, Decimal(0))))
    sells: dict[str, Decimal] = {}
    for s, usd in sells_usd.items():
        if usd < CENT:
            continue  # rounding dust, not an order
        if usd < min_order_usd and usd < values[s]:
            usd = min(min_order_usd, values[s])  # an order below the minimum is raised to it, not dropped
        units = (usd / prices[s]).quantize(QTY_QUANTUM, ROUND_UP_QTY)
        sells[s] = min(units, qty[s])
    return Plan(decided_on, dict(targets), sells, {}, "raise_cash")


def _water_fill(excess: Mapping[str, Decimal], remaining: Decimal, cost_rate: Mapping[str, Decimal],
                take: Callable[[str, Decimal], Decimal]) -> Decimal:
    """Sell from the largest positive excess down to a common level λ until ``remaining`` is raised
    or no excess is left; return the net raised. Ties in excess: the cheaper asset first."""
    over = sorted(((e, s) for s, e in excess.items() if e > 0),
                  key=lambda es: (-es[0], cost_rate.get(es[1], Decimal(0)), es[1]))
    if not over:
        return Decimal(0)
    raised = Decimal(0)
    for i in range(len(over)):
        level_next = over[i + 1][0] if i + 1 < len(over) else Decimal(0)
        group = [s for _, s in over[:i + 1]]
        # Lowering every member of the group from its current level to level_next:
        current = over[i][0]
        step = current - level_next
        net_if_full = sum((step * (ONE - cost_rate.get(s, Decimal(0))) for s in group), Decimal(0))
        if raised + net_if_full >= remaining:
            per_dollar = sum((ONE - cost_rate.get(s, Decimal(0)) for s in group), Decimal(0))
            gross_each = (remaining - raised) / per_dollar
            for s in group:
                raised += take(s, gross_each)
            return raised
        for s in group:
            raised += take(s, step)
    return raised
