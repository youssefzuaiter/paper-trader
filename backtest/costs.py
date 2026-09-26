"""The fill and cost model (design §3, owner decisions D1 and D3).

All money is ``Decimal``. Buys round up and sells down to $0.0001; cash
debits round up and credits down to the cent. *h* is the half-spread, *m*
the multiplier for a session's first five minutes, σ the slippage.

1. **Quote** at decision time: mid = close of the last regular 1-minute bar
   that ended by then; bid/ask = mid·(1 ∓ h·m), rounded outward to the cent.
2. **Limit buy** at *L*: fills on the first eligible bar whose low is
   strictly below *L*. Pessimistic: at *L* (the brief, literally). Central
   and optimistic: at min(*L*, *o*·(1 + h·m + σ)) (D1: the levels bracket reality).
3. **Market order**: the open *o* of the first eligible bar, ± (h·m + σ).

**Fees** come from ``fees.json`` (effective-dated, sourced). How they are
rounded is the cost level's ``fee_rounding`` (owner decision D12,
2026-09-26): ``daily`` sums each fee type per day and account and rounds
the total up to the cent, as Alpaca's fee schedule revised 2026-09-17 says,
at the optimistic and central levels; ``per_order`` rounds each fee type up
on every order (Alpaca's February 2026 support page) at the pessimistic
level. At $10 an order the difference is most of the cost.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path
from typing import Any, Final

import tier0

FEES_PATH: Final[Path] = Path(__file__).with_name("fees.json")
BPS: Final[Decimal] = Decimal("0.0001")
TICK: Final[Decimal] = Decimal("0.0001")
CENT: Final[Decimal] = Decimal("0.01")
OPENING_WINDOW: Final[timedelta] = timedelta(minutes=5)

#: The design's two spread groups.
TIGHT: Final[frozenset[str]] = frozenset({"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"})


@dataclass(frozen=True)
class CostLevel:
    name: str
    half_spread_bps_tight: Decimal     # AAPL MSFT NVDA AMZN GOOGL
    half_spread_bps_wide: Decimal      # META TSLA AMD (and anything else)
    open_multiplier: Decimal           # m, a session's first five minutes
    slippage_bps: Decimal              # σ, market orders (and D1's alternative entry)
    limit_fill_at_limit: bool          # True: pay L (pessimistic); False: min(L, o(1+hm+σ))
    fee_rounding: str                  # per_order | daily
    round_prices: bool = True          # False only for the frictionless consistency check

    def half_spread(self, symbol: str) -> Decimal:
        bps = self.half_spread_bps_tight if symbol in TIGHT else self.half_spread_bps_wide
        return bps * BPS

    def as_params(self) -> dict[str, Any]:
        return {k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self).items()}


LEVELS: Final[dict[str, CostLevel]] = {
    "optimistic": CostLevel("optimistic", Decimal("0.5"), Decimal("1"), Decimal("1"), Decimal("0"), False, "daily"),
    "central": CostLevel("central", Decimal("1.5"), Decimal("2.5"), Decimal("2"), Decimal("2"), False, "daily"),
    "pessimistic": CostLevel("pessimistic", Decimal("4"), Decimal("6"), Decimal("3"), Decimal("5"), True, "per_order"),
}
#: No spread, slippage, fees or rounding: only for checking the engine against labels (L4b).
FRICTIONLESS: Final[CostLevel] = CostLevel("frictionless", Decimal(0), Decimal(0), Decimal(1), Decimal(0), False,
                                           "none", round_prices=False)


def to_decimal(x: float) -> Decimal:
    """A bar price as the shortest decimal that round-trips the float (187.15, not 187.1499999…)."""
    return Decimal(repr(float(x)))


def _up(x: Decimal, level: CostLevel) -> Decimal:
    return x.quantize(TICK, rounding=ROUND_UP) if level.round_prices else x


def _down(x: Decimal, level: CostLevel) -> Decimal:
    return x.quantize(TICK, rounding=ROUND_DOWN) if level.round_prices else x


def multiplier(level: CostLevel, bar_start: datetime, session_open: datetime) -> Decimal:
    return level.open_multiplier if session_open <= bar_start < session_open + OPENING_WINDOW else Decimal(1)


@dataclass(frozen=True)
class Quote:
    bid: Decimal
    ask: Decimal


def quote(mid: Decimal, symbol: str, level: CostLevel, m: Decimal) -> Quote:
    spread = level.half_spread(symbol) * m
    bid = (mid * (1 - spread)).quantize(CENT, rounding=ROUND_DOWN)
    ask = (mid * (1 + spread)).quantize(CENT, rounding=ROUND_UP)
    return Quote(bid, ask)


@dataclass(frozen=True)
class FillPrice:
    price: Decimal
    ref: Decimal            # the bar's open: the frictionless price
    spread_cost: Decimal    # per share
    slippage_cost: Decimal  # per share


def market_fill(side: str, open_: Decimal, symbol: str, level: CostLevel, m: Decimal) -> FillPrice:
    h, sigma = level.half_spread(symbol) * m, level.slippage_bps * BPS
    if side == "buy":
        price = _up(open_ * (1 + h + sigma), level)
    else:
        price = _down(open_ * (1 - h - sigma), level)
    return FillPrice(price, open_, open_ * h, open_ * sigma)


def limit_buy_fill(limit: Decimal, open_: Decimal, low: Decimal, symbol: str, level: CostLevel,
                   m: Decimal) -> FillPrice | None:
    """Only through the limit: a bar whose low merely touches ``limit`` does not fill."""
    if not low < limit:
        return None
    if level.limit_fill_at_limit:
        return FillPrice(limit, open_, Decimal(0), Decimal(0))
    modelled = market_fill("buy", open_, symbol, level, m)
    if modelled.price < limit:
        return modelled
    return FillPrice(limit, open_, Decimal(0), Decimal(0))


def cash_debit(qty: Decimal, price: Decimal, level: CostLevel) -> Decimal:
    amount = qty * price
    return amount.quantize(CENT, rounding=ROUND_UP) if level.round_prices else amount


def cash_credit(qty: Decimal, price: Decimal, level: CostLevel) -> Decimal:
    amount = qty * price
    return amount.quantize(CENT, rounding=ROUND_DOWN) if level.round_prices else amount


# --- fees -------------------------------------------------------------------------------

@dataclass(frozen=True)
class FeeRow:
    fee: str
    side: str
    basis: str
    rate: Decimal
    cap: Decimal | None
    start: date
    end: date | None

    def applies(self, day: date, side: str) -> bool:
        return self.side in {side, "both"} and self.start <= day and (self.end is None or day <= self.end)


class FeeTable:
    def __init__(self, rows: list[FeeRow], version: str) -> None:
        self.rows = rows
        self.version = version

    @classmethod
    def load(cls, path: Path = FEES_PATH) -> FeeTable:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = [FeeRow(r["fee"], r["side"], r["basis"], Decimal(r["rate"]),
                       Decimal(r["cap"]) if r["cap"] is not None else None, date.fromisoformat(r["from"]),
                       date.fromisoformat(r["to"]) if r["to"] else None) for r in raw["fees"]]
        _check_coverage(rows)
        return cls(rows, raw["version"])

    def exact(self, day: date, side: str, qty: Decimal, price: Decimal) -> dict[str, Decimal]:
        """Each fee type's exact amount for one order, before any rounding."""
        out: dict[str, Decimal] = {}
        for row in self.rows:
            if not row.applies(day, side):
                continue
            amount = row.rate * (qty * price if row.basis == "principal" else qty)
            if row.cap is not None:
                amount = min(amount, row.cap)
            out[row.fee] = out.get(row.fee, Decimal(0)) + amount
        return out

    def order_fees(self, day: date, side: str, qty: Decimal, price: Decimal, rounding: str) -> dict[str, Decimal]:
        """What one order is charged now: rounded up per fee type (``per_order``),
        or exact, with the daily rounding settled by ``DailyFees`` (``daily``)."""
        exact = self.exact(day, side, qty, price)
        if rounding == "per_order":
            return {k: v.quantize(CENT, rounding=ROUND_CEILING) for k, v in exact.items()}
        if rounding in {"daily", "none"}:
            return exact
        raise ValueError(f"unknown fee rounding {rounding!r}")


def _check_coverage(rows: list[FeeRow]) -> None:
    """A fee must never silently stop applying: per fee type, consecutive
    rows without gaps or overlaps, the last one open-ended."""
    for fee in {r.fee for r in rows}:
        mine = sorted((r for r in rows if r.fee == fee), key=lambda r: r.start)
        for before, after in itertools.pairwise(mine):
            if before.end is None or after.start != before.end + timedelta(days=1):
                raise ValueError(f"fees.json: {fee} rows from {before.start} and {after.start} leave a gap or overlap")
        if mine[-1].end is not None:
            raise ValueError(f"fees.json: {fee} stops applying after {mine[-1].end}; the last row must be open-ended")


class DailyFees:
    """Accumulates exact fees per day and fee type; the day's rounding-up is its own charge."""

    def __init__(self) -> None:
        self._days: dict[date, dict[str, Decimal]] = {}

    def add(self, day: date, fees: dict[str, Decimal]) -> None:
        bucket = self._days.setdefault(day, {})
        for k, v in fees.items():
            bucket[k] = bucket.get(k, Decimal(0)) + v

    def rounding_charge(self, day: date) -> Decimal:
        bucket = self._days.get(day, {})
        return sum((v.quantize(CENT, rounding=ROUND_CEILING) - v for v in bucket.values()), Decimal(0))


def round_trip_fraction(symbol: str, level: CostLevel, fees: FeeTable, *, day: date, price: Decimal,
                        notional: Decimal, entry_in_opening_window: bool) -> float:
    """Expected cost of one round trip as a fraction of notional: the
    threshold rule's "central round-trip cost" (design §6). Entry as a
    marketable limit filled at the modelled price (D1), exit as a market
    sell outside the opening window, fees at ``notional``. Daily fee rounding
    is not attributable to one trade and is left out here; the replay charges it."""
    m = level.open_multiplier if entry_in_opening_window else Decimal(1)
    h, sigma = level.half_spread(symbol), level.slippage_bps * BPS
    if level.limit_fill_at_limit:  # pays plan_buy's limit: the ask plus the marketable-limit buffer
        entry = (1 + h * m) * (1 + tier0.LIMIT_BUFFER_PCT / 100) - 1
    else:
        entry = h * m + sigma
    spread_slip = entry + h + sigma
    qty = notional / price
    charged = {**{f"buy_{k}": v for k, v in fees.order_fees(day, "buy", qty, price, level.fee_rounding).items()},
               **{f"sell_{k}": v for k, v in fees.order_fees(day, "sell", qty, price, level.fee_rounding).items()}}
    return float(spread_slip + sum(charged.values(), Decimal(0)) / notional)
