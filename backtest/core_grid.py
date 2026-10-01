"""The long-term core's registered grid (core design §4): every configuration, frozen before any run.

``registration_params()`` is what ``register-core`` writes and ``run-core``
re-derives and compares, so nothing in this module can change after a
registration without the registration noticing: a changed grid needs a new
registration that names the one it supersedes.

Tickers are instruments to test, not picks; the mixes are standard reference
portfolios chosen to span the range, not recommendations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, Final

import core_alloc
from backtest import portfolio
from backtest.core_fetch import BTC, CORE_END, PRIMARY
from backtest.daily import DailyView

D = Decimal
ZERO: Final[Decimal] = Decimal(0)

FIVE: Final[tuple[str, ...]] = (PRIMARY["us_stocks"], PRIMARY["intl_stocks"], PRIMARY["bonds"], PRIMARY["gold"],
                                PRIMARY["real_estate"])
CASH: Final[str] = PRIMARY["cash"]
LOOKBACK: Final[int] = 63                 # M4's trailing window, sessions (O3)
CRYPTO_SLEEVE: Final[Decimal] = D("0.05")
BASE_MIXES: Final[tuple[str, ...]] = ("M1", "M2", "M3", "M4")
CRYPTO_MIXES: Final[tuple[str, ...]] = ("M5", "M6", "M7", "M8")
FIXED: Final[dict[str, dict[str, Decimal]]] = {
    "M1": {"VTI": D(1)},
    "M2": {"VTI": D("0.6"), "BND": D("0.4")},
    "M3": dict.fromkeys(FIVE, D("0.2")),
}
RULES: Final[tuple[str, ...]] = tuple(core_alloc.RULES)
LEVELS: Final[tuple[str, ...]] = ("optimistic", "central", "pessimistic")
SIZES: Final[tuple[Decimal, ...]] = (D("100000"), D("10000"))   # phase 1's $100,000 and the owner's (O5)
OWNER_SIZE: Final[Decimal] = D("10000")
MIN_ORDER_USD: Final[Decimal] = D("1")    # Alpaca's fractional minimum (to verify before paper mode, §9.3)
CASH_RESERVE_USD: Final[Decimal] = D("1")  # daily fee rounding never overdraws a fully invested account

#: Windows: from the first session after M4's 63-session warm-up to the last complete session fetched.
WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "full": (date(2016, 4, 6), date(2026, 9, 30)),
    "crypto": (date(2021, 4, 7), date(2026, 9, 30)),
}
#: S&P 500 closing peak to trough, confirmed on SPY's closes 2026-10-01.
STRESS: Final[dict[str, tuple[date, date]]] = {
    "2018 Q4 sell-off": (date(2018, 9, 20), date(2018, 12, 24)),
    "2020 COVID crash": (date(2020, 2, 19), date(2020, 3, 23)),
    "2022 stocks and bonds": (date(2022, 1, 3), date(2022, 10, 12)),
    "2025 drop": (date(2025, 2, 19), date(2025, 4, 8)),
}
#: The owner's tolerance, given 2026-10-01 (O5): R4 reads it at the pessimistic level.
TOLERANCE: Final[dict[str, str]] = {"max_drawdown": "-0.35", "underwater_years": "4"}
TAX_RATES: Final[tuple[Decimal, ...]] = (D("0.15"), D("0.30"))   # illustrative only
LOT_METHOD: Final[str] = "average"
BUFFERS: Final[tuple[Decimal, ...]] = (D("0"), D("0.05"))
SCENARIO_RULES: Final[tuple[str, ...]] = ("none", "quarterly")
NEED_FRACTIONS: Final[tuple[Decimal, ...]] = (D("0.02"), D("0.10"), D("0.25"))
RANDOM_NEED_DATES: Final[int] = 20
NEED_DATES_SEED: Final[int] = 11
#: Instrument substitutions (registered robustness rows, not candidates).
SUBSTITUTIONS: Final[tuple[tuple[str, dict[str, str]], ...]] = (
    ("M2", {"BND": "IEF"}), ("M2", {"BND": "TLT"}), ("M3", {"VTI": "SPY"}), ("M3", {"IAU": "GLD"}),
)
RESAMPLES: Final[int] = 10_000
MEAN_BLOCK: Final[int] = 5
BOOTSTRAP_SEED: Final[int] = 0

HISTORY_WARNING: Final[str] = (
    "About 10.5 years of one long bull market with four sharp shocks; no 2008, no 2000-2002; rates near zero "
    "for half of it; differences smaller than the intervals are not evidence; nothing here predicts the next "
    "ten years.")
NOT_ADVICE: Final[str] = ("Not financial advice. This report shows trade-offs; it ranks nothing and recommends no "
                          "mix. Tickers are instruments to test.")
READING_RULES: Final[dict[str, str]] = {
    "R1": ("Two configurations differ on a metric only if the paired 95% interval excludes zero at the central "
           "level, the sign agrees at all three cost levels, and it agrees in both halves of the window; "
           "otherwise: indistinguishable on this history."),
    "R2": ("Rebalancing helps on return for a mix under a rule if R1 holds for its mean daily excess over "
           "'none'; it helps on risk if R1 holds, with a negative sign, for the paired volatility difference. "
           "Maximum drawdown is shown without an interval."),
    "R3": ("Crypto changes the picture for Mk if R1 holds for M(k+4) - Mk on mean return or on volatility; "
           "the drawdown difference is always shown."),
    "R4": ("A mix is within your stated tolerance if, at the pessimistic level, its maximum drawdown and its "
           "longest underwater spell stay within the registered numbers. The history warning applies."),
    "no_winner": "The report never says which mix to hold.",
}


def mix_symbols(mix: str, substitute: Mapping[str, str] | None = None, buffer: Decimal = ZERO) -> tuple[str, ...]:
    base = list(FIVE) if mix in ("M4", "M8") else list(FIXED[_base(mix)])
    held = [(substitute or {}).get(s, s) for s in base]
    if mix in CRYPTO_MIXES:
        held.append(BTC)
    if buffer > 0:
        held.append(CASH)
    return tuple(held)


def _base(mix: str) -> str:
    return {"M5": "M1", "M6": "M2", "M7": "M3", "M8": "M4"}.get(mix, mix)


def weights_fn(mix: str, substitute: Mapping[str, str] | None = None, buffer: Decimal = ZERO):
    """The targets at a decision: constants, or M4/M8's inverse volatility through the view only."""
    sub = dict(substitute or {})
    sleeves: dict[str, Decimal] = {}
    if mix in CRYPTO_MIXES:
        sleeves[BTC] = CRYPTO_SLEEVE
    if buffer > 0:
        sleeves[CASH] = buffer
    if _base(mix) == "M4":
        risky = tuple(sub.get(s, s) for s in FIVE)

        def inverse_vol(view: DailyView | None) -> dict[str, Decimal]:
            if view is None:
                raise ValueError("M4 needs a point-in-time view")
            return core_alloc.inverse_vol_weights({s: view.closes(s, LOOKBACK + 1) for s in risky}, LOOKBACK,
                                                  sleeves)
        return inverse_vol, True
    base = {sub.get(s, s): w for s, w in FIXED[_base(mix)].items()}
    fixed = core_alloc.with_sleeves(base, sleeves) if sleeves else core_alloc.quantise_weights(base)
    return (lambda _view: fixed), False


@dataclass(frozen=True)
class Config:
    family: str                    # core | crypto | tax | instruments | buffer | raise_cash
    window: str
    mix: str
    rule: str
    level: str
    size: Decimal
    tax_rate: Decimal = ZERO
    buffer: Decimal = ZERO
    substitute: tuple[tuple[str, str], ...] = ()
    need: tuple[date, Decimal] | None = None
    raise_method: str = "plan"     # plan | pro_rata

    @property
    def run_key(self) -> str:
        parts = [self.family, self.window, self.mix, self.rule, self.level, f"${self.size}"]
        if self.tax_rate:
            parts.append(f"tax{self.tax_rate}")
        if self.buffer:
            parts.append(f"buffer{self.buffer}")
        if self.substitute:
            parts.append("+".join(f"{a}>{b}" for a, b in self.substitute))
        if self.need:
            parts.append(f"need{self.need[1]}@{self.need[0]}:{self.raise_method}")
        return ":".join(parts)

    def spec(self) -> portfolio.Spec:
        sub = dict(self.substitute)
        weights, needs_view = weights_fn(self.mix, sub, self.buffer)
        needs = ((self.need[0], self.need[1]),) if self.need else ()
        return portfolio.Spec(self.run_key, mix_symbols(self.mix, sub, self.buffer), weights,
                              core_alloc.RULES[self.rule], capital=self.size, min_order_usd=MIN_ORDER_USD,
                              cash_reserve_usd=CASH_RESERVE_USD, tax_rate=self.tax_rate, lot_method=LOT_METHOD,
                              cash_needs=needs, needs_view=needs_view,
                              buffer_symbol=CASH if self.buffer > 0 else None, raise_method=self.raise_method)


def grid() -> list[Config]:
    """Every registered simulation except the raise-cash scenarios (their dates need the calendar)."""
    out = [Config("core", "full", m, r, lv, size) for m in BASE_MIXES for r in RULES for lv in LEVELS for size in SIZES]
    out += [Config("crypto", "crypto", m, r, lv, size) for m in (*BASE_MIXES, *CRYPTO_MIXES) for r in RULES
            for lv in LEVELS for size in SIZES]
    out += [Config("tax", "full", m, r, "central", OWNER_SIZE, tax_rate=t) for m in BASE_MIXES for r in RULES
            for t in TAX_RATES]
    out += [Config("instruments", "full", m, r, "central", OWNER_SIZE, substitute=tuple(sub.items()))
            for m, sub in SUBSTITUTIONS for r in SCENARIO_RULES]
    out += [Config("buffer", "full", m, r, "central", OWNER_SIZE, buffer=b) for m in BASE_MIXES for r in SCENARIO_RULES
            for b in BUFFERS]
    return out


def need_dates(sessions: list[date]) -> list[date]:
    """Each stress window's trough, then ``RANDOM_NEED_DATES`` seeded dates; all at least one year
    (252 sessions) before the window's end, so 'value one year later' exists."""
    import random
    start, end = WINDOWS["full"]
    pool = [d for d in sessions if start < d and sessions.index(d) + 252 < len(sessions) and d <= end]
    troughs = [trough for _, trough in STRESS.values() if trough in pool]
    rng = random.Random(NEED_DATES_SEED)
    return troughs + sorted(rng.sample([d for d in pool if d not in troughs], RANDOM_NEED_DATES))


def scenarios(sessions: list[date]) -> list[Config]:
    """§6.3: X ∈ {2%, 10%, 25%} of the capital, at each need date, M1-M4 × buffer 0/5% × none/quarterly,
    central, the owner's size; each twice, this plan and the pro-rata baseline. (X is a fraction of the
    starting capital, a fixed amount registered in advance, not of a value known only on the day.)"""
    out = []
    for d in need_dates(sessions):
        for fraction in NEED_FRACTIONS:
            for m in BASE_MIXES:
                for b in BUFFERS:
                    for r in SCENARIO_RULES:
                        for method in ("plan", "pro_rata"):
                            out.append(Config("raise_cash", "full", m, r, "central", OWNER_SIZE, buffer=b,
                                              need=(d, (OWNER_SIZE * fraction).quantize(D("0.01"))),
                                              raise_method=method))
    return out


def selectable(window: str) -> int:
    """The deflated Sharpe's trial count (O6): mix × rule the owner could choose in that window."""
    return len(BASE_MIXES) * len(RULES) if window == "full" else (len(BASE_MIXES) + len(CRYPTO_MIXES)) * len(RULES)


def registration_params() -> dict[str, Any]:
    from backtest.costs import CORE_HALF_SPREAD_BPS, CORE_LEVELS, FeeTable

    fees = FeeTable.load()
    return {
        "universe": {"primary": PRIMARY, "crypto": BTC, "substitutes_fetched": ["SPY", "IEF", "TLT", "GLD", "SGOV"],
                     "data_end": CORE_END.isoformat()},
        "mixes": {"M1": "100% VTI", "M2": "60% VTI / 40% BND", "M3": "20% each of " + ", ".join(FIVE),
                  "M4": f"inverse volatility over {', '.join(FIVE)}: weights proportional to 1/std of the last "
                        f"{LOOKBACK} session log returns, recomputed only when a rebalance is due",
                  "M5-M8": "M1-M4 x 0.95 + 5% BTC/USD (M8: BTC fixed at 5%, the rest inverse volatility)"},
        "rules": {name: {"period": r.period, "band": str(r.band) if r.band is not None else None, "never": r.never}
                  for name, r in core_alloc.RULES.items()},
        "levels": {name: CORE_LEVELS[name].as_params() for name in LEVELS},
        "half_spread_bps": CORE_HALF_SPREAD_BPS,
        "fee_table_version": fees.version,
        "sizes_usd": [str(s) for s in SIZES], "owner_size_usd": str(OWNER_SIZE),
        "min_order_usd": str(MIN_ORDER_USD), "cash_reserve_usd": str(CASH_RESERVE_USD),
        "sizing_profile": portfolio.CORE.name, "breaker": portfolio.CORE.breaker,
        "breaker_pct": str(__import__("tier0").CIRCUIT_BREAKER_DAILY_PNL_PCT),
        "windows": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in WINDOWS.items()},
        "halves": "each window split at its midpoint session",
        "stress_windows": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in STRESS.items()},
        "tax": {"rates_illustrative": [str(t) for t in TAX_RATES], "lot_method": LOT_METHOD,
                "netting": "per calendar year, losses carried forward, paid at the next open"},
        "raise_cash": {"fractions_of_capital": [str(f) for f in NEED_FRACTIONS], "dates": "stress troughs + "
                       f"{RANDOM_NEED_DATES} seeded (seed {NEED_DATES_SEED})", "buffers": [str(b) for b in BUFFERS],
                       "rules": list(SCENARIO_RULES), "baseline": "pro_rata"},
        "substitutions": [[m, sub] for m, sub in SUBSTITUTIONS],
        "statistics": {"resamples": RESAMPLES, "mean_block_days": MEAN_BLOCK, "seed": BOOTSTRAP_SEED,
                       "riskfree": f"{CASH} total return", "intervals_at": "central",
                       "deflated_sharpe_trials": {w: selectable(w) for w in WINDOWS}},
        "reading_rules": READING_RULES,
        "tolerance": TOLERANCE,
        "history_warning": HISTORY_WARNING,
        "not_advice": NOT_ADVICE,
        "grid_runs": len(grid()),
    }
