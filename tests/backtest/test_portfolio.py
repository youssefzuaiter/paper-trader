"""The long-term core's engine (core design §3) on synthetic markets, and C7 on phase 1's real data."""

from __future__ import annotations

import ast
import inspect
import json
import math
import random
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import core_alloc as ca
from backtest import portfolio, strategies
from backtest.calendar import Session
from backtest.costs import FRICTIONLESS, LEVELS, FeeTable
from backtest.daily import DailyMarket
from backtest.data import LookAheadError
from swarm.common import NEW_YORK

D = Decimal
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name("fixtures") / "phase1_hold_golden.json"
REAL = (ROOT / ".cache" / "backtest" / "calendar.parquet").exists() and (ROOT / ".cache" / "training").exists()


def sessions(n: int, first: date = date(2023, 1, 3)) -> list[Session]:
    out, d = [], first
    while len(out) < n:
        if d.weekday() < 5:
            out.append(Session(d, datetime.combine(d, time(9, 30), NEW_YORK).astimezone(UTC),
                               datetime.combine(d, time(16, 0), NEW_YORK).astimezone(UTC), False))
        d += timedelta(days=1)
    return out


def market(paths: dict[str, list[float]], n: int, *, gaps: dict[str, dict[int, float]] | None = None) -> DailyMarket:
    """Closes from ``paths``; each open equals the previous close times an optional gap factor."""
    ss = sessions(n)
    adjusted = {}
    for symbol, closes in paths.items():
        bars = {}
        for k, s in enumerate(ss):
            prev = closes[k - 1] if k else closes[0]
            bars[s.date] = (prev * (gaps or {}).get(symbol, {}).get(k, 1.0), closes[k])
        adjusted[symbol] = bars
    return DailyMarket(ss, adjusted, adjusted)


def walk(rng: random.Random, n: int, vol: float, start: float = 100.0) -> list[float]:
    out = [start]
    for _ in range(n - 1):
        out.append(round(out[-1] * math.exp(rng.gauss(0.0003, vol)), 4))
    return out


def fixed(weights: dict[str, str]):
    w = {s: D(v) for s, v in weights.items()}
    return lambda _view: w


@pytest.fixture(scope="module")
def fees() -> FeeTable:
    return FeeTable.load()


def test_core_never_overdraws_and_value_is_cash_plus_positions(fees: FeeTable) -> None:
    """C8 on a random market: every day, value = cash + Σ positions, cash ≥ 0, weights sum ≤ 1."""
    rng = random.Random(3)
    n = 300
    m = market({"VTI": walk(rng, n, 0.015), "BND": walk(rng, n, 0.004), "IAU": walk(rng, n, 0.01)}, n)
    for rule in ("monthly", "band5", "quarterly_band5", "none"):
        spec = portfolio.Spec("M", ("VTI", "BND", "IAU"), fixed({"VTI": "0.5", "BND": "0.3", "IAU": "0.2"}),
                              ca.RULES[rule], capital=D("10000"))
        days = portfolio.simulate(m, spec, 1, n - 1, LEVELS["pessimistic"], fees)
        for d in days:
            assert d.cash >= 0, (rule, d.date, d.cash)
            assert sum(d.weights.values()) <= 1
            assert d.fees >= 0 and d.exec_cost >= 0
        assert days[0].rebalanced and days[0].traded > D("9900")


def test_frictionless_buy_and_hold_matches_an_independent_calculation(fees: FeeTable) -> None:
    """C9's idea, small: frictionless 'none' = Σ wᵢ·closeᵢ/open₀ᵢ, computed separately."""
    rng = random.Random(4)
    n = 120
    paths = {"VTI": walk(rng, n, 0.012), "BND": walk(rng, n, 0.003)}
    m = market(paths, n)
    spec = portfolio.Spec("M2", ("VTI", "BND"), fixed({"VTI": "0.6", "BND": "0.4"}), ca.RULES["none"],
                          capital=D("100000"), min_order_usd=D("0"))
    days = portfolio.simulate(m, spec, 1, n - 1, FRICTIONLESS, fees)
    open0 = {s: paths[s][0] for s in paths}  # session 1 opens at session 0's close
    for k, d in enumerate(days, start=1):
        ref = 100000 * (0.6 * paths["VTI"][k] / open0["VTI"] + 0.4 * paths["BND"][k] / open0["BND"])
        assert float(d.value) == pytest.approx(ref, rel=1e-8)


def test_a_rebalance_is_decided_at_a_close_and_filled_at_the_next_open(fees: FeeTable) -> None:
    """C3: the fill's session is strictly after the decision's, at that session's open."""
    rng = random.Random(5)
    n = 90
    m = market({"VTI": walk(rng, n, 0.02), "BND": walk(rng, n, 0.003)}, n)
    spec = portfolio.Spec("M2", ("VTI", "BND"), fixed({"VTI": "0.6", "BND": "0.4"}), ca.RULES["monthly"])
    days = portfolio.simulate(m, spec, 1, n - 1, LEVELS["central"], fees)
    decided = [i for i, d in enumerate(days) if d.decided]
    assert decided
    for i in decided:
        assert days[i + 1].rebalanced
        for order in days[i + 1].orders:
            assert D(order["ref"]) == m.open(order["symbol"], i + 2)  # days[0] is session 1
        assert ca.period_ends(days[i].date, days[i + 1].date) >= {"month"}


def test_the_breaker_defers_a_rebalance_with_buys_and_redecides_it(fees: FeeTable) -> None:
    """§3.1 a: a gap down of more than 2.5% at the open blocks any plan with a buy, whole."""
    n = 60
    ss = sessions(n)
    month_end = next(k for k in range(1, n - 1) if ss[k].date.month != ss[k + 1].date.month)
    flat = [100.0] * n
    vti = [100.0] * n
    for k in range(month_end - 3, month_end + 1):
        vti[k] = 130.0  # VTI rallies into the month end, so the rebalance sells VTI and buys BND
    for k in range(month_end + 1, n):
        vti[k] = 120.0
    gap = {"VTI": {month_end + 1: 0.85}}  # the next morning opens 15% lower
    m = market({"VTI": vti, "BND": flat}, n, gaps=gap)
    spec = portfolio.Spec("M2", ("VTI", "BND"), fixed({"VTI": "0.6", "BND": "0.4"}), ca.RULES["monthly"])
    days = portfolio.simulate(m, spec, 1, n - 1, LEVELS["central"], fees)
    by_k = {k: days[k - 1] for k in range(1, n)}
    assert by_k[month_end].decided == "rebalance"
    assert by_k[month_end + 1].deferred and not by_k[month_end + 1].rebalanced
    assert by_k[month_end + 1].decided == "rebalance"          # re-decided at that close
    assert by_k[month_end + 2].rebalanced and not by_k[month_end + 2].deferred
    # The same market without the breaker (phase 1's profile flag) fills on the gap day.
    no_breaker = portfolio.Profile("X", "core", None, False, False, True, True)
    other = portfolio.simulate(m, spec, 1, n - 1, LEVELS["central"], fees, no_breaker)
    assert other[month_end].rebalanced


def test_m4_weights_read_only_the_view(fees: FeeTable) -> None:
    """C1 in miniature: corrupting every price after the decision changes no weight."""
    rng = random.Random(6)
    n = 140
    paths = {"VTI": walk(rng, n, 0.015), "BND": walk(rng, n, 0.004)}
    m = market(paths, n)
    k = 100
    w = ca.inverse_vol_weights({s: m.view(k).closes(s, 64) for s in paths}, 63)
    corrupted = {s: c[: k + 1] + [x * 1e6 for x in c[k + 1:]] for s, c in paths.items()}
    w2 = ca.inverse_vol_weights({s: market(corrupted, n).view(k).closes(s, 64) for s in paths}, 63)
    assert w == w2
    with pytest.raises(LookAheadError):
        m.view(10).closes("VTI", 64)


def test_a_cash_need_is_raised_by_the_next_open_and_never_overdraws(fees: FeeTable) -> None:
    rng = random.Random(8)
    n = 80
    m = market({"VTI": walk(rng, n, 0.01), "BND": walk(rng, n, 0.003)}, n)
    need_day = sessions(n)[40].date
    spec = portfolio.Spec("M2", ("VTI", "BND"), fixed({"VTI": "0.6", "BND": "0.4"}), ca.RULES["none"],
                          capital=D("10000"), cash_needs=((need_day, D("2500")),))
    days = portfolio.simulate(m, spec, 1, n - 1, LEVELS["pessimistic"], fees)
    i = next(i for i, d in enumerate(days) if d.date == need_day)
    assert days[i].decided == "raise_cash"
    assert days[i + 1].flow == D("2500") and days[i + 1].cash >= 0
    assert all(o["side"] == "sell" for o in days[i + 1].orders)


# --- C7: phase 1 through the new engine -------------------------------------------------------------------

def test_run_hold_only_calls_the_engine() -> None:
    """C7's AST half: there is no second engine."""
    tree = ast.parse(inspect.getsource(strategies.run_hold))
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "portfolio.simulate" in calls
    assert not calls & {"costs.market_fill", "costs.cash_debit", "costs.cash_credit", "fees.order_fees"}


@pytest.mark.skipif(not REAL, reason="needs the fetched caches under .cache/")
def test_c7_run_hold_reproduces_phase_1_exactly() -> None:
    from backtest.data import MarketData
    from backtest.fetch import SYMBOLS, Paths
    from backtest.pipeline import Plan

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["generated_from_commit"].startswith("f700c0d")
    body = json.dumps(golden["series"], sort_keys=True, separators=(",", ":"))
    import hashlib
    assert hashlib.sha256(body.encode()).hexdigest() == golden["series_sha256"]
    market_ = MarketData.load(Paths(ROOT / ".cache" / "backtest"), symbols=SYMBOLS,
                              training_cache=ROOT / ".cache" / "training")
    plan = Plan()
    ss = market_.calendar.sessions_between(plan.first_session, plan.last_session)
    fees = FeeTable.load()
    for name, band in (("S0", None), ("S1", plan.s1_band)):
        for level in ("optimistic", "central", "pessimistic"):
            days = strategies.run_hold(market_, SYMBOLS, ss, LEVELS[level], fees, band=band)
            got = [[d.date.isoformat(), str(d.value), str(d.traded), str(d.costs)] for d in days]
            assert got == golden["series"][f"{name}:{level}"], f"{name} {level}"
