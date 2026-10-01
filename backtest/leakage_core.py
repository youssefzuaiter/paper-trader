"""The long-term core's leakage and consistency checks C1-C12 (core design §7), each with a broken twin.

As in phase 1, every check returns a ``Check`` verdict, and the same function
runs twice: on the real configuration, where it must pass, and on a twin that
breaks its rule on purpose, where it must fail. ``leakage-core`` runs them on
real data and records them; ``run-core`` refuses to start without a passing
run at the same commit (C10).

Two departures from the design's wording, both to make a check test what it
claims:

* **C1** checks the window on both sides: changing close *t* or *t*−63 (the
  oldest close a 63-return window reads) changes M4's weights, changing *t*−64
  does not. The design's text says "*t*−63 does not", contradicting its own
  definition of the window (closes *t*−63 … *t*).
* **C5** rescales a whole span of history (every price before a date *d*, as a
  proportional dividend adjustment does) and requires every decision whose
  window lies inside that span to be identical. Rescaling "before a random
  date" and comparing *every* return cannot pass: it changes the real return
  across *d*.
"""

from __future__ import annotations

import ast
import bisect
import copy
import hashlib
import inspect
import itertools
import json
import random
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import numpy as np

import core_alloc
from backtest import core_grid as G
from backtest import leakage, portfolio
from backtest.core_fetch import BTC, ETFS, CorePaths
from backtest.costs import CORE_LEVELS, FRICTIONLESS, LEVELS, FeeTable
from backtest.daily import BAR, DailyMarket, btc_samples
from backtest.data import LookAheadError
from backtest.leakage import Check
from swarm.common import NEW_YORK

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GOLDEN: Final[Path] = ROOT / "tests" / "backtest" / "fixtures" / "phase1_hold_golden.json"
D = Decimal
NO_BREAKER: Final[portfolio.Profile] = portfolio.Profile("CORE-no-breaker", "core", None, False, False, True, True)


def twin(name: str, check: Check) -> Check:
    """A twin passes when its check *fails*."""
    return Check(name, not check.passed, check.detail)


# --- market surgery for the twins -------------------------------------------------------------------------

def with_closes(market: DailyMarket, edit: Callable[[str, list[float | None]], list[float | None]],
                symbols: tuple[str, ...]) -> DailyMarket:
    """A shallow copy with ``symbols``' adjusted close series replaced by ``edit(symbol, closes)``."""
    out = copy.copy(market)
    out._close = dict(market._close)
    for s in symbols:
        out._close[s] = edit(s, list(market._close[s]))
    return out


class Shim:
    """A market whose fill-side prices are rewired: the broken engines of C3, C4 and C8's twins."""

    def __init__(self, market: DailyMarket, *, open_fn: Callable[[str, int], Decimal] | None = None,
                 close_fn: Callable[[str, int], Decimal] | None = None) -> None:
        self._m, self._open, self._close = market, open_fn, close_fn
        self.sessions = market.sessions

    def open(self, symbol: str, k: int) -> Decimal:
        return self._open(symbol, k) if self._open else self._m.open(symbol, k)

    def close(self, symbol: str, k: int) -> Decimal:
        return self._close(symbol, k) if self._close else self._m.close(symbol, k)

    def factor(self, symbol: str, k: int) -> Decimal:
        return self._m.factor(symbol, k)

    def asset_class(self, symbol: str) -> str:
        return self._m.asset_class(symbol)

    def view(self, k: int):
        return self._m.view(k)


def _window(name: str, market: DailyMarket) -> tuple[int, int]:
    a, b = G.WINDOWS[name]
    return market.index(a), market.index(b)


def _run(market: Any, cfg: G.Config, level: Any = None, profile: portfolio.Profile = portfolio.CORE,
         spec: portfolio.Spec | None = None) -> list[portfolio.Day]:
    first, last = _window(cfg.window, market._m if isinstance(market, Shim) else market)
    return portfolio.simulate(market, spec or cfg.spec(), first, last, level or CORE_LEVELS[cfg.level],
                              FeeTable.load(), profile)


# --- C1 the future canary for M4 --------------------------------------------------------------------------------

def c1_m4_canary(market: DailyMarket, *, instants: int, seed: int, leaky: bool = False) -> Check:
    """At seeded decision instants, every close after *t* × 10⁶ leaves M4's weights bit-identical;
    close *t* and *t*−63 change them, *t*−64 does not. ``leaky``: the window reaches *t*+1."""
    weights, _ = G.weights_fn("M4")
    if leaky:
        def weights(view, _m=market):
            k = view.k
            return core_alloc.inverse_vol_weights(
                {s: [float(c) for c in view._market._close[s][k - G.LOOKBACK + 1:k + 2]] for s in G.FIVE},
                G.LOOKBACK)
    first, last = _window("full", market)
    rng = random.Random(seed)
    ks = sorted(rng.sample(range(first - 1, last), instants))
    problems: list[str] = []
    for k in ks:
        base = weights(market.view(k))
        future = with_closes(market, lambda s, c, k=k: c[: k + 1] + [x * 1e6 if x else x for x in c[k + 1:]], G.FIVE)
        if weights(future.view(k)) != base:
            problems.append(f"{market.dates[k]}: future prices changed the weights")
        for offset, should_change in ((0, True), (G.LOOKBACK, True), (G.LOOKBACK + 1, False)):
            j = k - offset
            moved = with_closes(market, lambda s, c, j=j: c[:j] + [c[j] * 1.01] + c[j + 1:], ("VTI",))
            changed = weights(moved.view(k)) != base
            if changed != should_change:
                problems.append(f"{market.dates[k]}: close t-{offset} {'did not change' if should_change else 'changed'}"
                                " the weights")
    return Check("C1", not problems, {"instants": len(ks), "problems": problems[:10]})


# --- C2 the point-in-time oracle ------------------------------------------------------------------------------------

def c2_pit_oracle(market: DailyMarket, *, instants: int, seed: int, delay: timedelta | None = None) -> Check:
    """A view at an instant serves exactly the closes published by then (brute force), and reading past
    it raises. ``delay``: the twin's publication delay (0 serves a bar at its close, before +15 min)."""
    rng = random.Random(seed)
    published = np.array([(s.close_at + timedelta(minutes=15)).timestamp() for s in market.sessions])
    fast_times = [s.close_at + (delay if delay is not None else timedelta(minutes=15)) for s in market.sessions]
    lo, hi = market.sessions[0].open_at.timestamp(), market.sessions[-1].close_at.timestamp() + 86400
    problems: list[str] = []
    for _ in range(instants):
        t = datetime.fromtimestamp(rng.uniform(lo, hi), UTC)
        if rng.random() < 0.3:  # dwell on the boundary: the 15 minutes after a close
            s = market.sessions[rng.randrange(len(market.sessions))]
            t = s.close_at + timedelta(seconds=rng.uniform(0, 900))
        fast = bisect.bisect_right(fast_times, t) - 1
        brute = int(np.sum(published <= t.timestamp())) - 1
        if fast != brute:
            problems.append(f"{t}: index {fast} served, {brute} published")
            continue
        if fast < G.LOOKBACK + 1:
            continue
        view = market.view(fast)
        symbol = rng.choice(G.FIVE)
        n = rng.randint(1, G.LOOKBACK + 1)
        want = [float(x) for x in market._close[symbol][: brute + 1][-n:]]  # sessions published by t
        if view.closes(symbol, n) != want:
            problems.append(f"{t}: {symbol} closes differ")
        try:
            view.require(fast + 1)
            problems.append(f"{t}: reading session {fast + 1} did not raise")
        except LookAheadError:
            pass
    return Check("C2", not problems, {"instants": instants, "problems": problems[:10]})


# --- C3 execution lag ---------------------------------------------------------------------------------------------------

C3_CONFIGS: Final[tuple[G.Config, ...]] = (
    G.Config("core", "full", "M3", "monthly", "central", G.OWNER_SIZE),
    G.Config("core", "full", "M4", "band5", "pessimistic", D("100000")),
    G.Config("crypto", "crypto", "M7", "quarterly", "central", G.OWNER_SIZE),
)


def c3_execution_lag(market: DailyMarket, *, fill_at_decision_close: bool = False) -> Check:
    """Every fill is on the session after its decision, at that session's open."""
    prices = Shim(market, open_fn=lambda s, k: market.close(s, k - 1)) if fill_at_decision_close else market
    problems: list[str] = []
    fills = 0
    for cfg in C3_CONFIGS:
        days = _run(prices, cfg)
        first, _ = _window(cfg.window, market)
        for i, day in enumerate(days):
            for o in day.orders:
                fills += 1
                if i > 0 and days[i - 1].decided is None:
                    problems.append(f"{cfg.run_key} {day.date}: a fill with no decision the session before")
                if D(o["ref"]) != market.open(o["symbol"], first + i):
                    problems.append(f"{cfg.run_key} {day.date} {o['symbol']}: filled at {o['ref']}, not the open")
    return Check("C3", not problems and fills > 0, {"fills": fills, "problems": problems[:10]})


# --- C4 the band trigger ------------------------------------------------------------------------------------------------

C4_CONFIGS: Final[tuple[G.Config, ...]] = tuple(
    G.Config("core", "full", m, r, "central", G.OWNER_SIZE) for m in ("M2", "M3")
    for r in ("band5", "band10", "monthly_band5", "quarterly_band5"))


def c4_band_trigger(market: DailyMarket, *, trigger_reads_next_open: bool = False) -> Check:
    """Every band decision, recomputed from the stored end-of-day units at the true closes, matches."""
    prices = Shim(market, close_fn=lambda s, k: market.open(s, min(k + 1, len(market.sessions) - 1))) \
        if trigger_reads_next_open else market
    problems: list[str] = []
    checked = 0
    for cfg in C4_CONFIGS:
        days = _run(prices, cfg)
        first, _ = _window(cfg.window, market)
        targets = cfg.spec().weights(None)
        rule = core_alloc.RULES[cfg.rule]
        for i, day in enumerate(days[:-1]):
            if day.deferred or (i > 0 and days[i - 1].deferred):
                continue  # a deferred plan is re-decided regardless of the period (§3.1 a)
            closes = {s: market.close(s, first + i) for s in day.positions}
            weights = core_alloc.current_weights(day.positions, closes, day.cash)
            due = rule.due(core_alloc.period_ends(day.date, days[i + 1].date), weights, targets)
            checked += 1
            if due != (day.decided == "rebalance"):
                problems.append(f"{cfg.run_key} {day.date}: recomputed {due}, engine {day.decided}")
    return Check("C4", not problems, {"decisions": checked, "problems": problems[:10]})


# --- C5 adjustment invariance ----------------------------------------------------------------------------------------

def c5_adjustment_invariance(market: DailyMarket, *, seed: int, price_differences: bool = False) -> Check:
    """Rescale each instrument's prices before a seeded date *d* by its own factor (as its dividends
    rescale its own history):
    every decision and weight before *d* is identical and every return before it equal within 1e-12
    (frictionless; weights within 1e-9, as units are whole 10⁻⁹ shares).
    ``price_differences``: M4's volatility on price differences instead of returns."""
    rng = random.Random(seed)
    first, last = _window("full", market)
    d = rng.randrange(first + 400, last - 10)
    # Each instrument its own factor, as each one's dividends rescale only its own history.
    factors = {s: round(rng.uniform(0.4, 3.0), 4) for s in sorted(market._close)}
    scaled = copy.copy(market)
    scaled._open = {s: [x * factors[s] if (x is not None and j < d) else x for j, x in enumerate(v)]
                    for s, v in market._open.items()}
    scaled._close = {s: [x * factors[s] if (x is not None and j < d) else x for j, x in enumerate(v)]
                     for s, v in market._close.items()}
    cfg = G.Config("core", "full", "M4", "monthly", "central", D("100000"))  # many decisions before d
    spec = cfg.spec()
    if price_differences:
        def by_differences(view):
            closes = {s: view.closes(s, G.LOOKBACK + 1) for s in G.FIVE}
            inv = {s: 1.0 / core_alloc.sample_std([b - a for a, b in itertools.pairwise(c)])
                   for s, c in closes.items()}
            return core_alloc.quantise_weights(inv)
        spec = portfolio.Spec(spec.name, spec.symbols, by_differences, spec.rule, capital=spec.capital,
                              min_order_usd=D(0), needs_view=True)
    else:
        spec = portfolio.Spec(spec.name, spec.symbols, spec.weights, spec.rule, capital=spec.capital,
                              min_order_usd=D(0), needs_view=True)
    a = portfolio.simulate(market, spec, first, last, FRICTIONLESS, FeeTable.load(), NO_BREAKER)
    b = portfolio.simulate(scaled, spec, first, last, FRICTIONLESS, FeeTable.load(), NO_BREAKER)
    cut = d - first - 1   # days[i] is session first + i: compare decisions and returns strictly before d
    problems: list[str] = []
    for i in range(cut):
        if a[i].decided != b[i].decided:
            problems.append(f"{a[i].date}: decision {a[i].decided} vs {b[i].decided}")
        prev_a = a[i - 1].value if i else spec.capital
        prev_b = b[i - 1].value if i else spec.capital
        ra, rb = float(a[i].value / prev_a - 1), float(b[i].value / prev_b - 1)
        if abs(ra - rb) > 1e-12:
            problems.append(f"{a[i].date}: return {ra} vs {rb}")
        # Units are whole 10⁻⁹ shares, so rescaled weights agree to ~1e-11, not bit for bit.
        drift = max((abs(a[i].weights.get(s, D(0)) - b[i].weights.get(s, D(0))) for s in set(a[i].weights)
                     | set(b[i].weights)), default=D(0))
        if drift > D("1e-9"):
            problems.append(f"{a[i].date}: weights differ by {drift}")
    decisions = sum(1 for i in range(cut) if a[i].decided)
    return Check("C5", not problems and decisions > 0,
                 {"rescaled_before": market.dates[d].isoformat(), "factors": len(factors),
                  "decisions_compared": decisions,
                  "problems": problems[:10]})


# --- C6 the crypto cutoff --------------------------------------------------------------------------------------------

def c6_crypto_cutoff(market: DailyMarket, paths: CorePaths, *, sampler: str = "engine") -> Check:
    """No BTC close sample uses a bar ending after its session's close (half-days, DST included); the
    open sample is the bar starting at the open. ``sampler``: the twins (``bar_after_close``: the bar
    starting at the close; ``utc_2100``: a fixed 21:00 UTC cutoff, blind to DST and half-days)."""
    from backtest import parquet
    cols = parquet.read_numpy(paths.btc_5min, "epoch(t)::BIGINT AS t, o, c", "ORDER BY t")
    t = cols["t"]
    sessions = [s for s in market.sessions if s.date >= date(2021, 1, 4)]
    samples = btc_samples(sessions, [datetime.fromtimestamp(int(e), UTC) for e in t], list(cols["o"]),
                          list(cols["c"]))
    if sampler != "engine":
        samples = {}
        for s in sessions:
            cutoff = (s.close_at.timestamp() if sampler == "bar_after_close"
                      else datetime.combine(s.date, datetime.min.time(), UTC).timestamp() + 21 * 3600)
            i = int(np.searchsorted(t, cutoff, side="right")) - 1
            j = int(np.searchsorted(t, s.open_at.timestamp(), side="left"))
            samples[s.date] = (float(cols["o"][j]), float(cols["c"][i]))
    problems: list[str] = []
    half_days = dst = 0
    for s in sessions:
        end_limit = s.close_at.timestamp()
        i = int(np.searchsorted(t, end_limit - BAR.total_seconds(), side="right")) - 1  # last bar ending by close
        j = int(np.searchsorted(t, s.open_at.timestamp(), side="left"))
        want = (float(cols["o"][j]), float(cols["c"][i]))
        if samples.get(s.date) != want:
            problems.append(f"{s.date}: sample {samples.get(s.date)} vs bar ending by {s.close_at} {want}")
        half_days += s.half_day
        dst += s.close_at.astimezone(NEW_YORK).utcoffset() != (s.close_at - timedelta(days=7)).astimezone(
            NEW_YORK).utcoffset()
    return Check("C6", not problems, {"sessions": len(sessions), "half_days": half_days, "dst_weeks": dst,
                                      "problems": problems[:10]})


# --- C7 phase 1 through the new engine -------------------------------------------------------------------------

def c7_phase1(root: Path, store: Any, run_id: str | None, *, profile: portfolio.Profile = portfolio.PHASE1,
              fee_rounding: str | None = None) -> Check:
    """``run_hold``, now a call to ``portfolio.simulate``, reproduces the golden file of every phase 1
    S0/S1 day exactly, and the S0/S1 rows of the clean phase 1 run. Twins: cash-safe sizing (the CORE
    profile) or daily fee rounding in the PHASE1 profile."""
    from backtest import pipeline, report, strategies
    from backtest.daily import Phase1Prices
    from backtest.data import MarketData
    from backtest.fetch import SYMBOLS, Paths

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    body = json.dumps(golden["series"], sort_keys=True, separators=(",", ":"))
    problems: list[str] = []
    if hashlib.sha256(body.encode()).hexdigest() != golden["series_sha256"]:
        problems.append("the golden file's series do not match its hash")
    tree = ast.parse(inspect.getsource(strategies.run_hold))
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    if "portfolio.simulate" not in calls or calls & {"costs.market_fill", "costs.cash_debit", "fees.order_fees"}:
        problems.append(f"run_hold is not just a call to portfolio.simulate: {sorted(calls)}")
    market = MarketData.load(Paths(root / ".cache" / "backtest"), symbols=SYMBOLS,
                             training_cache=root / ".cache" / "training")
    plan = pipeline.Plan()
    sessions = market.calendar.sessions_between(plan.first_session, plan.last_session)
    fees = FeeTable.load()
    prof = profile if fee_rounding is None else portfolio.Profile(
        profile.name + "-twin", profile.sizing, fee_rounding, profile.open_multiplier, profile.breaker,
        profile.drift_includes_cash, profile.raw_fees)
    prices = Phase1Prices(market.daily, SYMBOLS, sessions)
    for name, band in (("S0", None), ("S1", plan.s1_band)):
        for level in ("optimistic", "central", "pessimistic"):
            days = portfolio.simulate(prices, portfolio.hold_spec(SYMBOLS, capital=D("100000"), band=band), 0,
                                      len(sessions) - 1, LEVELS[level], fees, prof)
            got = [[d.date.isoformat(), str(d.value), str(d.traded), str(d.costs)] for d in days]
            if got != golden["series"][f"{name}:{level}"]:
                bad = sum(g != w for g, w in zip(got, golden["series"][f"{name}:{level}"], strict=False))
                problems.append(f"{name} {level}: {bad} of {len(got)} days differ from the golden file")
    rows_compared = 0
    if run_id and not problems:
        recorded = json.loads(store.experiment(run_id)["metrics"])["table"]
        dates = [s.date for s in sessions]
        for name, band in (("S0", None), ("S1", plan.s1_band)):
            for level in ("optimistic", "central", "pessimistic"):
                s = pipeline.hold_series(name, level, D(1000), market, SYMBOLS, dates, fees, band)
                row = report.series_row(s, resamples=plan.resamples, seed=plan.bootstrap_seed)
                ours = {"total": row["total"], "annualised": row["annualised"], "max_drawdown": row["max_drawdown"],
                        "drawdown_days": row["drawdown_days"], "sharpe": row["sharpe"].as_dict()}
                for size in plan.sizes:
                    theirs = recorded[f"{name}|{level}|{size}"]
                    if {k: theirs[k] for k in ours} != ours:
                        problems.append(f"{name}|{level}|{size}: differs from {run_id}")
                    rows_compared += 1
    return Check("C7", not problems, {"golden_commit": golden["generated_from_commit"][:8], "phase1_run": run_id,
                                      "rows_compared": rows_compared, "problems": problems[:10]})


# --- C8 accounting ----------------------------------------------------------------------------------------------

def c8_accounting(market: DailyMarket, configs: list[G.Config], *, uncapped_buys: bool = False) -> Check:
    """Every day of every run: value = cash + Σ units × close exactly, cash ≥ 0, weights sum ≤ 1,
    fees ≥ 0. ``uncapped_buys``: buys not scaled to the cash available."""
    original = core_alloc.scale_buys
    if uncapped_buys:
        core_alloc.scale_buys = lambda buys, available, cost_rate, cents=True: dict(buys)  # type: ignore[assignment]
    problems: list[str] = []
    days_checked = 0
    try:
        for cfg in configs:
            days = _run(market, cfg)
            first, _ = _window(cfg.window, market)
            for i, d in enumerate(days):
                days_checked += 1
                marked = d.cash + sum((q * market.close(s, first + i) for s, q in d.positions.items()), D(0))
                if marked != d.value or d.cash < 0 or sum(d.weights.values()) > 1 or d.fees < 0:
                    problems.append(f"{cfg.run_key} {d.date}: value {d.value} marked {marked} cash {d.cash}")
                    break
    finally:
        core_alloc.scale_buys = original
    return Check("C8", not problems, {"runs": len(configs), "days": days_checked, "problems": problems[:10]})


# --- C9 an independent reference -------------------------------------------------------------------------------

def reference_values(market: DailyMarket, weights: dict[str, float], first: int, last: int, rule: str,
                     capital: float, *, shift: int = 0) -> np.ndarray:
    """Frictionless buy-and-hold ('none') or monthly constant-mix, in numpy floats, written apart from
    the engine: decide at a month's last close, sell by quantity and buy by notional from that close,
    fill at the next open, buys scaled down if the sells' proceeds fall short. ``shift``: the twin."""
    syms = list(weights)
    o = np.array([[float(market._open[s][k]) for s in syms] for k in range(len(market.sessions))])
    c = np.array([[float(market._close[s][k]) for s in syms] for k in range(len(market.sessions))])
    w = np.array([weights[s] for s in syms])
    units, cash = np.zeros(len(syms)), capital
    pending = ("buy", w * capital, np.zeros(len(syms)))
    values = []
    for k in range(first, last + 1):
        if pending is not None:
            _, buys, sells = pending
            fill = k + shift if k + shift <= last else k
            cash += float(np.sum(sells * o[fill]))
            units -= sells
            need = float(np.sum(buys))
            scale = 1.0 if need <= cash or need == 0 else cash / need
            units += buys * scale / o[fill]
            cash -= need * scale
            pending = None
        value = cash + float(np.sum(units * c[k]))
        values.append(value)
        if k < last and rule == "monthly" and market.dates[k].month != market.dates[k + 1].month:
            delta = w * value - units * c[k]
            pending = ("rebalance", np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta / c[k], 0.0))
    return np.array(values)


def c9_reference(market: DailyMarket, *, shift: int = 0) -> Check:
    problems: list[str] = []
    worst = 0.0
    first, last = _window("full", market)
    for mix in ("M2", "M3"):
        for rule in ("none", "monthly"):
            cfg = G.Config("core", "full", mix, rule, "central", D("100000"))
            spec = cfg.spec()
            spec = portfolio.Spec(spec.name, spec.symbols, spec.weights, spec.rule, capital=spec.capital,
                                  min_order_usd=D(0))
            days = portfolio.simulate(market, spec, first, last, FRICTIONLESS, FeeTable.load(), NO_BREAKER)
            weights = {s: float(v) for s, v in spec.weights(None).items()}
            ref = reference_values(market, weights, first, last, rule, 100000.0, shift=shift)
            got = np.array([float(d.value) for d in days])
            rel = float(np.max(np.abs(got / ref - 1)))
            worst = max(worst, rel)
            if rel > 1e-10:
                problems.append(f"{mix} {rule}: relative difference {rel:.2e}")
    return Check("C9", not problems, {"worst_relative_difference": worst, "problems": problems})


# --- C10 the registration lock ----------------------------------------------------------------------------------

def c10_registration_lock(*, guard: bool = True) -> Check:
    """``run-core`` refuses without a registration, with changed parameters, or without a passing
    ``leakage-core``. ``guard=False``: the twin, the guards removed."""
    from backtest import core_runs
    from backtest.store import Store

    refused: dict[str, bool] = {}
    with Store() as store:
        def attempt(label: str, registration_id: str) -> None:
            try:
                if guard:
                    core_runs.registered(store, registration_id)
                    core_runs.passing_leakage(store, "0" * 40)
                refused[label] = False
            except (PermissionError, KeyError):
                refused[label] = True

        attempt("no registration", "x-missing")
        store.register({"experiment_id": "x-reg", "created_at": datetime.now(UTC), "hypothesis": "h",
                        "command": core_runs.REGISTRATION, "parent_id": None, "status": "running",
                        "window_start": None, "window_end": None, "lockbox": False, "git_commit": "0" * 40,
                        "git_dirty": False, "data_hash": "-", "manifest": {}, "params": {
                            "registration": {**G.registration_params(), "tolerance": {"max_drawdown": "-0.99"}}},
                        "seeds": {}, "environment": {}})
        store.finish("x-reg", status="done", metrics={}, report_hash=None, conclusion="c",
                     finished_at=datetime.now(UTC))
        attempt("changed parameters", "x-reg")
        store.register({"experiment_id": "x-reg2", "created_at": datetime.now(UTC), "hypothesis": "h",
                        "command": core_runs.REGISTRATION, "parent_id": None, "status": "running",
                        "window_start": None, "window_end": None, "lockbox": False, "git_commit": "0" * 40,
                        "git_dirty": False, "data_hash": "-", "manifest": {},
                        "params": {"registration": G.registration_params()}, "seeds": {}, "environment": {}})
        store.finish("x-reg2", status="done", metrics={}, report_hash=None, conclusion="c",
                     finished_at=datetime.now(UTC))
        attempt("no passing leakage-core", "x-reg2")
    return Check("C10", all(refused.values()), {"refused": refused})


# --- C11 data completeness ------------------------------------------------------------------------------------------

def c11_completeness(market: DailyMarket, paths: CorePaths, *, forward_fill_gap: bool = False) -> Check:
    """Every instrument has a raw-file bar for every session from its warm-up to the window's end, and the
    market serves exactly the file's prices for it: nothing missing, nothing filled. ``forward_fill_gap``
    (the twin): one session's bar is replaced by the previous session's, as a forward-filling loader would."""
    from backtest import parquet
    file_bars = {(r["symbol"], r["t"].astimezone(NEW_YORK).date()): (r["o"], r["c"])
                 for r in parquet.read_rows(paths.daily("all"))}
    served = market
    if forward_fill_gap:
        gap = market.index(G.WINDOWS["full"][0]) + 100
        served = copy.copy(market)
        served._open = dict(market._open)
        served._close = dict(market._close)
        served._open["VTI"] = list(market._open["VTI"])
        served._close["VTI"] = list(market._close["VTI"])
        served._open["VTI"][gap] = market._open["VTI"][gap - 1]
        served._close["VTI"][gap] = market._close["VTI"][gap - 1]
    first, last = _window("full", market)
    problems: list[str] = []
    for s in ETFS:
        start = max(first - G.LOOKBACK - 1, served.first_index(s))
        for k in range(start, last + 1):
            bar = file_bars.get((s, market.dates[k]))
            if bar is None:
                problems.append(f"{s} {market.dates[k]}: no bar in the raw file")
            elif not served.has(s, k):
                problems.append(f"{s} {market.dates[k]}: in the file, not served")
            elif (served._open[s][k], served._close[s][k]) != bar:
                problems.append(f"{s} {market.dates[k]}: served {served._close[s][k]}, the file has {bar[1]}")
    b_first, _ = _window("crypto", market)
    missing_btc = [market.dates[k].isoformat() for k in range(b_first - G.LOOKBACK - 1, last + 1)
                   if not market.has(BTC, k)]
    problems += [f"BTC/USD {d}: no sample" for d in missing_btc]
    return Check("C11", not problems, {"instruments": len(ETFS) + 1, "problems": problems[:10]})


# --- C12 one implementation ---------------------------------------------------------------------------------------

def c12_one_implementation(*, overrides: dict[str, str] | None = None) -> Check:
    """The engine calls ``core_alloc``'s own functions; no other module defines a copy of them; tier 0's
    constants equal their source (phase 1's L10). ``overrides``: module path → source, for the twin."""
    names = {n for n, obj in vars(core_alloc).items() if inspect.isfunction(obj) and obj.__module__ == "core_alloc"}
    problems: list[str] = []
    if portfolio.core_alloc is not core_alloc:
        problems.append("portfolio does not import core_alloc")
    sources = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8")
               for p in [*ROOT.glob("*.py"), *ROOT.glob("backtest/*.py"), *ROOT.glob("risk_router/*.py"),
                         *ROOT.glob("swarm/*.py")] if p.name != "core_alloc.py"}
    sources.update(overrides or {})
    for path, text in sorted(sources.items()):
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                problems.append(f"{path} defines its own {node.name}")
    l10 = leakage.one_engine()
    if not l10.passed:
        problems += [f"L10: {p}" for p in l10.detail["problems"]]
    return Check("C12", not problems, {"functions": len(names), "files": len(sources), "problems": problems[:10]})


# --- all of them -------------------------------------------------------------------------------------------------

def run_all(root: Path, store: Any, params: dict[str, Any]) -> list[Check]:
    paths = CorePaths.under(root / ".cache" / "backtest")
    market = DailyMarket.load(paths, [*ETFS, BTC], last=G.WINDOWS["full"][1])
    seed = params["seed"]
    grid = G.grid()
    checks = [
        c1_m4_canary(market, instants=params["c1_instants"], seed=seed),
        twin("C1 twin: the window includes t+1", c1_m4_canary(market, instants=20, seed=seed, leaky=True)),
        c2_pit_oracle(market, instants=params["c2_instants"], seed=seed),
        twin("C2 twin: a bar served at its close", c2_pit_oracle(market, instants=2000, seed=seed,
                                                                 delay=timedelta(0))),
        c3_execution_lag(market),
        twin("C3 twin: fills at the decision close", c3_execution_lag(market, fill_at_decision_close=True)),
        c4_band_trigger(market),
        twin("C4 twin: the trigger reads t+1's open", c4_band_trigger(market, trigger_reads_next_open=True)),
        c5_adjustment_invariance(market, seed=seed),
        twin("C5 twin: volatility on price differences", c5_adjustment_invariance(market, seed=seed,
                                                                                  price_differences=True)),
        c6_crypto_cutoff(market, paths),
        twin("C6 twin: the bar after the close", c6_crypto_cutoff(market, paths, sampler="bar_after_close")),
        twin("C6 twin: a fixed 21:00 UTC cutoff", c6_crypto_cutoff(market, paths, sampler="utc_2100")),
        c7_phase1(root, store, params["phase1_run"]),
        twin("C7 twin: cash-safe sizing", _c7_cash_safe(root)),
        twin("C7 twin: daily fee rounding", c7_phase1(root, store, None, fee_rounding="daily")),
        c8_accounting(market, grid),
        twin("C8 twin: buys not capped by cash", c8_accounting(market, list(C3_CONFIGS), uncapped_buys=True)),
        c9_reference(market),
        twin("C9 twin: the reference shifted one day", c9_reference(market, shift=1)),
        c10_registration_lock(),
        twin("C10 twin: the guards removed", c10_registration_lock(guard=False)),
        c11_completeness(market, paths),
        twin("C11 twin: a gap forward-filled", c11_completeness(market, paths, forward_fill_gap=True)),
        c12_one_implementation(),
        twin("C12 twin: a copied function", c12_one_implementation(overrides={
            "backtest/portfolio.py": (ROOT / "backtest" / "portfolio.py").read_text(encoding="utf-8")
            + "\n\ndef plan_rebalance(*args, **kwargs):\n    return None\n"})),
    ]
    return checks


def _c7_cash_safe(root: Path) -> Check:
    """C7 with PHASE1's sizing replaced by the cash-safe CORE sizing (fees and multiplier as phase 1)."""
    cash_safe = portfolio.Profile("PHASE1-cash-safe", "core", "per_order", True, False, False, False)
    return c7_phase1(root, None, None, profile=cash_safe)
