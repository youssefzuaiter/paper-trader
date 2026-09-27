"""Leakage and consistency checks (design §9), each with a broken twin.

Every check returns a ``Check`` verdict instead of asserting, so the same
function runs twice: on the real configuration, where it must pass, and on
a *broken twin* that violates its rule on purpose, where it must fail. Both
stay in the test suite, so each check's power to catch its violation is
re-established on every run. The CLI runs the checks that need real data
and records them in an experiment.

L2's strategy half (S2 against the B3 placebo band) and L3's full-run half
need S2 and S3 replays; they are implemented here but, per the owner's
instruction, run on real data only after the D3 size and D5 criteria are
registered.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.metrics import roc_auc_score

import tier0
from backtest import walkforward
from backtest.calendar import Calendar
from backtest.costs import FRICTIONLESS, LEVELS, FeeTable
from backtest.data import (
    INTRADAY_CACHE,
    BarSeries,
    LookAheadError,
    MarketData,
    load_bar_cache,
    regular_series,
)
from backtest.dataset import Samples, build_legacy
from backtest.engine import Engine, EngineParams, PendingSignal, RunResult, Strategy
from backtest.lockbox import DEV_EVENTS_END, LOCKBOX_SESSIONS_FROM, LockBoxError
from backtest.metrics import bootstrap_auc
from backtest.sim_broker import FillTimingError, SimAlpaca
from backtest.store import Store
from backtest.walkforward import CostFn, FoldResult, Schedule, s2_nights, threshold_rule
from risk_router.schemas import TradeSignal
from swarm.alpaca_data import Bar
from swarm.common import NEW_YORK
from swarm.features import DAILY_FEATURES, SESSION_PUBLISHED_AT, completed_sessions, daily_features
from swarm.train_return_model import PURGE, RegularBars, train_and_evaluate

CANARY_DELAY: Final[timedelta] = timedelta(days=30)
ROOT: Final[Path] = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: dict[str, Any]


def pooled_auc(samples: Samples, folds: Sequence[FoldResult]) -> float:
    idx = np.concatenate([f.scored for f in folds])
    return float(roc_auc_score(samples.y[idx], np.concatenate([f.prob_scored for f in folds])))


# --- L1 future canary (P1-P3) -----------------------------------------------------------------------------

def canary_column(samples: Samples, *, join: str) -> np.ndarray:
    """The sample's own label, published 30 days after its session. ``asof``
    joins it like any input (the latest canary known at ``made_at``, for the
    same symbol); ``session`` — the broken twin — joins it on its own session."""
    if join == "session":
        return samples.y.astype(float)
    if join != "asof":
        raise ValueError(join)
    known = samples.resolved_at + CANARY_DELAY.total_seconds()
    out = np.full(len(samples), np.nan)
    for symbol in np.unique(samples.symbol):
        idx = np.flatnonzero(samples.symbol == symbol)
        order = idx[np.argsort(known[idx], kind="stable")]
        k = np.searchsorted(known[order], samples.made_at[idx], side="right") - 1
        out[idx[k >= 0]] = samples.y[order[k[k >= 0]]]
    return out


def future_canary(samples: Samples, schedule: Schedule, calendar: Calendar, cost_fn: CostFn, *, join: str = "asof",
                  base_folds: Sequence[FoldResult] | None = None, seed: int = 0) -> Check:
    base_folds = base_folds if base_folds is not None else walkforward.run(samples, schedule, calendar, cost_fn)
    augmented = replace(samples, X=np.column_stack([samples.X, canary_column(samples, join=join)]))
    folds = walkforward.run(augmented, schedule, calendar, cost_fn)
    base, with_canary = pooled_auc(samples, base_folds), pooled_auc(augmented, folds)
    column = samples.X.shape[1]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([f.scored for f in folds])
    drops = []
    for _ in range(3):
        probs = []
        for f in folds:
            X = augmented.X[f.scored].copy()
            X[:, column] = rng.permutation(X[:, column])
            probs.append(f.model.predict_proba(X)[:, 1])
        drops.append(with_canary - roc_auc_score(samples.y[idx], np.concatenate(probs)))
    importance = float(np.mean(drops))
    return Check("L1", abs(with_canary - base) < 0.01 and importance < 0.005,
                 {"join": join, "auc_without": base, "auc_with_canary": with_canary, "canary_importance": importance})


# --- L2 label shuffling (P6) ------------------------------------------------------------------------------------

def shuffle_labels(samples: Samples, seed: int) -> Samples:
    """Labels permuted in symbol-day blocks: each block takes another block's
    labels (cycled or cut to its size), so a day's shared label stays shared."""
    rng = np.random.default_rng(seed)
    day = samples.session if (samples.session >= 0).all() else samples.day
    blocks: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(zip(samples.symbol.astype(str), day.astype(str), strict=True)):
        blocks.setdefault(key, []).append(i)
    names = sorted(blocks)
    y, fwd = samples.y.copy(), samples.fwd.copy()
    for dst, j in zip(names, rng.permutation(len(names)), strict=True):
        target, source = blocks[dst], blocks[names[j]]
        take = [source[k % len(source)] for k in range(len(target))]
        y[target], fwd[target] = samples.y[take], samples.fwd[take]
    return replace(samples, y=y, fwd=fwd, idealised_fwd=fwd, exit_price=samples.entry_price * (1 + fwd))


def shuffled_auc(samples: Samples, schedule: Schedule, calendar: Calendar, cost_fn: CostFn, *,
                 seeds: Sequence[int], bootstrap: int = 1000) -> Check:
    """With labels shuffled, the walk-forward's AUC interval must contain 0.5."""
    runs = []
    for seed in seeds:
        shuffled = shuffle_labels(samples, seed)
        runs.append((seed, shuffled, walkforward.run(shuffled, schedule, calendar, cost_fn)))
    return auc_contains_half(runs, bootstrap=bootstrap)


def auc_contains_half(runs: Sequence[tuple[int, Samples, Sequence[FoldResult]]], *, bootstrap: int = 1000) -> Check:
    """At 95%, one seed in twenty misses by chance: at most 2 of 10 may, and the
    seeds' mean AUC must be within 0.01 of 0.5."""
    per_seed = []
    for seed, shuffled, folds in runs:
        idx = np.concatenate([f.scored for f in folds])
        prob = np.concatenate([f.prob_scored for f in folds])
        interval = bootstrap_auc(shuffled.y[idx], prob, shuffled.session[idx], n=bootstrap, seed=seed)
        per_seed.append({"seed": seed, **interval.as_dict(), "contains_half": interval.contains(0.5)})
    misses = sum(not s["contains_half"] for s in per_seed)
    mean = float(np.mean([s["estimate"] for s in per_seed]))
    allowed = max(1, round(0.2 * len(runs))) if len(runs) >= 5 else 0
    return Check("L2-auc", misses <= allowed and abs(mean - 0.5) < 0.01,
                 {"seeds": per_seed, "misses": misses, "allowed": allowed, "mean_auc": mean})


def leaky_folds(samples: Samples, folds: Sequence[FoldResult], cost_fn: CostFn) -> list[FoldResult]:
    """The L2 twin: S2's threshold chosen on each fold's *scored* month (P6 broken)."""
    out = []
    for f in folds:
        units = s2_nights(samples, f.scored, f.prob_scored, collapsed=False)
        chosen = None
        if len(units["score"]):
            chosen, _ = threshold_rule(units, cost_fn(samples, units["first"]), per_symbol_day_first=False)
        out.append(replace(f, thresholds={**f.thresholds, "S2": chosen}))
    return out


def s2_screen(samples: Samples, folds: Sequence[FoldResult], night_returns: Mapping[tuple[str, int], float], *,
              placebo_seeds: int = 100, seed: int = 0) -> Check:
    """S2's selection against its placebo, vectorised: screening only, never a
    result. Per scored session S2 takes its symbol-nights at or above the
    fold's threshold; each placebo draw takes as many random symbols that
    session. Returns are gross (09:32 open to close): the same cost on both
    sides cancels. S2 must sit inside the band, below its 95th percentile."""
    picked: list[float] = []
    counts: dict[int, int] = {}
    for f in folds:
        threshold = f.thresholds.get("S2")
        units = s2_nights(samples, f.scored, f.prob_scored, collapsed=False)
        if threshold is None or not len(units["score"]):
            continue
        keep = np.flatnonzero(units["score"] >= threshold)
        picked.extend(units["ret"][keep].tolist())
        for s in units["session"][keep]:
            counts[int(s)] = counts.get(int(s), 0) + 1
    symbols = sorted({k[0] for k in night_returns})
    rng = np.random.default_rng(seed)
    placebo = []
    for _ in range(placebo_seeds):
        draws = [night_returns.get((str(sym), s), np.nan)
                 for s, k in sorted(counts.items()) for sym in rng.choice(symbols, size=min(k, len(symbols)),
                                                                          replace=False)]
        placebo.append(float(np.nanmean(draws)) if draws else 0.0)
    mean = float(np.mean(picked)) if picked else 0.0
    upper = float(np.quantile(placebo, 0.95)) if placebo else 0.0
    return Check("L2-screen", mean <= upper, {"s2_mean": mean, "placebo_p95": upper, "s2_units": len(picked),
                                              "placebo_seeds": placebo_seeds})


def night_returns(market: MarketData, sessions: Sequence[int], offset_bars: int = 2) -> dict[tuple[str, int], float]:
    """Every symbol's return from the 09:32 open to the close, per session index."""
    out = {}
    for symbol, series in market.minute.items():
        for s in sessions:
            lo = int(np.searchsorted(series.session, s, side="left"))
            hi = int(np.searchsorted(series.session, s, side="right")) - 1
            if hi - lo >= offset_bars:
                out[(symbol, s)] = float(series.c[hi] / series.o[lo + offset_bars] - 1)
    return out


def s2_inside_placebo(s2: RunResult, placebo: Sequence[RunResult]) -> Check:
    """The engine version of L2's strategy half: S2's total P&L below the 95th
    percentile of its B3 placebo runs (run after D3/D5 are registered)."""
    total = float(sum(d.pnl for d in s2.days))
    band = [float(sum(d.pnl for d in r.days)) for r in placebo]
    upper = float(np.quantile(band, 0.95))
    return Check("L2-engine", total <= upper, {"s2_total_pnl": total, "placebo_p95": upper, "runs": len(band)})


# --- L3 no bar after the decision (P2-P4) ------------------------------------------------------------------------

def pit_oracle(market: MarketData, *, n: int = 10_000, seed: int = 0) -> Check:
    """``PitView`` and the fill index against brute force at random instants:
    in session, on exact minute boundaries (±1 µs) and overnight."""
    rng = np.random.default_rng(seed)
    with_bars = set().union(*(np.unique(series.session).tolist() for series in market.minute.values()))
    sessions = [market.calendar.sessions[i] for i in sorted(with_bars)]
    starts = {sym: {int(t): k for k, t in enumerate(series.start)} for sym, series in market.minute.items()}
    bad: list[str] = []
    for k in range(n):
        symbol = str(rng.choice(market.symbols))
        session = sessions[int(rng.integers(len(sessions)))]
        kind = k % 3
        if kind == 0:
            t = session.open_at + timedelta(seconds=float(rng.uniform(0, (session.close_at - session.open_at)
                                                                     .total_seconds())))
        elif kind == 1:
            minute = session.open_at + timedelta(minutes=int(rng.integers(0, session.minutes + 1)))
            t = minute + timedelta(microseconds=int(rng.choice([-1, 0, 1])))
        else:
            t = session.close_at + timedelta(seconds=float(rng.uniform(0, 17 * 3600)))
        view = market.as_of(t)
        series = market.minute[symbol]
        got = view.quote_bar(symbol)
        want = _brute_last_ended(series, starts[symbol], t)
        if (got is None) != (want is None) or (got is not None and int(got.start.timestamp()) != want):
            bad.append(f"quote {symbol} {t.isoformat()}")
        if got is not None and got.end > t:
            bad.append(f"quote after {t.isoformat()}")
        if view.daily(symbol) != completed_sessions(market.daily[symbol], t):
            bad.append(f"daily {symbol} {t.isoformat()}")
        fill_from = t + timedelta(minutes=1)
        i = market.fill_index(symbol, fill_from)
        if i < len(series) and (series.start[i] < fill_from.timestamp()
                                or (i > 0 and series.start[i - 1] >= fill_from.timestamp())):
            bad.append(f"fill {symbol} {t.isoformat()}")
    return Check("L3-oracle", not bad, {"instants": n, "mismatches": bad[:10], "n_mismatches": len(bad)})


def _brute_last_ended(series: BarSeries, index: Mapping[int, int], t: datetime) -> int | None:
    """Walk back minute by minute from the last whole minute: independent of any bisect."""
    minute = int(t.timestamp()) // 60 * 60 - series.seconds
    first = int(series.start[0]) if len(series) else 0
    while minute >= first:
        if minute in index and minute + series.seconds <= t.timestamp():
            return minute
        minute -= 60
    return None


def assertions_silent(run: Callable[[], Any]) -> Check:
    """A replay in which neither ``PitView`` nor the broker's P4 assertion fires."""
    try:
        run()
    except (LookAheadError, FillTimingError) as exc:
        return Check("L3-run", False, {"assertion": f"{type(exc).__name__}: {exc}"})
    return Check("L3-run", True, {})


# --- L4 reproduce report.md ---------------------------------------------------------------------------------------

def _normalise(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _differences(ours: Any, theirs: Any, path: str = "") -> list[str]:
    if isinstance(ours, dict) and isinstance(theirs, dict):
        out = [f"{path}/{k}: missing" for k in set(ours) ^ set(theirs)]
        for k in sorted(set(ours) & set(theirs)):
            out.extend(_differences(ours[k], theirs[k], f"{path}/{k}"))
        return out
    if isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            return [f"{path}: {len(ours)} rows vs {len(theirs)}"]
        return [d for k, (a, b) in enumerate(zip(ours, theirs, strict=True)) for d in _differences(a, b, f"{path}[{k}]")]
    return [] if ours == theirs else [f"{path}: {ours!r} != {theirs!r}"]


def _bisect_right_labels(samples: Samples, intraday: Mapping[str, list[Bar]]) -> Samples:
    """The L4 twin's label: an article stamped exactly on a bar start skips that bar."""
    regular = {s: RegularBars.build(b) for s, b in intraday.items()}
    fwd = samples.fwd.copy()
    for i in range(len(samples)):
        bars = regular[str(samples.symbol[i])]
        k = bisect_right(bars.starts, datetime.fromtimestamp(samples.published_at[i], UTC))
        if k < len(bars.starts):
            fwd[i] = float(bars.closes[bars.last_index[bars.session_of[k]]]) / float(bars.opens[k]) - 1
    return replace(samples, fwd=fwd, y=(fwd > 0.0025).astype(int))


def reproduce_report(meta: Mapping[str, Any], watch: Sequence[str], training_cache: Path, *,
                     purge: timedelta = PURGE, entry: str = "bisect_left") -> tuple[Check, dict[str, Any]]:
    """``report.md``'s numbers, recomputed in legacy mode. Returns the check and
    the recomputed metrics (for the reproduction report)."""
    samples = build_legacy(watch, training_cache)
    if entry == "bisect_right":
        intraday = {s: [b for b in bars if b.t.astimezone(NEW_YORK).date() < LOCKBOX_SESSIONS_FROM]
                    for s, bars in load_bar_cache(training_cache / INTRADAY_CACHE, watch).items()}
        samples = _bisect_right_labels(samples, intraday)
    elif entry != "bisect_left":
        raise ValueError(entry)
    _, metrics = train_and_evaluate(samples.to_legacy_dataset(), purge=purge)
    move = metrics.pop("move_table")
    ours = {"metrics": _normalise(metrics), "move_table": _normalise(move), "n_samples": len(samples),
            "digest": samples.digest()[:8]}
    theirs = {"metrics": meta["metrics"], "move_table": meta["move_table"], "n_samples": meta["n_samples"],
              "digest": meta["version"].rsplit("-", 1)[1]}
    diffs = _differences(ours, theirs)
    return Check("L4", not diffs, {"purge_s": purge.total_seconds(), "entry": entry, "differences": diffs[:20],
                                   "n_differences": len(diffs)}), ours


# --- L4b the engine agrees with the labels -----------------------------------------------------------------------

def legacy_series(intraday: Mapping[str, list[Bar]], calendar: Calendar) -> dict[str, BarSeries]:
    """``RegularBars``' own bars (weekday 09:30-15:30 starts, New York) as
    series the simulated broker can fill on; a session is the bar's date."""
    out = {}
    for symbol, bars in intraday.items():
        regular = RegularBars.build(bars)
        start = np.array([int(t.timestamp()) for t in regular.starts], dtype=np.int64)
        o, c = regular.opens.astype(float), regular.closes.astype(float)
        session = np.array([calendar.index[d] for d in regular.session_of], dtype=np.int32)
        out[symbol] = BarSeries(symbol, 1800, start, o, np.maximum(o, c), np.minimum(o, c), c, np.zeros(len(o)),
                                session)
    return out


class _ContainingBarSim(SimAlpaca):
    """The L4b twin: fills on the bar *containing* the decision, not the next one."""

    def _first_bar(self, series: BarSeries, eligible_epoch: float) -> int:
        return max(0, int(np.searchsorted(series.start, eligible_epoch, side="right")) - 1)


def engine_agrees_with_labels(samples: Samples, market: MarketData, series: Mapping[str, BarSeries], *,
                              twin: bool = False, tolerance: float = 1e-12) -> Check:
    """Each legacy sample replayed through the broker as a frictionless trade
    (a market buy at its publication, a market-on-close sell) must return its label's ``fwd``."""
    sim_class = _ContainingBarSim if twin else SimAlpaca
    sim = sim_class(market, FRICTIONLESS, FeeTable.load(), latency_bars=0, bars=series, legacy=True)
    for i in range(len(samples)):
        t = datetime.fromtimestamp(float(samples.published_at[i]), UTC)
        sim.advance(t)
        symbol = str(samples.symbol[i])
        for side, tif in (("buy", "day"), ("sell", "cls")):
            status, body = sim.submit({"symbol": symbol, "qty": "1", "side": side, "type": "market",
                                       "time_in_force": tif, "client_order_id": f"{side}-{i}"})
            if status != 200:
                raise RuntimeError(f"legacy replay order refused: {body}")
    last = max(int(s.start[-1]) + s.seconds for s in series.values())
    sim.advance(datetime.fromtimestamp(last, UTC))
    errors = np.empty(len(samples))
    unfilled = 0
    for i in range(len(samples)):
        buy, sell = sim._by_client_id[f"buy-{i}"], sim._by_client_id[f"sell-{i}"]
        if buy.filled_avg_price is None or sell.filled_avg_price is None:
            unfilled += 1
            errors[i] = np.inf
            continue
        errors[i] = abs(float(sell.filled_avg_price / buy.filled_avg_price) - 1 - float(samples.fwd[i]))
    worst = float(errors.max()) if len(errors) else 0.0
    return Check("L4b", worst <= tolerance and unfilled == 0,
                 {"samples": len(samples), "max_abs_error": worst, "unfilled": unfilled,
                  "over_tolerance": int((errors > tolerance).sum()), "twin": twin})


# --- L5 training cut-off (P5) and L6 story straddling ----------------------------------------------------------------

def training_cutoff(samples: Samples, folds: Sequence[FoldResult], calendar: Calendar, *,
                    embargo_sessions: int = 5) -> Check:
    bad = []
    for f in folds:
        used = np.concatenate([f.train, f.calib])
        latest = datetime.fromtimestamp(float(samples.resolved_at[used].max()), UTC)
        first = calendar.next_session(f.bounds.start)
        gap = sum(1 for s in calendar.sessions if latest < s.open_at < first.open_at)
        scored_min = float(samples.made_at[f.scored].min()) if len(f.scored) else np.inf
        if gap < embargo_sessions or latest >= f.bounds.start or scored_min < f.bounds.start.timestamp():
            bad.append({"month": f.bounds.month.isoformat(), "latest_label": latest.isoformat(), "sessions_between": gap})
    return Check("L5", not bad, {"folds": len(folds), "violations": bad[:10]})


def story_straddle(samples: Samples, folds: Sequence[FoldResult]) -> Check:
    bad = []
    for f in folds:
        trained = {s for s in samples.story_id[np.concatenate([f.train, f.calib])] if s}
        scored = {s for s in samples.story_id[f.scored] if s}
        shared = trained & scored
        if shared:
            bad.append({"month": f.bounds.month.isoformat(), "stories": sorted(shared)[:5], "n": len(shared)})
    return Check("L6", not bad, {"folds": len(folds), "straddles": bad[:10]})


# --- L7 lock-box ----------------------------------------------------------------------------------------------------------

class UnguardedMarketData(MarketData):
    """The L7 twin: the lock-box guard switched off."""

    def _guarded(self) -> bool:
        return False


def lockbox_guard(market: MarketData, store: Store | None = None) -> Check:
    leaks = []
    session = market.calendar.session(LOCKBOX_SESSIONS_FROM)
    if session is not None:
        try:
            market.as_of(session.open_at + timedelta(hours=1))
            leaks.append("PitView opened inside the lock-box")
        except LockBoxError:
            pass
        first_locked = market.calendar.index[LOCKBOX_SESSIONS_FROM]
        for symbol, series in market.minute.items():
            if len(series) and int(series.session.max()) >= first_locked:
                leaks.append(f"{symbol}: lock-box minute bars in memory")
    for symbol, bars in market.daily.items():
        if bars and bars[-1].t.astimezone(NEW_YORK).date() >= LOCKBOX_SESSIONS_FROM:
            leaks.append(f"{symbol}: lock-box daily bars in memory")
    if store is not None and any(e["known_at"] >= DEV_EVENTS_END for e in store.events()):
        leaks.append("store served lock-box events")
    return Check("L7", not leaks, {"leaks": leaks})


# --- L8 feature parity -------------------------------------------------------------------------------------------------

def feature_parity(predictions: Sequence[Mapping[str, Any]], market: MarketData, *, n: int = 1000, seed: int = 0,
                   late: bool = False) -> Check:
    """Stored inputs against ``daily_features(completed_sessions(bars, made_at))``
    recomputed independently. The twin stores features cut off one session late."""
    # Sampled and reported by (event, symbol, time), never by prediction_id: prediction ids include
    # the experiment, so an id-ordered sample would differ between a run and its reproduction.
    ordered = sorted(predictions, key=lambda r: (str(r["event_id"]), str(r["symbol"]), r["made_at"]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ordered), size=min(n, len(ordered)), replace=False)
    bad = []
    for k in pick:
        p = ordered[int(k)]
        inputs = json.loads(p["inputs"]) if isinstance(p["inputs"], str) else p["inputs"]
        bars = market.daily[p["symbol"]]
        expected = daily_features(completed_sessions(bars, p["made_at"]))
        if late:
            upcoming = market.calendar.next_session(p["made_at"]) or market.calendar.sessions[-1]
            cut = datetime.combine(upcoming.date, SESSION_PUBLISHED_AT, NEW_YORK)
            inputs = daily_features(completed_sessions(bars, cut))
        if any(inputs[name] != expected[name] for name in DAILY_FEATURES):
            bad.append(f"{p['event_id']}|{p['symbol']}|{p['made_at'].isoformat()}")
    return Check("L8", not bad, {"checked": len(pick), "mismatches": len(bad), "examples": bad[:5], "late": late})


# --- L9 outcome isolation ----------------------------------------------------------------------------------------------

ISOLATED: Final[tuple[str, ...]] = ("backtest.dataset", "backtest.walkforward", "backtest.strategies")


def _imports(module: str, source: str) -> set[str]:
    out: set[str] = set()
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1] + ([base] if base else []))
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return {m for m in out if m.startswith("backtest")}


def outcome_isolation(modules: Sequence[str] = ISOLATED, *, overrides: Mapping[str, str] | None = None) -> Check:
    """Nothing that builds features, fits models or chooses parameters reaches
    ``backtest.outcomes``, directly or through another backtest module."""
    overrides = overrides or {}
    seen: set[str] = set()
    todo = list(modules)
    while todo:
        name = todo.pop()
        if name in seen or name == "backtest":
            continue
        seen.add(name)
        path = ROOT / (name.replace(".", "/") + ".py")
        if name not in overrides and not path.exists():
            continue
        source = overrides.get(name) or path.read_text(encoding="utf-8")
        todo.extend(_imports(name, source) - seen)
    return Check("L9", "backtest.outcomes" not in seen, {"reached": sorted(seen)})


# --- L10 one engine ------------------------------------------------------------------------------------------------------

def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        a, b = _literal(node.left), _literal(node.right)
        return a * b if isinstance(node.op, ast.Mult) else a + b if isinstance(node.op, ast.Add) else a - b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_literal(a) for a in node.args]
        kwargs = {k.arg: _literal(k.value) for k in node.keywords}
        if node.func.id == "Decimal":
            return Decimal(*args)
        if node.func.id == "timedelta":
            return timedelta(*args, **kwargs)
    raise ValueError(ast.dump(node))


def tier0_source_constants() -> dict[str, Any]:
    """Every ``Final`` risk constant as written in tier0.py."""
    out = {}
    for node in ast.parse(Path(tier0.__file__).read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None
                and "Final" in ast.unparse(node.annotation)):
            out[node.target.id] = _literal(node.value)
    return out


def source_hashes() -> dict[str, str]:
    files = ("tier0.py", "risk_router/policy.py", "risk_router/gatekeeper.py", "risk_router/guards.py",
             "risk_router/alpaca_async.py", "swarm/features.py", "swarm/return_model.py", "swarm/quant.py")
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def one_engine() -> Check:
    """The router, its policy, sizing and exit rule are the production objects,
    every tier-0 constant still equals its source, and production wiring gates on tier 0."""
    from backtest import engine
    from risk_router import app, gatekeeper, policy

    problems = []
    if engine.RiskRouter is not gatekeeper.RiskRouter:
        problems.append("the engine's RiskRouter is not the production class")
    if gatekeeper.check_can_open is not policy.check_can_open:
        problems.append("the router's check_can_open is not policy.check_can_open")
    if gatekeeper.tier0 is not tier0 or inspect.getsourcefile(tier0.plan_buy) != tier0.__file__:
        problems.append("plan_buy is not tier0's")
    if engine.exit_reason is not gatekeeper.exit_reason:
        problems.append("the engine's exit_reason is not the router's")
    for name, value in tier0_source_constants().items():
        if getattr(tier0, name) != value:
            problems.append(f"tier0.{name} is {getattr(tier0, name)!r} at run time, {value!r} in its source")
    if inspect.signature(gatekeeper.RiskRouter).parameters["min_prob_up"].default is not tier0.ROUTER_MIN_PROB_UP:
        problems.append("RiskRouter's default gate is not tier0.ROUTER_MIN_PROB_UP")
    wiring = ast.parse(inspect.getsource(app.build_router_from_env))
    for call in (n for n in ast.walk(wiring) if isinstance(n, ast.Call) and ast.unparse(n.func) == "RiskRouter"):
        if any(k.arg == "min_prob_up" for k in call.keywords):
            problems.append("production wiring passes min_prob_up")
    return Check("L10", not problems, {"problems": problems, "source_sha256": source_hashes()})


# --- L11 calendar --------------------------------------------------------------------------------------------------------

def calendar_rules(result: RunResult, truth: Calendar, *, half_day: date, late_signal: str) -> Check:
    """Against the *true* calendar: no fill outside a session's regular hours,
    the half-day signal after 12:30 refused, and the half-day exit decided at 12:50."""
    problems = []
    for row in result.rows:
        if row["state"] in {"filled", "exit_filled"}:
            start = datetime.fromisoformat(row["detail"]["bar_start"])
            if truth.session_at(start) is None:
                problems.append(f"fill outside a session at {start.isoformat()}")
    late = [r for r in result.rows if r["trade_id"] == late_signal and r["state"] in {"rejected", "submitted"}]
    if not late or late[0].get("reason_code") != "entry_window_closed":
        problems.append(f"half-day entry after 12:30 was {late[0]['state'] if late else 'never decided'}")
    session = truth.session(half_day)
    exits = [r["occurred_at"] for r in result.rows if r["state"] == "exit_submitted"
             and r["occurred_at"].astimezone(NEW_YORK).date() == half_day]
    if not exits or set(exits) != {session.close_at - tier0.SESSION_EXIT_BEFORE_CLOSE}:
        problems.append(f"half-day exits decided at {[e.isoformat() for e in exits]}")
    return Check("L11", not problems, {"problems": problems})


# --- L12 minute skipping -------------------------------------------------------------------------------------------------

class _NoIntradayExits(Engine):
    """The L12 twin: exit passes skipped until the session-close window, whatever the prices."""

    def _exits_needed(self, sim: SimAlpaca, t: datetime, closing_from: datetime) -> bool:
        return t >= closing_from


def skipping_invariance(market: MarketData, strategy: Strategy, sessions: Sequence[Any], *,
                        twin: bool = False) -> Check:
    level, fees = LEVELS["central"], FeeTable.load()
    full = asyncio.run(Engine(market, strategy, level, fees, EngineParams(skip_idle=False)).run(sessions))
    engine_class = _NoIntradayExits if twin else Engine
    skipped = asyncio.run(engine_class(market, strategy, level, fees, EngineParams(skip_idle=True)).run(sessions))
    same = full.rows == skipped.rows and [d.pnl for d in full.days] == [d.pnl for d in skipped.days]
    return Check("L12", same, {"rows": [len(full.rows), len(skipped.rows)],
                               "requests": [full.requests, skipped.requests], "twin": twin})


# --- the revision diagnostic (§9) ---------------------------------------------------------------------------------------

REVISION_BUCKETS: Final[tuple[tuple[str, float, float], ...]] = (
    ("never revised", -1.0, 0.5), ("within a minute", 0.5, 60.0), ("1 min to 1 h", 60.0, 3600.0),
    ("1 h to 1 day", 3600.0, 86400.0), ("over a day", 86400.0, float("inf")))


def revision_auc(samples: Samples, folds: Sequence[FoldResult], updated_at: Mapping[str, datetime | None]) -> list[
        dict[str, Any]]:
    """AUC of scored predictions by how long after publication the article was
    last revised. An edge concentrated in later-revised articles would mean
    the headline we hold may postdate ``known_at``."""
    idx = np.concatenate([f.scored for f in folds])
    prob = np.concatenate([f.prob_scored for f in folds])
    gap = np.array([(updated_at[str(samples.event_id[i])] - datetime.fromtimestamp(samples.published_at[i], UTC))
                    .total_seconds() if updated_at.get(str(samples.event_id[i])) else np.nan for i in idx])
    rows = []
    for name, lo, hi in REVISION_BUCKETS:
        mask = (gap >= lo) & (gap < hi)
        y = samples.y[idx][mask]
        auc = float(roc_auc_score(y, prob[mask])) if 0 < y.sum() < len(y) else None
        rows.append({"bucket": name, "n": int(mask.sum()), "auc": auc})
    return rows


# --- fixed-signal replays, for the engine checks (L3, L11, L12) --------------------------------------------------------

class FixedSignals:
    """Signals fixed in advance: for checking the engine, never a trading strategy."""

    name = "fixed-signals"

    def __init__(self, signals: Sequence[PendingSignal], min_prob_up: Decimal = Decimal("0.5")) -> None:
        self._signals = list(signals)
        self._min = min_prob_up

    def min_prob_up(self, session: Any) -> Decimal:
        return self._min

    def signals_for(self, session: Any, market: MarketData) -> list[PendingSignal]:
        return [p for p in self._signals if session.open_at <= p.emit_at <= session.close_at]


def fixed_signal(n: int, at: datetime, symbol: str = "AAPL", *, prob: float = 0.7,
                 created_at: datetime | None = None) -> PendingSignal:
    return PendingSignal(at, TradeSignal(signal_id=f"sig-{n:08d}", symbol=symbol, source="check", model_name="check",
                                         prob_up=prob, predicted_move_pct=0.5, confidence=0.5,
                                         created_at=created_at or at))


def path_market(path: Callable[[date, int], tuple[float, float, float, float]], calendar: Calendar,
                days: Sequence[date], daily: Mapping[str, list[Bar]], symbol: str = "AAPL") -> MarketData:
    """A one-symbol market whose 1-minute bars follow ``path(day, minute_of_session)``."""
    cols: dict[str, list[float]] = {k: [] for k in "tohlcv"}
    for s in calendar.sessions:
        if s.date not in days:
            continue
        for k in range(s.minutes):
            o, h, lo, c = path(s.date, k)
            for key, value in zip("tohlcv", ((s.open_at + timedelta(minutes=k)).timestamp(), o, h, lo, c, 1e4),
                                  strict=True):
                cols[key].append(value)
    a = {k: np.asarray(v) for k, v in cols.items()}
    series = regular_series(symbol, 60, a["t"].astype(np.int64), a["o"], a["h"], a["l"], a["c"], a["v"], calendar)
    return MarketData(calendar, {symbol: series}, {symbol: list(daily.get(symbol, []))}, {symbol: {}})


def fixture_month(calendar: Calendar, month: date) -> tuple[MarketData, FixedSignals, list[Any]]:
    """L12's fixture: a month of flat days, one entry at 10:00 each, one day
    touching the stop (-6% intrabar), another closing through the target (+10.7%)."""
    days = [s.date for s in calendar.sessions if s.date.year == month.year and s.date.month == month.month
            and not s.half_day]
    stop_day, target_day = days[len(days) // 3], days[2 * len(days) // 3]

    def path(day: date, k: int) -> tuple[float, float, float, float]:
        if day == stop_day and k == 100:
            return 100.0, 100.0, 94.0, 99.0
        if day == target_day and k >= 150:
            return 110.6, 110.8, 110.5, 110.7
        return 100.0, 100.02, 99.95, 100.0

    market = path_market(path, calendar, days, {})
    signals = [fixed_signal(n, calendar.session(d).open_at + timedelta(minutes=30)) for n, d in enumerate(days, 1)]
    return market, FixedSignals(signals), [calendar.session(d) for d in days]
