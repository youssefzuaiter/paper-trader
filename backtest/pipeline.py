"""The real runs (design §5, §6, §10; D3): the walk-forward at both order sizes, every
strategy at every cost level, and the owner's registered pass rule.

**Order sizes (D3).** Every replay trades Tier-0's $10 through the production router,
whose limits stay hardcoded (D2). The registered real-money size is reached by
scaling every dollar amount by k (100 for $1,000): the router's dollar caps scale
with it and its counts do not change, so every decision is the same, and fees and
cent rounding are recomputed at the scaled quantities with the level's own rounding
(D12). What does depend on size is the threshold rule's fee cost, so S2, S3 and B3
are replayed per size; B1 and B2 do not use thresholds and are replayed once per level.

Replays are independent and deterministic, so they run in a process pool; results
are gathered by key, never by completion order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import numpy as np

from backtest import costs, metrics
from backtest.costs import LEVELS, DailyFees, FeeTable
from backtest.data import MarketData
from backtest.dataset import Samples
from backtest.engine import Engine, EngineParams
from backtest.fetch import Paths
from backtest.strategies import B1, B2, B3, S2, S3, run_hold
from backtest.walkforward import FoldResult
from swarm.common import NEW_YORK

logger = logging.getLogger("backtest.pipeline")

TIER0: Final[Decimal] = Decimal("10")
CAPITAL_AT_TIER0: Final[Decimal] = Decimal("50")     # tier0.MAX_GROSS_EXPOSURE_USD
STRATEGY_KINDS: Final[tuple[str, ...]] = ("S2", "S2_collapsed", "S3", "S3_collapsed")


@dataclass(frozen=True)
class Plan:
    """Everything a run depends on, registered before it starts."""

    levels: tuple[str, ...] = ("optimistic", "central", "pessimistic")
    sizes: tuple[Decimal, ...] = (TIER0, Decimal("1000"))
    first_session: date = date(2025, 1, 2)
    last_session: date = date(2026, 9, 18)
    b1_seed: int = 1
    b3_seeds: tuple[int, ...] = tuple(range(100))
    latency_bars: int = 1
    resamples: int = metrics.RESAMPLES
    bootstrap_seed: int = 0
    s1_band: Decimal = Decimal("0.05")
    workers: int = 1              # execution only: replays are deterministic, results do not depend on it

    def as_params(self) -> dict[str, Any]:
        return {"levels": list(self.levels), "sizes_usd": [str(s) for s in self.sizes],
                "first_session": self.first_session.isoformat(), "last_session": self.last_session.isoformat(),
                "b1_seed": self.b1_seed, "b3_seeds": [self.b3_seeds[0], self.b3_seeds[-1], len(self.b3_seeds)],
                "latency_bars": self.latency_bars, "resamples": self.resamples,
                "bootstrap_seed": self.bootstrap_seed, "mean_block_days": metrics.MEAN_BLOCK,
                "s1_band": str(self.s1_band), "capital_base_at_tier0_usd": str(CAPITAL_AT_TIER0)}


def scale(size: Decimal) -> Decimal:
    return size / TIER0


# --- scaling a replay to another order size ------------------------------------------------------

def daily_pnl_at(rows: Sequence[Mapping[str, Any]], level: costs.CostLevel, fees: FeeTable,
                 k: Decimal) -> dict[date, Decimal]:
    """A replay's daily P&L with every quantity multiplied by ``k``: the same fills,
    with cash legs, fees and the day's fee rounding recomputed at the new size."""
    out: dict[date, Decimal] = defaultdict(Decimal)
    daily = DailyFees()
    for r in rows:
        if r["state"] not in {"filled", "exit_filled"}:
            continue
        day = datetime.fromisoformat(r["detail"]["bar_start"]).astimezone(NEW_YORK).date()
        qty, price, side = r["qty"] * k, r["price"], r["side"]
        cash = -costs.cash_debit(qty, price, level) if side == "buy" else costs.cash_credit(qty, price, level)
        charged = fees.order_fees(day, side, qty, price, level.fee_rounding)
        if level.fee_rounding == "daily":
            daily.add(day, charged)
        if level.fee_rounding == "none":
            charged = {}
        out[day] += cash - sum(charged.values(), Decimal(0))
    if level.fee_rounding == "daily":
        for day in list(out):
            out[day] -= daily.rounding_charge(day)
    return dict(out)


# --- replays in worker processes ---------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    strategy: str             # S2 | S2_collapsed | S3 | S3_collapsed | B1 | B2 | B3
    level: str
    size: Decimal | None      # the threshold size for S2/S3/B3; None for B1/B2
    seed: int | None = None
    entries_by_date: tuple[tuple[date, int], ...] = ()

    @property
    def key(self) -> str:
        size = "-" if self.size is None else f"${self.size}"
        seed = "" if self.seed is None else f":seed{self.seed}"
        return f"{self.strategy}:{self.level}:{size}{seed}"


_STATE: dict[str, Any] = {}


def _init_worker(root: str, symbols: tuple[str, ...], samples: Samples, folds: Mapping[str, list[FoldResult]],
                 prediction_ids: Mapping[tuple[str, str, str], str], dates: tuple[date, ...],
                 market: MarketData | None = None) -> None:
    logging.getLogger("risk_router").setLevel(logging.ERROR)
    if market is None:
        market = MarketData.load(Paths(Path(root) / ".cache" / "backtest"), symbols=symbols,
                                 training_cache=Path(root) / ".cache" / "training")
    _STATE.update(market=market, samples=samples, folds=folds, prediction_ids=prediction_ids,
                  sessions=[market.calendar.session(d) for d in dates], symbols=symbols, fees=FeeTable.load())


def _strategy(task: Task) -> Any:
    samples, folds = _STATE["samples"], _STATE["folds"]
    size_folds = folds[str(task.size)] if task.size is not None else folds[str(TIER0)]
    if task.strategy in {"S2", "S2_collapsed"}:
        return S2(samples, size_folds, collapsed=task.strategy.endswith("collapsed"), name=task.strategy)
    if task.strategy in {"S3", "S3_collapsed"}:
        return S3(samples, size_folds, _STATE["prediction_ids"], collapsed=task.strategy.endswith("collapsed"),
                  name=task.strategy)
    if task.strategy == "B1":
        return B1(_STATE["symbols"], task.seed)
    if task.strategy == "B2":
        return B2(S2(samples, size_folds, name="S2"))
    if task.strategy == "B3":
        return B3(_STATE["symbols"], dict(task.entries_by_date), task.seed)
    raise ValueError(task.strategy)


def run_task(task: Task, experiment_id: str, latency_bars: int) -> dict[str, Any]:
    """One replay. B3 returns only its daily P&L at its size; the others return their rows."""
    level = LEVELS[task.level]
    engine = Engine(_STATE["market"], _strategy(task), level, _STATE["fees"], EngineParams(latency_bars=latency_bars),
                    run_id=f"{experiment_id}:{task.key}")
    result = asyncio.run(engine.run(_STATE["sessions"]))
    replayed = {d.date: d.pnl for d in result.days}
    at_tier0 = daily_pnl_at(result.rows, level, _STATE["fees"], Decimal(1))
    carried = sum(len(d.carried) for d in result.days)
    if not carried and any(at_tier0.get(d, Decimal(0)) != p for d, p in replayed.items()):
        raise AssertionError(f"{task.key}: rescaling at k=1 does not reproduce the replay's P&L")
    out: dict[str, Any] = {"key": task.key, "task": task, "days": [d.date for d in result.days],
                           "entries": Counter(r["occurred_at"].astimezone(NEW_YORK).date() for r in result.rows
                                              if r["state"] == "submitted"),
                           "carried": carried, "requests": result.requests}
    if task.strategy == "B3":
        k = scale(task.size)
        out["pnl"] = daily_pnl_at(result.rows, level, _STATE["fees"], k)
        return out
    out.update(rows=result.rows, rejections=dict(result.rejections), fees_paid=result.fees_paid)
    return out


def run_tasks(tasks: Sequence[Task], *, root: Path, symbols: tuple[str, ...], samples: Samples,
              folds: Mapping[str, list[FoldResult]], prediction_ids: Mapping[tuple[str, str, str], str],
              dates: Sequence[date], experiment_id: str, plan: Plan,
              market: MarketData | None = None) -> dict[str, dict[str, Any]]:
    """``market`` in process only (``plan.workers <= 1``); a pool's workers load it from ``root``."""
    init = (str(root), symbols, samples, dict(folds), dict(prediction_ids), tuple(dates))
    if plan.workers <= 1:
        _init_worker(*init, market=market)
        return {t.key: run_task(t, experiment_id, plan.latency_bars) for t in sorted(tasks, key=lambda t: t.key)}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=ctx, initializer=_init_worker,
                             initargs=init) as pool:
        futures = {t.key: pool.submit(run_task, t, experiment_id, plan.latency_bars) for t in tasks}
        return {key: futures[key].result() for key in sorted(futures)}


# --- daily series and the pass rule ------------------------------------------------------------------

@dataclass
class Series:
    """One strategy at one level and size: daily returns on its capital base."""

    name: str
    level: str
    size: Decimal
    dates: list[date]
    returns: np.ndarray
    pnl: list[Decimal]
    rows: list[dict[str, Any]] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)

    def by_date(self) -> dict[date, float]:
        return dict(zip(self.dates, self.returns.tolist(), strict=True))


def router_series(name: str, level: str, size: Decimal, dates: Sequence[date], result: Mapping[str, Any],
                  fees: FeeTable) -> Series:
    k = scale(size)
    pnl_by_day = result["pnl"] if "pnl" in result else daily_pnl_at(result["rows"], LEVELS[level], fees, k)
    pnl = [pnl_by_day.get(d, Decimal(0)) for d in dates]
    base = CAPITAL_AT_TIER0 * k
    return Series(name, level, size, list(dates), np.array([float(p / base) for p in pnl]), pnl,
                  result.get("rows", []), result.get("rejections", {}))


def hold_series(name: str, level: str, size: Decimal, market: MarketData, symbols: tuple[str, ...],
                dates: Sequence[date], fees: FeeTable, band: Decimal | None) -> Series:
    sessions = [market.calendar.session(d) for d in dates]
    days = run_hold(market, symbols, sessions, LEVELS[level], fees, band=band)
    capital = Decimal("100000")
    values = [capital] + [d.value for d in days]
    returns = np.array([float(values[i + 1] / values[i] - 1) for i in range(len(days))])
    return Series(name, level, size, list(dates), returns, [values[i + 1] - values[i] for i in range(len(days))])


def half_year(d: date) -> str:
    return f"{d.year}H{1 if d.month <= 6 else 2}"


def symbol_nights(series: Series) -> int:
    """Distinct (symbol, session) pairs S2 actually entered."""
    return len({(r["symbol"], datetime.fromisoformat(r["detail"]["bar_start"]).astimezone(NEW_YORK).date())
                for r in series.rows if r["state"] == "filled"})


def pass_conditions(s2: Series, b1: Series, *, min_symbol_nights: int, resamples: int, seed: int) -> dict[str, Any]:
    """The registered rule's development conditions for one S2 variant at one level and size."""
    b1_by_date = b1.by_date()
    excess = np.array([r - b1_by_date[d] for d, r in zip(s2.dates, s2.returns, strict=True)])
    interval = metrics.block_bootstrap(excess, np.mean, n=resamples, seed=seed)
    halves: dict[str, list[float]] = defaultdict(list)
    for d, x in zip(s2.dates, excess, strict=True):
        halves[half_year(d)].append(float(x))
    half_means = {h: float(np.mean(v)) for h, v in sorted(halves.items())}
    nights = symbol_nights(s2)
    checks = {"interval_above_zero": interval.low > 0, "positive_each_half_year": all(m > 0 for m in half_means.values()),
              "min_symbol_nights": nights >= min_symbol_nights}
    return {"mean_daily_excess": interval.as_dict(), "half_years": half_means, "symbol_nights": nights,
            "checks": checks, "passes": all(checks.values())}


def assess(series: Mapping[tuple[str, str, str], Series], plan: Plan, criteria: Mapping[str, Any]) -> dict[str, Any]:
    """The registered pass rule, at the registered size. Development only: the lock-box is closed."""
    size = str(Decimal(criteria["order_notional_usd"]))
    per_level = {}
    for level in plan.levels:
        per_level[level] = {variant: pass_conditions(series[(variant, level, size)], series[("B1", level, size)],
                                                     min_symbol_nights=300, resamples=plan.resamples,
                                                     seed=plan.bootstrap_seed)
                            for variant in ("S2", "S2_collapsed")}
    central = per_level["central"]
    development = central["S2"]["passes"] and central["S2_collapsed"]["passes"]
    only_optimistic = (not development and per_level.get("optimistic", {}).get("S2", {}).get("passes", False))
    verdict = ("passes on development data; the lock-box criteria are still to confirm it" if development
               else "fails")
    return {"size_usd": size, "levels": per_level, "development_passes": development,
            "holds_only_at_optimistic": only_optimistic, "verdict": verdict,
            "lockbox": "not opened: the registered lock-box criteria need 1,000 symbol-nights after 2026-09-18"}


def to_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
