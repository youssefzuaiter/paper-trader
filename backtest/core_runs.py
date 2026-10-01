"""The long-term core's commands (core design §1, §4, §7): ``register-core``, ``leakage-core``, ``run-core``.

* ``register-core`` writes one experiment before any core number exists: the
  whole grid, windows, stress dates, cost levels, sizes, tax rates, raise-cash
  scenarios, reading rules, the owner's tolerance and the history warning
  (``core_grid.registration_params``). It refuses once a ``run-core`` result
  exists, unless it names the registration it supersedes (which still counts).
* ``leakage-core`` runs checks C1-C12, each beside a broken twin that must fail.
* ``run-core`` takes a registration id and refuses (C10) without it, if the
  code's grid no longer equals what was registered, or without a passing
  ``leakage-core`` at the same commit. It runs every registered configuration
  and writes the report; ``reproduce`` re-runs it from its commit.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from backtest import core_grid as G
from backtest import core_report, portfolio, registry
from backtest.core_fetch import BTC, ETFS, CorePaths
from backtest.costs import CORE_LEVELS, FeeTable
from backtest.daily import DailyMarket
from backtest.store import Store, canonical_json

logger = logging.getLogger("backtest.core")

REGISTRATION: Final[str] = "core-registration"
LEAKAGE: Final[str] = "leakage-core"
RUN: Final[str] = "run-core"
PERSISTED_FAMILIES: Final[frozenset[str]] = frozenset({"core", "crypto", "tax", "instruments", "buffer"})

WriteReport = Callable[[Path, registry.Experiment, str, str, Store], Path]


def core_paths(root: Path) -> CorePaths:
    return CorePaths.under(root / ".cache" / "backtest")


def core_inputs(root: Path) -> list[Path]:
    paths = core_paths(root)
    return [*paths.market_files(), paths.quotes_sample, Path(__file__).with_name("fees.json")]


def load_market(root: Path) -> DailyMarket:
    paths = core_paths(root)
    from backtest import core_fetch
    if core_fetch.missing(paths):
        raise FileNotFoundError("core data missing: run `python -m backtest ingest-core` first")
    return DailyMarket.load(paths, [*ETFS, BTC], last=G.WINDOWS["full"][1])


# --- ingest-core ---------------------------------------------------------------------------------------------

def cmd_ingest_core(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                    write_report: WriteReport) -> dict[str, Any]:
    root = Path(params["root"])
    paths = core_paths(root)
    from backtest import core_fetch
    if core_fetch.missing(paths):
        logger.info("fetching the core's missing files")
        core_fetch.run(paths)
    with registry.begin(store, hypothesis="The core's daily bars, BTC samples and quote sample are complete",
                        command="ingest-core", params=params, seeds={}, inputs=core_inputs(root), data_root=root,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        market = load_market(root)
        first, last = market.index(G.WINDOWS["full"][0]) - G.LOOKBACK - 1, market.index(G.WINDOWS["full"][1])
        gaps = {s: [market.dates[k].isoformat() for k in range(max(first, market.first_index(s)), last + 1)
                    if not market.has(s, k)] for s in (*ETFS, BTC)}
        metrics = {"sessions": len(market.sessions), "gaps": {s: len(v) for s, v in gaps.items()},
                   "first_session": market.dates[0].isoformat(), "last_session": market.dates[-1].isoformat()}
        body = "\n".join(["# Core ingest", "", *(f"- {k}: {v}" for k, v in metrics.items()), ""])
        exp.finish(metrics, report_body=body, conclusion="no gaps" if not any(gaps.values()) else "gaps found")
        write_report(root, exp, "ingest-core", body, store)
    return metrics


# --- leakage-core ---------------------------------------------------------------------------------------------

def leakage_defaults() -> dict[str, Any]:
    return {"c1_instants": 500, "c2_instants": 10_000, "seed": 1,
            "phase1_run": "x-20260927-112403-badb5a8d"}  # the clean phase 1 run (f700c0d), C7's target


def cmd_leakage_core(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                     write_report: WriteReport) -> dict[str, Any]:
    from backtest import leakage_core
    from backtest.data import DAILY_CACHE
    from backtest.fetch import SYMBOLS, Paths

    root = Path(params["root"])
    phase1 = Paths(root / ".cache" / "backtest")
    inputs = [*core_inputs(root), phase1.calendar, *(phase1.bars_1min(s) for s in SYMBOLS), phase1.bars_1day_raw,
              root / ".cache" / "training" / DAILY_CACHE, leakage_core.GOLDEN]
    with registry.begin(store, hypothesis="The core's point-in-time, execution, accounting and one-engine rules hold "
                        "on real data, and each check catches its broken twin", command=LEAKAGE, params=params,
                        seeds={"checks": params["seed"]}, inputs=inputs, data_root=root,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        checks = leakage_core.run_all(root, store, params)
        all_pass = all(c.passed for c in checks)
        lines = ["# Core leakage and consistency checks (C1-C12)", "",
                 "A twin row passes when its check fails on the deliberately broken configuration.", "",
                 "| Check | Result | Detail |", "|---|---|---|",
                 *(f"| {c.name} | {'pass' if c.passed else '**FAIL**'} | {_short(c.detail)} |" for c in checks), ""]
        body = "\n".join(lines)
        metrics = {"checks": {c.name: c.passed for c in checks}, "all_pass": all_pass}
        exp.finish(metrics, report_body=body, conclusion="all pass" if all_pass else "FAILURES")
        path = write_report(root, exp, LEAKAGE, body, store)
    return {"experiment": exp.experiment_id, "report": str(path), **metrics}


def _short(detail: dict[str, Any]) -> str:
    parts = []
    for k, v in detail.items():
        parts.append(f"{k}: {len(v)}" if isinstance(v, (list, dict)) and k != "refused" else f"{k} {v}")
    return "; ".join(parts)[:200]


# --- register-core ---------------------------------------------------------------------------------------------

def cmd_register_core(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                      write_report: WriteReport) -> dict[str, Any]:
    root = Path(params["root"])
    if parent_id is None:  # a reproduction re-registers after runs exist, by design
        runs = [r for r in store.experiments(command=RUN) if r["status"] == "done" and r["parent_id"] is None]
        supersedes = params.get("supersedes")
        if runs and not supersedes:
            raise PermissionError("run-core results exist: a new registration must name the one it supersedes "
                                  "(--supersedes ID); the superseded one still counts")
        if supersedes:
            old = store.experiment(supersedes)
            if old["command"] != REGISTRATION or old["status"] != "done":
                raise ValueError(f"{supersedes} is not a completed core registration")
    with registry.begin(store, hypothesis="The long-term core's grid, windows, costs, reading rules and the owner's "
                        "tolerance, fixed before any core result", command=REGISTRATION, params=params, seeds={
                            "bootstrap": G.BOOTSTRAP_SEED, "need_dates": G.NEED_DATES_SEED},
                        inputs=core_inputs(root), data_root=root, window=G.WINDOWS["full"],
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        reg = params["registration"]
        body = "\n".join(["# Core registration", "", f"> {reg['not_advice']}", "",
                          f"> **History warning.** {reg['history_warning']}", "",
                          "```json", json.dumps(reg, indent=1, sort_keys=True), "```", ""])
        exp.finish({"grid_runs": reg["grid_runs"]}, report_body=body, conclusion="registered")
        write_report(root, exp, "core-registration", body, store)
    if parent_id is None and params.get("supersedes"):
        store.supersede(params["supersedes"], exp.experiment_id)
    return {"experiment": exp.experiment_id, "grid_runs": params["registration"]["grid_runs"]}


def registered(store: Store, registration_id: str) -> dict[str, Any]:
    row = store.experiment(registration_id)
    if row["command"] != REGISTRATION or row["status"] != "done":
        raise PermissionError(f"{registration_id} is not a completed core registration")
    reg = json.loads(row["params"])["registration"]
    if canonical_json(reg) != canonical_json(G.registration_params()):
        raise PermissionError("the code's grid differs from the registration: register again, naming the old one "
                              "(C10)")
    return reg


def passing_leakage(store: Store, commit: str) -> str:
    rows = [r for r in store.experiments(command=LEAKAGE)
            if r["status"] == "done" and r["git_commit"] == commit and not r["git_dirty"]
            and json.loads(r["metrics"] or "{}").get("all_pass")]
    if not rows:
        raise PermissionError(f"no passing leakage-core at commit {commit[:8]}: run `python -m backtest "
                              "leakage-core` first (C10)")
    return rows[-1]["parent_id"] or rows[-1]["experiment_id"]


# --- run-core --------------------------------------------------------------------------------------------------

def simulate_all(market: DailyMarket, configs: list[G.Config], fees: FeeTable,
                 on_days: Callable[[G.Config, list[portfolio.Day]], None] | None = None) -> dict[str, core_report.Run]:
    """Every configuration, compacted as soon as it is simulated; ``on_days`` sees the full days first."""
    out = {}
    for i, cfg in enumerate(configs):
        start, end = G.WINDOWS[cfg.window]
        first, last = market.index(start), market.index(end)
        days = portfolio.simulate(market, cfg.spec(), first, last, CORE_LEVELS[cfg.level], fees)
        if on_days is not None:
            on_days(cfg, days)
        out[cfg.run_key] = core_report.Run.from_days(cfg, days, first)
        if (i + 1) % 500 == 0:
            logger.info("simulated %d of %d", i + 1, len(configs))
    return out


def cmd_run_core(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                 write_report: WriteReport) -> dict[str, Any]:
    root = Path(params["root"])
    registration_id = params["registration"]
    registered(store, registration_id)
    commit, dirty = registry.git_state()
    leakage_id = "dirty run: not checked" if allow_dirty and dirty else passing_leakage(store, commit)
    with registry.begin(store, hypothesis="How the registered reference mixes and rules behaved, 2016-2026, by the "
                        "registered reading rules", command=RUN, params=params,
                        seeds={"bootstrap": G.BOOTSTRAP_SEED, "need_dates": G.NEED_DATES_SEED},
                        inputs=core_inputs(root), data_root=root, window=G.WINDOWS["full"],
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        market = load_market(root)
        fees = FeeTable.load()
        configs = G.grid()
        scenarios = G.scenarios(market.dates)
        logger.info("simulating %d runs and %d raise-cash scenarios", len(configs), len(scenarios))
        persister = _Persister(store, exp.experiment_id)
        runs = simulate_all(market, configs + scenarios, fees, persister.add)
        persister.flush()
        logger.info("statistics and report")
        body, metrics = core_report.build(runs, market, registration_id=registration_id, leakage_id=leakage_id,
                                          configuration=registry.configuration(store, exp.experiment_id),
                                          grid_runs=len(configs), scenario_runs=len(scenarios))
        exp.finish(metrics, report_body=body, conclusion=f"{len(configs)} runs, {len(scenarios)} scenarios")
        path = write_report(root, exp, "run-core", body, store)
    return {"experiment": exp.experiment_id, "report": str(path)}


class _Persister:
    """Writes each run's row, and the daily rows of the persisted families, as runs finish."""

    def __init__(self, store: Store, experiment_id: str) -> None:
        self.store, self.experiment_id = store, experiment_id
        self.rows: list[dict[str, Any]] = []
        self.days: list[dict[str, Any]] = []

    def add(self, c: G.Config, run_days: list[portfolio.Day]) -> None:
        run_id = f"{self.experiment_id}:{c.run_key}"
        self.rows.append({"run_id": run_id, "experiment_id": self.experiment_id, "strategy": c.mix,
                          "cost_level": c.level, "variant": c.run_key,
                          "params": {"family": c.family, "window": c.window, "rule": c.rule, "size": str(c.size),
                                     "tax_rate": str(c.tax_rate), "buffer": str(c.buffer),
                                     "substitute": dict(c.substitute),
                                     "need": [c.need[0].isoformat(), str(c.need[1])] if c.need else None,
                                     "raise_method": c.raise_method},
                          "metrics": {"end_value": str(run_days[-1].value)}})
        if c.family in PERSISTED_FAMILIES:
            self.days += [{"run_id": run_id, "day": d.date, "value": d.value, "cash": d.cash,
                           "weights": {s: str(w) for s, w in d.weights.items()}, "traded": d.traded,
                           "exec_cost": d.exec_cost, "fees": d.fees, "tax": d.tax, "flow": d.flow,
                           "rebalanced": d.rebalanced, "deferred": d.deferred, "decided": d.decided,
                           "orders": d.orders or None} for d in run_days]
        if len(self.days) >= 200_000:
            self.flush()

    def flush(self) -> None:
        self.store.put_runs(self.rows)
        self.store.put_core_days(self.days)
        self.rows, self.days = [], []
