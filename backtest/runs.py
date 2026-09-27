"""The walk-forward, its leakage checks, and the strategy runs, as registered commands.

All three need the owner's registered criteria (D3, D5): the walk-forward's thresholds
depend on the order size, and nothing reads an S2 result before the pass rule exists.
Every parameter, including exactly how the pass rule is evaluated, is written into the
experiment row before anything is computed.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from backtest import costs, leakage, metrics, outcomes, pipeline, registry, walkforward
from backtest import events as ev
from backtest import report as reports
from backtest.costs import LEVELS, FeeTable
from backtest.data import DAILY_CACHE, NEWS_CACHE, MarketData
from backtest.dataset import FINBERT_CACHE, DatasetConfig, Samples, build_v2, load_sentiment
from backtest.fetch import Paths
from backtest.store import Store, chunks
from backtest.walkforward import THRESHOLD_GRID, FoldResult, Schedule, central_cost

logger = logging.getLogger("backtest.runs")

WINDOW = (pipeline.Plan().first_session, pipeline.Plan().last_session)


def _paths(root: Path) -> tuple[Path, Path]:
    return root / ".cache" / "backtest", root / ".cache" / "training"


def research_inputs(root: Path, store: Store, symbols: tuple[str, ...]) -> tuple[MarketData, Samples, dict[str, int]]:
    backtest_cache, training = _paths(root)
    market = MarketData.load(Paths(backtest_cache), symbols=symbols, training_cache=training)
    events = ev.from_store(store.events(), store.event_symbols())
    sentiment = load_sentiment([training / FINBERT_CACHE, backtest_cache / FINBERT_CACHE])
    samples, dropped = build_v2(events, market, sentiment, DatasetConfig())
    return market, samples, dropped


def research_manifest(root: Path, symbols: tuple[str, ...]) -> list[Path]:
    backtest_cache, training = _paths(root)
    paths = Paths(backtest_cache)
    return [paths.calendar, *(paths.bars_1min(s) for s in symbols), paths.bars_1day_raw, paths.news,
            training / NEWS_CACHE, training / DAILY_CACHE, training / FINBERT_CACHE,
            Path(__file__).with_name("fees.json")]


def fit(samples: Samples, market: MarketData, sizes: list[Decimal], schedule: Schedule) -> dict[str, list[FoldResult]]:
    """One set of fitted folds; thresholds per order size (only the fee cost differs)."""
    fees = FeeTable.load()
    primary = walkforward.run(samples, schedule, market.calendar, central_cost(fees, sizes[0], market.calendar))
    out = {str(sizes[0]): primary}
    for size in sizes[1:]:
        out[str(size)] = walkforward.at_cost(samples, primary, central_cost(fees, size, market.calendar))
    return out


def _criteria(store: Store) -> tuple[str, dict[str, Any]]:
    done = [r for r in store.experiments(command="criteria") if r["status"] == "done"]
    if not done:
        raise PermissionError("no registered order size and pass criteria: run `python -m backtest register-criteria`")
    return done[-1]["experiment_id"], json.loads(done[-1]["params"])


def _sizes(criteria: dict[str, Any]) -> list[Decimal]:
    registered = Decimal(criteria["order_notional_usd"])
    return [pipeline.TIER0] + ([registered] if registered != pipeline.TIER0 else [])


# --- the walk-forward -----------------------------------------------------------------------------------

def cmd_walkforward(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                    write_report: Any) -> dict[str, Any]:
    root, symbols = Path(params["root"]), tuple(params["symbols"])
    criteria_id, criteria = _criteria(store)
    params = {**params, "criteria_experiment": criteria_id, "sizes_usd": [str(s) for s in _sizes(criteria)],
              "schedule": Schedule().as_params(), "dataset": DatasetConfig().as_params(),
              "threshold_rule": {"grid": list(THRESHOLD_GRID), "min_symbol_days": walkforward.MIN_SYMBOL_DAYS,
                                 "cost": "central level, fees rounded per order at each order size"},
              "auc_bootstrap": {"mean_block": metrics.MEAN_BLOCK, "resamples": metrics.RESAMPLES, "seed": 0}}
    with registry.begin(store, hypothesis="Monthly walk-forward on v2 labels: an out-of-sample prediction for every "
                        "development event, S2/S3 thresholds fixed from each fold's calibration segment",
                        command="walkforward", params=params, seeds={"gbm": 7, "auc_bootstrap": 0},
                        inputs=research_manifest(root, symbols), data_root=root, window=WINDOW,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        market, samples, dropped = research_inputs(root, store, symbols)
        sizes = [Decimal(s) for s in params["sizes_usd"]]
        folds = fit(samples, market, sizes, Schedule())
        primary = folds[str(sizes[0])]
        rows = walkforward.model_rows(primary, experiment_id=exp.experiment_id,
                                      git_commit=store.experiment(exp.experiment_id)["git_commit"],
                                      data_hash=store.experiment(exp.experiment_id)["data_hash"],
                                      label=DatasetConfig().as_params(), schedule=Schedule())
        for row, month_folds in zip(rows, zip(*folds.values(), strict=True), strict=True):
            row["thresholds"] = {f"${size}": f.thresholds for size, f in zip(folds, month_folds, strict=True)}
        store.put_models(rows)
        predictions = walkforward.prediction_rows(samples, primary, experiment_id=exp.experiment_id)
        for batch in chunks(predictions, 20_000):
            store.put_predictions(batch)
        resolved = outcomes.resolve(predictions, market)
        for batch in chunks(resolved, 50_000):
            store.put_outcomes(batch)
        summary = walkforward_metrics(samples, primary, folds, dropped, store)
        body = walkforward_report(summary, sizes)
        exp.finish(pipeline.to_json(summary), report_body=body,
                   conclusion=f"pooled out-of-sample AUC {summary['auc']['all']['estimate']:.4f}")
        path = write_report(root, exp, "walkforward", body, store)
    return {"experiment": exp.experiment_id, "report": str(path), "samples": len(samples),
            "predictions": len(predictions), "outcomes": len(resolved)}


def walkforward_metrics(samples: Samples, primary: list[FoldResult], folds: dict[str, list[FoldResult]],
                        dropped: dict[str, int], store: Store) -> dict[str, Any]:
    idx = np.concatenate([f.scored for f in primary])
    prob = np.concatenate([f.prob_scored for f in primary])
    day = samples.session[idx]
    auc = {}
    for name, mask in (("all", np.ones(len(idx), bool)), ("night", samples.night[idx]),
                       ("tradeable", samples.tradeable[idx])):
        auc[name] = {**metrics.bootstrap_auc(samples.y[idx][mask], prob[mask], day[mask],
                                             n=metrics.RESAMPLES, seed=0).as_dict(), "n": int(mask.sum())}
    updated = {e["event_id"]: e["updated_at"] for e in store.events()}
    per_fold = []
    for k, f in enumerate(primary):
        y = samples.y[f.scored]
        per_fold.append({
            "month": f.bounds.month.isoformat(), "train": len(f.train), "calib": len(f.calib), "scored": len(f.scored),
            "calib_auc": f.calib_auc,
            "scored_auc": float(roc_auc_score(y, f.prob_scored)) if 0 < y.sum() < len(y) else None,
            "thresholds": {size: folds[size][k].thresholds for size in folds},
            "model_version": f.model_version, "artifact_sha256": f.artifact_sha256})
    trading = {size: {s: sum(1 for f in fs if f.thresholds[s] is not None) for s in walkforward.STRATEGIES}
               for size, fs in folds.items()}
    return {"samples": len(samples), "digest": samples.digest(), "dropped": dropped, "scored": len(idx),
            "auc": auc, "folds": per_fold, "folds_trading": trading,
            "revision": leakage.revision_auc(samples, primary, updated)}


def walkforward_report(m: dict[str, Any], sizes: list[Decimal]) -> str:
    lines = ["# Walk-forward (v2 labels, monthly refits, 5-session embargo)", "", reports.SELECTION_BIAS, "",
             f"{m['samples']} samples (digest `{m['digest'][:12]}`), {m['scored']} scored out of sample in 21 folds "
             "(2025-01 to 2026-09). Dropped before sampling: "
             + ", ".join(f"{k} {v}" for k, v in sorted(m["dropped"].items())) + ".", "",
             "| Scored population | n | AUC [95%, days resampled] |", "|---|---:|---|"]
    for name, a in m["auc"].items():
        lines.append(f"| {name} | {a['n']} | {a['estimate']:.4f} [{a['low']:.4f}, {a['high']:.4f}] |")
    lines += ["", "Folds whose threshold rule found a tradeable threshold (of 21):", "",
              "| Order size | " + " | ".join(walkforward.STRATEGIES) + " |", "|---|" + "---:|" * len(walkforward.STRATEGIES)]
    for size, counts in m["folds_trading"].items():
        lines.append(f"| ${size} | " + " | ".join(str(counts[s]) for s in walkforward.STRATEGIES) + " |")
    lines += ["", "| Month | Train | Calib | Scored | Calib AUC | Scored AUC | "
              + " | ".join(f"S2 ${s} | S3 ${s}" for s in sizes) + " |",
              "|---|---:|---:|---:|---:|---:|" + "---:|---:|" * len(sizes)]
    for f in m["folds"]:
        cells = [f"{_t(f['thresholds'][str(s)]['S2'])} | {_t(f['thresholds'][str(s)]['S3'])}" for s in sizes]
        lines.append(f"| {f['month'][:7]} | {f['train']} | {f['calib']} | {f['scored']} | {_a(f['calib_auc'])} | "
                     f"{_a(f['scored_auc'])} | " + " | ".join(cells) + " |")
    lines += ["", ("Revision diagnostic (AUC of scored predictions by how long after publication the article was "
              "last revised):"), "", "| Revised | n | AUC |", "|---|---:|---:|",
              *(f"| {r['bucket']} | {r['n']} | {_a(r['auc'])} |" for r in m["revision"]), ""]
    return "\n".join(lines)


def _t(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def _a(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


# --- leakage checks on the real walk-forward ------------------------------------------------------------------

def cmd_leakage_wf(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
                   write_report: Any) -> dict[str, Any]:
    from dataclasses import replace

    root, symbols = Path(params["root"]), tuple(params["symbols"])
    criteria_id, criteria = _criteria(store)
    size = _sizes(criteria)[-1]
    params = {**params, "criteria_experiment": criteria_id, "size_usd": str(size), "schedule": Schedule().as_params(),
              "l2_seeds": 10, "l2_bootstrap": 1000, "l2_placebo_seeds": 100, "l8_n": 1000, "seed": 0}
    with registry.begin(store, hypothesis="On the real walk-forward, the future canary, label shuffling, the "
                        "training cut-off, story straddling and feature parity rules hold, and their twins are caught",
                        command="leakage-wf", params=params, seeds={"l2": list(range(10)), "l8": 0},
                        inputs=research_manifest(root, symbols), data_root=root, window=WINDOW,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        market, samples, _ = research_inputs(root, store, symbols)
        calendar, cost = market.calendar, central_cost(FeeTable.load(), size, market.calendar)
        schedule = Schedule()
        folds = walkforward.run(samples, schedule, calendar, cost)
        checks = [leakage.future_canary(samples, schedule, calendar, cost, base_folds=folds),
                  _twin("L1 twin: canary joined on its session",
                        leakage.future_canary(samples, schedule, calendar, cost, join="session", base_folds=folds)),
                  leakage.training_cutoff(samples, folds, calendar)]
        broken = replace(schedule, embargo_sessions=-1)
        checks.append(_twin("L5 twin: embargo of -1 session", leakage.training_cutoff(
            samples, walkforward.run(samples, broken, calendar, cost), calendar)))
        checks.append(leakage.story_straddle(samples, folds))
        no_embargo = replace(schedule, embargo_sessions=0)
        checks.append(_twin("L6 twin: embargo 0", leakage.story_straddle(
            samples, walkforward.run(samples, no_embargo, calendar, cost))))
        predictions = walkforward.prediction_rows(samples, folds, experiment_id=exp.experiment_id)
        checks.append(leakage.feature_parity(predictions, market, n=1000, seed=0))
        checks.append(_twin("L8 twin: features cut off a session late",
                            leakage.feature_parity(predictions, market, n=1000, seed=0, late=True)))
        sessions = sorted({int(s) for f in folds for s in samples.session[f.scored]})
        night = leakage.night_returns(market, sessions)
        shuffles, screens, leaky = [], [], []
        for seed in range(10):
            shuffled = leakage.shuffle_labels(samples, seed)
            s_folds = walkforward.run(shuffled, schedule, calendar, cost)
            shuffles.append((seed, shuffled, s_folds))
            screens.append(leakage.s2_screen(shuffled, s_folds, night, seed=seed))
            leaky.append(leakage.s2_screen(shuffled, leakage.leaky_folds(shuffled, s_folds, cost), night, seed=seed))
        checks.append(leakage.auc_contains_half(shuffles, bootstrap=1000))
        inside = sum(c.passed for c in screens)
        caught = sum(not c.passed for c in leaky)
        checks.append(leakage.Check("L2-screen", inside >= 8, {"seeds_inside_band": inside, "of": 10}))
        checks.append(leakage.Check("L2-screen twin: thresholds on the scored month", caught >= 8,
                                    {"seeds_caught": caught, "of": 10}))
        body = "\n".join(["# Leakage checks on the real walk-forward", "", reports.SELECTION_BIAS, "",
                          "A twin row passes when its check fails on the deliberately broken configuration.", "",
                          "| Check | Result | Detail |", "|---|---|---|",
                          *(f"| {c.name} | {'pass' if c.passed else '**FAIL**'} | {_short(c)} |" for c in checks),
                          "", ("Not run: L2's engine half (S2 replayed against 100 B3 placebo seeds for each of the "
                          "10 shuffles); the vectorised screen above tests the same rule (P6)."), ""])
        metrics_out = {"checks": {c.name: c.passed for c in checks},
                       "details": {c.name: pipeline.to_json(c.detail) for c in checks}}
        exp.finish(metrics_out, report_body=body,
                   conclusion="all pass" if all(c.passed for c in checks) else "FAILURES: see the report")
        path = write_report(root, exp, "leakage-walkforward", body, store)
    return {"experiment": exp.experiment_id, "report": str(path), "checks": metrics_out["checks"]}


def _twin(name: str, check: leakage.Check) -> leakage.Check:
    return leakage.Check(name, not check.passed, check.detail)


def _short(check: leakage.Check) -> str:
    parts = []
    for k, v in check.detail.items():
        if isinstance(v, float):
            parts.append(f"{k} {v:.4g}")
        elif isinstance(v, (list, dict)):
            parts.append(f"{k}: {len(v)}")
        else:
            parts.append(f"{k} {v}")
    return "; ".join(parts)[:180]


# --- the strategies ------------------------------------------------------------------------------------------------

PASS_RULE_EVALUATION = {
    "size": "the registered order size (D3: decide on the real-money size); $10 reported beside it",
    "level": "central decides; optimistic and pessimistic are reported, and a pass only at optimistic is a fail",
    "variants": "S2 (every event, score = mean prob_up, D7) and S2 with duplicate stories collapsed must both pass",
    "excess": "daily return on the capital base ($50 x size/$10) minus B1's, every session 2025-01-02..2026-09-18",
    "interval": "95% stationary block bootstrap of the mean daily excess over days (mean block 5, 10,000 resamples, "
                "seed 0); its lower bound must be above zero",
    "half_years": "2025H1, 2025H2, 2026H1, 2026H2 (to 09-18): each mean daily excess above zero",
    "symbol_nights": "at least 300 distinct (symbol, session) entries actually filled, for each variant",
    "lockbox": "not evaluable yet: the lock-box has not accrued 1,000 symbol-nights",
}


def cmd_run(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None,
            write_report: Any) -> dict[str, Any]:
    root, symbols = Path(params["root"]), tuple(params["symbols"])
    criteria_id, criteria = _criteria(store)
    walkforwards = [r for r in store.experiments(command="walkforward") if r["status"] == "done"
                    and r["superseded_by"] is None]
    if not walkforwards:
        raise PermissionError("run the walk-forward first")
    wf = walkforwards[-1]
    plan = pipeline.Plan(sizes=tuple(_sizes(criteria)), workers=int(params.pop("workers", 1)))
    params = {**params, "criteria_experiment": criteria_id, "walkforward_experiment": wf["experiment_id"],
              "plan": plan.as_params(), "strategies": ["S0", "S1", "S2", "S2_collapsed", "S3", "S3_collapsed",
                                                        "B1", "B2", "B3"],
              "pass_rule": criteria["pass_rule"], "pass_rule_evaluation": PASS_RULE_EVALUATION}
    with registry.begin(store, hypothesis="S2 beats B1 after costs at the registered size (the registered pass rule); "
                        "S0-S3 and the baselines at three cost levels and two order sizes", command="run",
                        params=params, seeds={"b1": plan.b1_seed, "b3": list(plan.b3_seeds),
                                              "bootstrap": plan.bootstrap_seed},
                        inputs=research_manifest(root, symbols), data_root=root, window=WINDOW,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        market, samples, _ = research_inputs(root, store, symbols)
        folds = fit(samples, market, list(plan.sizes), Schedule())
        refit = walkforward.prediction_rows(samples, folds[str(plan.sizes[0])], experiment_id=wf["experiment_id"])
        registered = {r["prediction_id"]: r["prob_up"] for r in store.predictions(experiment_id=wf["experiment_id"])}
        # The same models as the registered walk-forward: every out-of-sample probability equal, bit for bit.
        # (Pickled artifacts are not compared: a fitted model's _BinMapper stores the OpenMP thread count.)
        if len(refit) != len(registered) or any(registered.get(r["prediction_id"]) != r["prob_up"] for r in refit):
            raise AssertionError(f"the refit walk-forward differs from {wf['experiment_id']}")
        registered_thresholds = {r["fold_month"].isoformat(): json.loads(r["thresholds"])
                                 for r in store.models(experiment_id=wf["experiment_id"])}
        for size, size_folds in folds.items():
            for f in size_folds:
                if registered_thresholds[f.bounds.month.isoformat()][f"${size}"] != f.thresholds:
                    raise AssertionError(f"thresholds for {f.bounds.month} at ${size} differ from {wf['experiment_id']}")
        prediction_ids = {(r["event_id"], r["symbol"], r["model_version"]): r["prediction_id"] for r in refit}
        dates = [s.date for s in market.calendar.sessions_between(plan.first_session, plan.last_session)]
        common = {"root": root, "symbols": symbols, "samples": samples, "folds": folds,
                  "prediction_ids": prediction_ids, "dates": dates, "experiment_id": exp.experiment_id, "plan": plan}
        first = [pipeline.Task(s, level, size) for s in pipeline.STRATEGY_KINDS for level in plan.levels
                 for size in plan.sizes]
        first += [pipeline.Task("B1", level, None, plan.b1_seed) for level in plan.levels]
        first += [pipeline.Task("B2", level, None) for level in plan.levels]
        logger.info("replaying %d runs", len(first))
        results = pipeline.run_tasks(first, **common)
        b3 = [pipeline.Task("B3", level, size, seed, tuple(sorted(results[f"S2:{level}:${size}"]["entries"].items())))
              for level in plan.levels for size in plan.sizes for seed in plan.b3_seeds]
        logger.info("replaying %d placebo runs", len(b3))
        results.update(pipeline.run_tasks(b3, **common))
        summary, body = _assemble(results, plan, market, dates, symbols, criteria, criteria_id, wf, store,
                                  exp.experiment_id)
        exp.finish(pipeline.to_json(summary), report_body=body, conclusion=summary["assessment"]["verdict"])
        path = write_report(root, exp, "strategies", body, store)
    return {"experiment": exp.experiment_id, "report": str(path), "verdict": summary["assessment"]["verdict"]}


def _assemble(results: dict[str, dict[str, Any]], plan: pipeline.Plan, market: MarketData, dates: list[Any],
              symbols: tuple[str, ...], criteria: dict[str, Any], criteria_id: str, wf: dict[str, Any], store: Store,
              experiment_id: str) -> tuple[dict[str, Any], str]:
    fees = FeeTable.load()
    series: dict[tuple[str, str, str], pipeline.Series] = {}
    for size in plan.sizes:
        for level in plan.levels:
            for name in ("S0", "S1"):
                series[(name, level, str(size))] = pipeline.hold_series(
                    name, level, size, market, symbols, dates, fees, None if name == "S0" else plan.s1_band)
            for name in pipeline.STRATEGY_KINDS:
                series[(name, level, str(size))] = pipeline.router_series(
                    name, level, size, dates, results[f"{name}:{level}:${size}"], fees)
            for name in ("B1", "B2"):
                key = f"{name}:{level}:-" + (f":seed{plan.b1_seed}" if name == "B1" else "")
                series[(name, level, str(size))] = pipeline.router_series(name, level, size, dates, results[key], fees)
    trade_rows = [r for key, res in results.items() if "rows" in res for r in res["rows"]]
    for batch in [trade_rows[i:i + 50_000] for i in range(0, len(trade_rows), 50_000)]:
        store.append_trade_events(batch)
    for key, res in results.items():
        store.put_run({"run_id": f"{experiment_id}:{key}", "experiment_id": experiment_id,
                       "strategy": res["task"].strategy, "cost_level": res["task"].level,
                       "variant": "-" if res["task"].size is None else f"${res['task'].size}",
                       "params": {"seed": res["task"].seed},
                       "metrics": {"entries": sum(res["entries"].values()), "carried": res["carried"]}})
    assessment = pipeline.assess(series, plan, criteria)
    table, placebo, bets, rejections, costs_by_run = {}, {}, {}, {}, {}
    for (name, level, size), s in series.items():
        key = f"{name}|{level}|{size}"
        row = reports.series_row(s, resamples=plan.resamples, seed=plan.bootstrap_seed)
        row["vs_S0"] = reports.excess_interval(s, series[("S0", level, size)], resamples=plan.resamples,
                                               seed=plan.bootstrap_seed) if name != "S0" else None
        if name in pipeline.STRATEGY_KINDS:
            row["vs_B1"] = reports.excess_interval(s, series[("B1", level, size)], resamples=plan.resamples,
                                                   seed=plan.bootstrap_seed)
            row["vs_B2"] = reports.excess_interval(s, series[("B2", level, size)], resamples=plan.resamples,
                                                   seed=plan.bootstrap_seed)
        table[key] = row
        if name == "S2":
            totals = [float(sum(r["pnl"].values(), Decimal(0)) / (pipeline.CAPITAL_AT_TIER0 * pipeline.scale(
                Decimal(size)))) for k, r in results.items() if k.startswith(f"B3:{level}:${size}:")]
            placebo[key] = {"percentile": float(np.mean(np.asarray(totals) <= s.returns.sum()) * 100),
                            "p5": float(np.quantile(totals, 0.05)), "p95": float(np.quantile(totals, 0.95))}
        if s.rows:
            k = pipeline.scale(Decimal(size))
            bets[key] = reports.independent_bets(s.rows)
            c = reports.cost_components(s.rows)
            costs_by_run[key] = {"spread_cost": c.get("spread_cost", 0) * float(k),
                                 "slippage_cost": c.get("slippage_cost", 0) * float(k), "fees": _fees_at(s, k)}
            rejections[key] = s.rejections
    wf_metrics = json.loads(wf["metrics"])
    wf_summary = [(f"Pooled out-of-sample AUC: all {wf_metrics['auc']['all']['estimate']:.4f} "
                  f"[{wf_metrics['auc']['all']['low']:.4f}, {wf_metrics['auc']['all']['high']:.4f}], night "
                  f"{wf_metrics['auc']['night']['estimate']:.4f} [{wf_metrics['auc']['night']['low']:.4f}, "
                  f"{wf_metrics['auc']['night']['high']:.4f}], tradeable {wf_metrics['auc']['tradeable']['estimate']:.4f} "
                  f"[{wf_metrics['auc']['tradeable']['low']:.4f}, {wf_metrics['auc']['tradeable']['high']:.4f}]."),
                  "Folds with a tradeable threshold (of 21): " + "; ".join(
                      f"${size}: " + ", ".join(f"{s} {n}" for s, n in counts.items())
                      for size, counts in wf_metrics["folds_trading"].items()) + "."]
    n, m = registry.configuration(store, experiment_id)
    notes = [f"Configuration {n} of {m} run on this window.",
             ("Every replay trades Tier-0's $10 through the production router; the registered size scales every "
             "dollar amount (fees and cent rounding recomputed at the scaled quantities, D12)."),
             ("S0 and S1 are per dollar, bought at the first open of the window; S2, S3 and the baselines are on their "
             "capital base ($50 per $10 order size, the router's gross cap)."),
             ("Not run in this configuration: latency of 2 and 5 bars, quarterly refits, the max and latest S2 "
             "scores (D7 variants). Each would be a new registered configuration.")]
    body = reports.run_report(window=(plan.first_session, plan.last_session), criteria=criteria,
                              criteria_id=criteria_id, walkforward_id=wf["experiment_id"],
                              assessment=assessment, table=table, bets=bets, rejections=rejections, placebo=placebo,
                              costs_by_run=costs_by_run, walkforward_summary=wf_summary, notes=notes)
    summary = {"assessment": assessment,
               "table": {k: {**{x: (v.as_dict() if isinstance(v, metrics.Interval) else v) for x, v in r.items()}}
                         for k, r in table.items()},
               "placebo": placebo, "bets": bets, "rejections": rejections,
               "entries": {k: sum(r["entries"].values()) for k, r in results.items() if not k.startswith("B3")},
               "carried": sum(r["carried"] for r in results.values())}
    return summary, body


def _fees_at(s: pipeline.Series, k: Decimal) -> float:
    """Fees paid at size k, the day's rounding included: the cash legs minus the P&L."""
    level = LEVELS[s.level]
    cash = Decimal(0)
    for r in s.rows:
        if r["state"] == "filled":
            cash -= costs.cash_debit(r["qty"] * k, r["price"], level)
        elif r["state"] == "exit_filled":
            cash += costs.cash_credit(r["qty"] * k, r["price"], level)
    return float(cash - sum(s.pnl, Decimal(0)))
