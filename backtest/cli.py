"""``python -m backtest <command>``. Every command that produces a number
registers an experiment before computing anything (design §8).

    ingest              fetch what is missing; events and stories into the store
    legacy              reproduce models/return_model/report.md (L4, L4b and their twins)
    leakage             the leakage checks that need no strategy run, on real data
    register-criteria   the owner's D3 order size and D5 pass criteria, before any S2 result
    walkforward         the monthly walk-forward: models, predictions, outcomes, thresholds per size
    leakage-wf          L1, L2, L5, L6 and L8 on the real walk-forward, each beside its twin
    run                 S0-S3 and B1-B3 at three cost levels and both sizes; the registered pass rule
    ingest-core         the long-term core's data: daily bars, BTC/USD 5-minute bars, the quote sample
    register-core       the core's whole grid, reading rules and the owner's tolerance, before any core result
    leakage-core        checks C1-C12 for the core, each beside a broken twin that must fail
    run-core ID         every registered core configuration and the report (ID: the registration)
    fx-core             the core's mixes translated into shekels (Bank of Israel USD/ILS), a registered study
    experiments         the registry, newest last
    reproduce ID        re-run an experiment from its commit; identical metrics and report hash or fail

Reports go to ``.cache/backtest/reports/``. ``walkforward``, ``leakage-wf``
and ``run`` refuse to start until the owner's order size and pass criteria
are registered: nothing reads an S2 result before the pass rule exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np

from backtest import core_fx, core_runs, leakage, registry, runs
from backtest import events as ev
from backtest import report as reports
from backtest.calendar import Calendar
from backtest.data import (
    DAILY_CACHE,
    INTRADAY_CACHE,
    NEWS_CACHE,
    MarketData,
    load_bar_cache,
    load_news_cache,
)
from backtest.dataset import FINBERT_CACHE, build_legacy
from backtest.engine import Engine
from backtest.fetch import SYMBOLS, Paths
from backtest.lockbox import LOCKBOX_SESSIONS_FROM
from backtest.store import Store
from swarm.common import NEW_YORK

logger = logging.getLogger("backtest")

REPO: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_PASS_RULE: Final[str] = (
    "At the central level, S2's mean daily excess over B1 has a 95% interval above zero across the walk-forward "
    "folds and is positive in each half-year, and the pre-registered lock-box criteria confirm it. A result that "
    "holds only at the optimistic level fails.")


def _paths(root: Path) -> tuple[Path, Path, Path]:
    return root / ".cache" / "backtest", root / ".cache" / "training", root / ".cache" / "backtest" / "store.duckdb"


def _write_report(root: Path, experiment: registry.Experiment, name: str, body: str, store: Store) -> Path:
    n, m = registry.configuration(store, experiment.experiment_id)
    header = {"experiment": experiment.experiment_id, "command": experiment.command,
              "written": datetime.now(UTC).isoformat(timespec="seconds"), "configuration": f"{n} of {m}"}
    # Beside the registry that references them: a test's temporary store keeps its reports to itself.
    folder = (Path(store.path).parent if store.path != ":memory:" else _paths(root)[0]) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    text = reports.document(header, body)
    (folder / f"{experiment.experiment_id}-{name}.md").write_text(text, encoding="utf-8")
    latest = folder / f"{name}.md"
    latest.write_text(text, encoding="utf-8")
    return latest


def _market_inputs(backtest_cache: Path, symbols: tuple[str, ...]) -> list[Path]:
    paths = Paths(backtest_cache)
    return [paths.calendar, *(paths.bars_1min(s) for s in symbols), paths.bars_1day_raw]


# --- ingest ------------------------------------------------------------------------------------------

def cmd_ingest(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None) -> dict[str, Any]:
    root = Path(params["root"])
    backtest_cache, training, _ = _paths(root)
    paths = Paths(backtest_cache)
    missing = [p for p in paths.all_files(SYMBOLS) if not p.exists()]
    if missing:
        from backtest import fetch  # network: only when files are missing
        logger.info("fetching %d missing files", len(missing))
        fetch.run(paths)
    inputs = [paths.news, training / NEWS_CACHE]
    with registry.begin(store, hypothesis="Development-window news becomes point-in-time events with story ids",
                        command="ingest", params=params, seeds={}, inputs=inputs, data_root=root,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        from backtest import parquet
        fetched = parquet.read_rows(paths.news, "ORDER BY created_at, id")
        cached = load_news_cache(training / NEWS_CACHE)
        events, counts = ev.from_articles(fetched, cached)
        watch = frozenset(params["symbols"])
        events = ev.with_stories(events, ev.cluster(events, watch))
        rows, symbols = ev.store_rows(events)
        new = store.put_events(rows, symbols)
        stories = Counter(e.story_id for e in events if e.story_id)
        gaps = np.array([(e.updated_at - e.published_at).total_seconds() for e in events if e.updated_at])
        metrics = {**counts, "events": len(events), "new_in_store": new,
                   "with_watched_symbol": sum(1 for e in events if e.story_id),
                   "stories": len(stories), "stories_with_duplicates": sum(1 for n in stories.values() if n > 1),
                   "created_before_2024": sum(1 for e in events if e.published_at < datetime(2024, 1, 2, tzinfo=UTC)),
                   "revised_after_1h": int((gaps >= 3600).sum()), "revised_after_1d": int((gaps >= 86400).sum()),
                   "min_updated_at": min(e.updated_at for e in events if e.updated_at).isoformat()}
        body = "\n".join(["# Ingest", "", *(f"- {k}: {v}" for k, v in metrics.items()), ""])
        exp.finish(metrics, report_body=body, conclusion=f"{len(events)} events, {len(stories)} stories")
        _write_report(root, exp, "ingest", body, store)
    return metrics


# --- the report.md reproduction --------------------------------------------------------------------------

def _legacy_findings(samples: Any, calendar: Calendar, news_rows: list[dict[str, Any]],
                     cached: list[dict[str, Any]]) -> dict[str, Any]:
    entry_days = [datetime.fromtimestamp(t, UTC).astimezone(NEW_YORK) for t in samples.entry_at]
    half = [d for d in entry_days if (s := calendar.session(d.date())) is not None and s.half_day]
    after_close = [d for d in half if d.hour >= 13]
    pre = int((samples.published_at < datetime(2024, 1, 2, tzinfo=UTC).timestamp()).sum())
    old = [r for r in news_rows if r["created_at"] < datetime(2024, 1, 2, tzinfo=UTC)]
    fetched = {str(r["id"]): (r["headline"] or "").strip() for r in news_rows}
    held = {str(a["id"]): (a.get("headline") or "").strip() for a in cached}
    same_ids = set(fetched) == set(held)
    changed = sum(1 for k in set(fetched) & set(held) if fetched[k] != held[k])
    notes = [
        (f"**{len(half)} samples** are labelled on a 13:00 half-day ({len({d.date() for d in half})} sessions). "
        "The 30-minute cache holds extended hours, and `RegularBars.build` keeps bars starting 09:30-15:30, so "
        f"their exit is the 16:00 after-hours close, and {len(after_close)} of them even enter after the 13:00 close."),
        (f"**{pre} samples** come from articles created before 2024-01-02, weeks before the first 30-minute bar: "
        "their 'same-session' label is the 2024-01-02 session."),
        (f"The news API filters on `updated_at`, not `created_at`: all {len(old)} cached articles created "
        "2019-2023 were updated in 2024 "
        f"({min(r['updated_at'] for r in old).date()} to {max(r['updated_at'] for r in old).date()}). The "
        f"re-fetch returned {'the same' if same_ids else 'different'} {len(fetched):,} ids, {changed} with a "
        "different headline. A headline held may postdate `known_at` (the revision diagnostic runs with the "
        "walk-forward)."),
        ("v2 labels (the walk-forward's) take sessions from the exchange calendar and raw 1-minute bars, and drop "
        "an event whose entry session has no bars."),
    ]
    return {"half_day_samples": len(half), "half_day_after_close_entries": len(after_close),
            "pre_2024_samples": pre, "notes": notes}


def cmd_legacy(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None) -> dict[str, Any]:
    root = Path(params["root"])
    backtest_cache, training, _ = _paths(root)
    meta_path = root / "models" / "return_model" / "meta.json"
    symbols = tuple(params["symbols"])
    inputs = [meta_path, training / NEWS_CACHE, training / DAILY_CACHE, training / INTRADAY_CACHE,
              training / FINBERT_CACHE, *_market_inputs(backtest_cache, symbols), Paths(backtest_cache).news]
    with registry.begin(store, hypothesis="The backtester reproduces every number of report.md in legacy mode, "
                        "and its broker returns every legacy label", command="legacy", params=params, seeds={},
                        inputs=inputs, data_root=root, window=(date(2023, 11, 25), date(2026, 9, 18)),
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        l4, ours = leakage.reproduce_report(meta, symbols, training)
        twin_purge, _ = leakage.reproduce_report(meta, symbols, training, purge=timedelta(0))
        twin_entry, _ = leakage.reproduce_report(meta, symbols, training, entry="bisect_right")
        market = MarketData.load(Paths(backtest_cache), symbols=symbols, training_cache=training)
        samples = build_legacy(symbols, training)
        intraday = {s: [b for b in bars if b.t.astimezone(NEW_YORK).date() < LOCKBOX_SESSIONS_FROM]
                    for s, bars in load_bar_cache(training / INTRADAY_CACHE, symbols).items()}
        series = leakage.legacy_series(intraday, market.calendar)
        l4b = leakage.engine_agrees_with_labels(samples, market, series, tolerance=params["tolerance"])
        l4b_twin = leakage.engine_agrees_with_labels(samples, market, series, twin=True,
                                                     tolerance=params["tolerance"])
        from backtest import parquet
        findings = _legacy_findings(samples, market.calendar, parquet.read_rows(Paths(backtest_cache).news),
                                    load_news_cache(training / NEWS_CACHE))
        checks = [l4, _twin("L4 twin: purge 0", twin_purge), _twin("L4 twin: bisect_right entry", twin_entry),
                  l4b, _twin("L4b twin: fill on the bar containing the publication", l4b_twin)]
        body, counts = reports.reproduction_report(ours, meta, checks, findings)
        metrics = {**counts, "checks": {c.name: c.passed for c in checks}, "digest": ours["digest"],
                   "n_samples": ours["n_samples"], "l4b_max_abs_error": l4b.detail["max_abs_error"],
                   "l4b_twin_over_tolerance": l4b_twin.detail["over_tolerance"],
                   **{k: v for k, v in findings.items() if k != "notes"}}
        passed = all(c.passed for c in checks)
        exp.finish(metrics, report_body=body, conclusion=("reproduced exactly; every twin caught" if passed
                                                          else "NOT reproduced: see the report"))
        path = _write_report(root, exp, "legacy-reproduction", body, store)
    return {"experiment": exp.experiment_id, "report": str(path), **metrics}


def _twin(name: str, check: leakage.Check) -> leakage.Check:
    """A twin passes when its check *fails*."""
    return leakage.Check(name, not check.passed, check.detail)


# --- leakage checks on real data ------------------------------------------------------------------------------

def cmd_leakage(params: dict[str, Any], store: Store, *, allow_dirty: bool, parent_id: str | None) -> dict[str, Any]:
    root = Path(params["root"])
    backtest_cache, training, _ = _paths(root)
    symbols = tuple(params["symbols"])
    inputs = [*_market_inputs(backtest_cache, symbols), training / DAILY_CACHE]
    with registry.begin(store, hypothesis="The pipeline's point-in-time, lock-box, one-engine and calendar rules "
                        "hold on real data, and each check catches its broken twin", command="leakage",
                        params=params, seeds={"oracle": params["seed"]}, inputs=inputs, data_root=root,
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        market = MarketData.load(Paths(backtest_cache), symbols=symbols, training_cache=training)
        truth = market.calendar
        checks = [leakage.pit_oracle(market, n=params["oracle_instants"], seed=params["seed"]),
                  leakage.lockbox_guard(market, store),
                  _twin("L7 twin: guard off", leakage.lockbox_guard(_unguarded(backtest_cache, symbols, training))),
                  leakage.outcome_isolation(),
                  _twin("L9 twin: an outcomes import added", leakage.outcome_isolation(overrides={
                      "backtest.strategies": (REPO / "backtest" / "strategies.py").read_text()
                      + "\nfrom backtest.outcomes import resolve\n"})),
                  leakage.one_engine()]
        half_day, holiday = date.fromisoformat(params["half_day"]), date.fromisoformat(params["holiday"])
        for twin in (False, True):
            result = _calendar_replay(market, truth, half_day, holiday, backtest_cache, twin=twin)
            check = leakage.calendar_rules(result, truth, half_day=half_day, late_signal="sig-00000002")
            checks.append(_twin("L11 twin: weekday-rule calendar", check) if twin else check)
        month = date.fromisoformat(params["fixture_month"])
        fixture = leakage.fixture_month(truth, month)
        checks.append(leakage.skipping_invariance(*fixture))
        checks.append(_twin("L12 twin: exit passes skipped", leakage.skipping_invariance(*fixture, twin=True)))
        lines = ["# Leakage checks on real data", "", reports.SELECTION_BIAS, "",
                 "A twin row passes when its check fails on the deliberately broken configuration.", "",
                 "| Check | Result | Detail |", "|---|---|---|",
                 *(f"| {c.name} | {'pass' if c.passed else '**FAIL**'} | {_short(c)} |" for c in checks), "",
                 ("Run separately: L4 and L4b (with `legacy`); L1, L2, L5, L6 and L8 need the real walk-forward, "
                 "L2's strategy half and L3's full-run half need S2/S3 replays: both wait for `register-criteria`."),
                 ""]
        body = "\n".join(lines)
        metrics = {"checks": {c.name: c.passed for c in checks}}
        exp.finish(metrics, report_body=body, conclusion="all pass" if all(c.passed for c in checks) else "FAILURES")
        path = _write_report(root, exp, "leakage", body, store)
    return {"experiment": exp.experiment_id, "report": str(path), **metrics}


def _short(check: leakage.Check) -> str:
    d = check.detail
    parts = []
    for k, v in d.items():
        if isinstance(v, (list, dict)):
            parts.append(f"{k}: {len(v)}")
        else:
            parts.append(f"{k} {v}")
    return "; ".join(parts)[:160]


def _unguarded(backtest_cache: Path, symbols: tuple[str, ...], training: Path) -> MarketData:
    guarded = MarketData.load(Paths(backtest_cache), symbols=symbols, training_cache=training)
    daily = load_bar_cache(training / DAILY_CACHE, symbols)
    return leakage.UnguardedMarketData(guarded.calendar, guarded.minute, daily, {s: {} for s in symbols})


def _calendar_replay(market: MarketData, truth: Calendar, half_day: date, holiday: date, backtest_cache: Path, *,
                     twin: bool) -> Any:
    """L11's replay: an entry at 12:00 and one at 12:31 on a half-day, one on a holiday. The twin
    files the same raw bars (extended hours included) under a weekday-rule calendar."""
    from backtest import parquet
    from backtest.costs import LEVELS, FeeTable
    from backtest.data import regular_series

    if twin:
        naive = Calendar.weekday_rule(truth.sessions[0].date, truth.sessions[-1].date)
        paths = Paths(backtest_cache)
        minute = {}
        for s in ("AAPL", "TSLA"):
            cols = parquet.read_numpy(paths.bars_1min(s), "epoch(t)::BIGINT AS t, o, h, l, c, v", "ORDER BY t")
            minute[s] = regular_series(s, 60, cols["t"], cols["o"], cols["h"], cols["l"], cols["c"], cols["v"], naive)
        market = MarketData(naive, minute, {s: market.daily[s] for s in minute}, {s: {} for s in minute})
    sessions = [s for s in market.calendar.sessions if half_day <= s.date <= holiday + timedelta(days=3)]
    at = {s.date: s for s in truth.sessions}
    signals = [leakage.fixed_signal(1, at[half_day].open_at + timedelta(hours=2, minutes=30)),       # 12:00
               leakage.fixed_signal(2, at[half_day].open_at + timedelta(hours=3, minutes=1), "TSLA"),  # 12:31
               leakage.fixed_signal(3, datetime.combine(holiday, datetime.min.time(), NEW_YORK) + timedelta(hours=11))]
    engine = Engine(market, leakage.FixedSignals(signals), LEVELS["central"], FeeTable.load())
    return asyncio.run(engine.run(sessions))


# --- the owner's decisions ---------------------------------------------------------------------------------------

def cmd_register_criteria(params: dict[str, Any], store: Store, *, allow_dirty: bool,
                          parent_id: str | None) -> dict[str, Any]:
    """D3 and D5: the real-money order size and the pass criteria, fixed before any S2 result is read."""
    if Decimal(str(params["order_notional_usd"])) <= 0:
        raise ValueError("the order size must be positive")
    with registry.begin(store, hypothesis="The owner's order size and pass criteria, fixed before any S2 result",
                        command="criteria", params=params, seeds={}, inputs=[], data_root=Path(params["root"]),
                        allow_dirty=allow_dirty, parent_id=parent_id) as exp:
        body = "\n".join(["# Registered decisions (D3, D5)", "", (f"- Order size: ${params['order_notional_usd']} "
                          "(reported beside Tier-0's $10)"), f"- Pass rule: {params['pass_rule']}",
                          f"- Lock-box criteria: {params['lockbox_criteria']}", ""])
        exp.finish(params, report_body=body, conclusion="registered")
    return {"experiment": exp.experiment_id}


def registered_criteria(store: Store) -> dict[str, Any]:
    done = [r for r in store.experiments(command="criteria") if r["status"] == "done"]
    if not done:
        raise PermissionError("no registered order size and pass criteria: run `python -m backtest register-criteria` "
                              "first (design D3, D5)")
    return json.loads(done[-1]["params"])


# --- dispatch ---------------------------------------------------------------------------------------------------------

COMMANDS: Final[dict[str, Callable[..., dict[str, Any]]]] = {
    "ingest": cmd_ingest, "legacy": cmd_legacy, "leakage": cmd_leakage, "criteria": cmd_register_criteria,
    "walkforward": partial(runs.cmd_walkforward, write_report=_write_report),
    "leakage-wf": partial(runs.cmd_leakage_wf, write_report=_write_report),
    "run": partial(runs.cmd_run, write_report=_write_report),
    "ingest-core": partial(core_runs.cmd_ingest_core, write_report=_write_report),
    core_runs.REGISTRATION: partial(core_runs.cmd_register_core, write_report=_write_report),
    core_runs.LEAKAGE: partial(core_runs.cmd_leakage_core, write_report=_write_report),
    core_runs.RUN: partial(core_runs.cmd_run_core, write_report=_write_report),
    core_fx.COMMAND: partial(core_fx.cmd_fx_core, write_report=_write_report),
}


def _defaults(command: str, args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {"root": str(Path(args.data_root).resolve()), "symbols": list(SYMBOLS)}
    if command == "run":
        params["workers"] = args.workers  # read and removed before registering: not a result parameter
    if command == "legacy":
        params["tolerance"] = 1e-12
    if command == "leakage":
        params.update({"oracle_instants": 10_000, "seed": 1, "half_day": "2025-07-03", "holiday": "2025-07-04",
                       "fixture_month": "2025-06-01"})
    if command == "criteria":
        params.update({"order_notional_usd": str(args.order_notional), "pass_rule": args.pass_rule,
                       "lockbox_criteria": args.lockbox_criteria})
    if command in ("ingest-core", core_runs.REGISTRATION, core_runs.LEAKAGE, core_runs.RUN, core_fx.COMMAND):
        params = {"root": params["root"]}
    if command == core_fx.COMMAND:
        params.update(core_fx.params())
    if command == core_runs.REGISTRATION:
        from backtest.core_grid import registration_params
        params.update({"registration": registration_params(), "supersedes": args.supersedes})
    if command == core_runs.LEAKAGE:
        params.update(core_runs.leakage_defaults())
    if command == core_runs.RUN:
        params["registration"] = args.registration
    return params


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("risk_router").setLevel(logging.ERROR)  # a replay logs every OPEN and EXIT; the store has them
    parser = argparse.ArgumentParser(prog="python -m backtest", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "legacy", "leakage", "walkforward", "leakage-wf", "run"):
        p = sub.add_parser(name)
        p.add_argument("--allow-dirty", action="store_true", help="run on uncommitted code (cannot be cited)")
        if name == "run":
            p.add_argument("--workers", type=int, default=1,
                           help="replay processes (default 1: one core; results do not depend on it)")
    crit = sub.add_parser("register-criteria")
    crit.add_argument("--order-notional", type=Decimal, required=True, help="the real-money order size, USD (D3)")
    crit.add_argument("--pass-rule", default=DEFAULT_PASS_RULE)
    crit.add_argument("--lockbox-criteria", required=True, help="what the lock-box must show to confirm (D5, D8)")
    crit.add_argument("--allow-dirty", action="store_true")
    for name in ("ingest-core", "leakage-core", "fx-core"):
        sub.add_parser(name).add_argument("--allow-dirty", action="store_true")
    reg = sub.add_parser("register-core")
    reg.add_argument("--supersedes", default=None, help="the registration this one replaces (it still counts)")
    reg.add_argument("--allow-dirty", action="store_true")
    run_core = sub.add_parser("run-core")
    run_core.add_argument("registration", help="the core-registration experiment id")
    run_core.add_argument("--allow-dirty", action="store_true")
    sub.add_parser("experiments")
    rep = sub.add_parser("reproduce")
    rep.add_argument("experiment_id")
    replay = sub.add_parser("replay-experiment", help=argparse.SUPPRESS)
    replay.add_argument("experiment_id")
    replay.add_argument("--store", required=True)
    for p in sub.choices.values():
        p.add_argument("--data-root", default=str(REPO), help="where .cache/ and models/ live (default: this repo)")
        if "--store" not in p._option_string_actions:
            p.add_argument("--store", default=None, help="the store file (default: .cache/backtest/store.duckdb)")
    args = parser.parse_args(argv)

    store_path = Path(args.store) if args.store else _paths(Path(args.data_root))[2]
    if args.command == "reproduce":
        print(json.dumps(registry.reproduce(store_path, args.experiment_id, repo=REPO), indent=1))
        return 0
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with Store(store_path) as store:
        if args.command == "experiments":
            for r in store.experiments():
                print(f"{r['experiment_id']}  {r['command']:9} {r['status']:7} dirty={r['git_dirty']!s:5} "
                      f"{r['git_commit'][:8]}  {r['conclusion'] or ''}")
            return 0
        if args.command == "replay-experiment":
            original = store.experiment(args.experiment_id)
            result = COMMANDS[original["command"]](json.loads(original["params"]), store, allow_dirty=False,
                                                   parent_id=args.experiment_id)
        else:
            command = {"register-criteria": "criteria", "register-core": core_runs.REGISTRATION}.get(
                args.command, args.command)
            result = COMMANDS[command](_defaults(command, args), store, allow_dirty=args.allow_dirty, parent_id=None)
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
