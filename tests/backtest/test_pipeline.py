"""The real-run pipeline on synthetic data, where every answer is known in advance."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_UP, Decimal

import numpy as np
import pytest
from conftest import SYMBOLS, ny
from test_sim_engine import custom_market, flat, replay, sig
from test_strategies import OPEN, fold_over, samples  # noqa: F401 — the fixture

from backtest import pipeline, runs
from backtest.costs import LEVELS, FeeTable
from backtest.pipeline import Plan, Series, Task, daily_pnl_at
from backtest.store import Store

D = Decimal
DAY = date(2025, 6, 30)
CENT = D("0.01")


# --- scaling to the registered order size ------------------------------------------------------------

def test_rescaling_reproduces_the_replay_and_recomputes_costs_at_size() -> None:
    fees = FeeTable.load()
    for level_name in ("pessimistic", "central"):
        level = LEVELS[level_name]
        result = replay(custom_market(flat), [sig(1, ny(DAY, 10))], level_name)
        assert daily_pnl_at(result.rows, level, fees, D(1)) == {DAY: result.days[0].pnl}   # k = 1 is the replay
        buy = next(r for r in result.rows if r["state"] == "filled")
        sell = next(r for r in result.rows if r["state"] == "exit_filled")
        qty = buy["qty"] * 100
        # an independent derivation of the $1,000 day
        cash = -(qty * buy["price"]).quantize(CENT, ROUND_UP) + (qty * sell["price"]).quantize(CENT, ROUND_DOWN)
        sale = fees.exact(DAY, "sell", qty, sell["price"])
        purchase = fees.exact(DAY, "buy", qty, buy["price"])
        if level_name == "pessimistic":   # each fee type rounded up on each order
            charged = sum(v.quantize(CENT, ROUND_CEILING) for v in (*sale.values(), *purchase.values()))
        else:                             # each fee type summed over the day, then rounded up
            totals = {k: sale.get(k, D(0)) + purchase.get(k, D(0)) for k in set(sale) | set(purchase)}
            charged = sum(v.quantize(CENT, ROUND_CEILING) for v in totals.values())
        assert daily_pnl_at(result.rows, level, fees, D(100)) == {DAY: cash - charged}
    assert cash - charged == D("-0.73")                      # central, $1,000: fees are 2 cents, not 3


def test_router_series_divides_by_the_scaled_capital_base() -> None:
    result = replay(custom_market(flat), [sig(1, ny(DAY, 10))], "pessimistic")
    s = pipeline.router_series("S2", "pessimistic", D(1000), [DAY], {"rows": result.rows}, FeeTable.load())
    assert s.pnl == [D("-3.92")] and s.returns[0] == pytest.approx(-3.92 / 5000)
    tier0 = pipeline.router_series("S2", "pessimistic", D(10), [DAY], {"rows": result.rows}, FeeTable.load())
    assert tier0.pnl == [D("-0.07")] and tier0.returns[0] == pytest.approx(-0.07 / 50)


# --- the registered pass rule ----------------------------------------------------------------------------

def _dates() -> list[date]:
    d, out = date(2025, 1, 2), []
    while d <= date(2026, 9, 18):
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _series(name: str, returns: np.ndarray, nights: int = 0) -> Series:
    dates = _dates()
    rows = [{"state": "filled", "symbol": SYMBOLS[k % 2],
             "detail": {"bar_start": datetime.combine(dates[k // 2], datetime.min.time(), UTC).isoformat()}}
            for k in range(nights)]
    return Series(name, "central", D(1000), dates, returns, [D(0)] * len(dates), rows)


def test_pass_conditions_give_the_known_answers() -> None:
    rng = np.random.default_rng(0)
    b1 = _series("B1", rng.normal(0, 0.01, len(_dates())))
    good = pipeline.pass_conditions(_series("S2", b1.returns + 0.002, nights=300), b1, min_symbol_nights=300,
                                    resamples=500, seed=0)
    assert good["passes"] and good["mean_daily_excess"]["estimate"] == pytest.approx(0.002)
    assert list(good["half_years"]) == ["2025H1", "2025H2", "2026H1", "2026H2"]

    too_few = pipeline.pass_conditions(_series("S2", b1.returns + 0.002, nights=299), b1, min_symbol_nights=300,
                                       resamples=500, seed=0)
    assert not too_few["passes"] and not too_few["checks"]["min_symbol_nights"]

    one_bad_half = b1.returns + np.array([-0.001 if d >= date(2026, 1, 1) and d.month <= 6 else 0.003
                                          for d in _dates()])
    mixed = pipeline.pass_conditions(_series("S2", one_bad_half, nights=400), b1, min_symbol_nights=300,
                                     resamples=500, seed=0)
    assert mixed["checks"]["interval_above_zero"] and not mixed["checks"]["positive_each_half_year"]


def test_assess_needs_both_variants_at_central_and_never_passes_on_optimistic_alone() -> None:
    rng = np.random.default_rng(1)
    b1 = rng.normal(0, 0.01, len(_dates()))
    plan = Plan(levels=("optimistic", "central", "pessimistic"), resamples=300)
    def build(central_s2c_ok: bool, central_ok: bool = True) -> dict:
        series = {}
        for level in plan.levels:
            edge = 0.002 if (level == "optimistic" or central_ok) else -0.0005
            series[("B1", level, "1000")] = _series("B1", b1)
            series[("S2", level, "1000")] = _series("S2", b1 + edge, nights=400)
            s2c_edge = edge if (central_s2c_ok or level != "central") else -0.0005
            series[("S2_collapsed", level, "1000")] = _series("S2_collapsed", b1 + s2c_edge, nights=400)
        return pipeline.assess(series, plan, {"order_notional_usd": "1000"})

    assert build(True)["development_passes"]
    assert build(True)["verdict"].startswith("passes on development data")
    assert not build(False)["development_passes"]                 # collapsing duplicates breaks it: fail
    only_optimistic = build(True, central_ok=False)
    assert not only_optimistic["development_passes"] and only_optimistic["holds_only_at_optimistic"]


# --- replays in process, and the whole assembly -----------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_run(market, samples):  # noqa: F811 — the imported fixture
    folds = {str(size): [fold_over(samples, OPEN)] for size in (D(10), D(1000))}
    plan = Plan(levels=("central",), b3_seeds=(0, 1), workers=1, resamples=200,
                first_session=date(2025, 6, 30), last_session=date(2025, 7, 3))
    dates = [s.date for s in market.calendar.sessions_between(plan.first_session, plan.last_session)]
    common = {"root": "unused", "symbols": SYMBOLS, "samples": samples, "folds": folds, "prediction_ids": {},
              "dates": dates, "experiment_id": "x-test", "plan": plan, "market": market}
    first = [Task(s, "central", size) for s in pipeline.STRATEGY_KINDS for size in plan.sizes]
    first += [Task("B1", "central", None, plan.b1_seed), Task("B2", "central", None)]
    results = pipeline.run_tasks(first, **common)
    b3 = [Task("B3", "central", size, seed, tuple(sorted(results[f"S2:central:${size}"]["entries"].items())))
          for size in plan.sizes for seed in plan.b3_seeds]
    results.update(pipeline.run_tasks(b3, **common))
    return results, plan, dates


def test_replays_return_rows_and_placebos_return_their_scaled_pnl(pipeline_run) -> None:
    results, _, _ = pipeline_run
    assert set(results) >= {"S2:central:$10", "S2:central:$1000", "B1:central:-:seed1", "B2:central:-",
                            "B3:central:$1000:seed0"}
    assert "rows" in results["S2:central:$10"] and "pnl" in results["B3:central:$1000:seed1"]
    assert sum(results["S2:central:$10"]["entries"].values()) >= 1          # the stub fold trades AAPL's night


def test_assembly_writes_the_store_and_the_report(pipeline_run, market) -> None:
    results, plan, dates = pipeline_run
    with Store() as store:
        store.register({"experiment_id": "x-test", "created_at": datetime(2026, 9, 26, tzinfo=UTC),
                        "hypothesis": "h", "command": "run", "status": "running", "lockbox": False,
                        "git_commit": "c", "git_dirty": False, "data_hash": "d", "manifest": {}, "params": {},
                        "seeds": {}, "environment": {}})
        auc = {"estimate": 0.52, "low": 0.5, "high": 0.54}
        wf = {"experiment_id": "x-wf", "metrics": json.dumps({"auc": {"all": auc, "night": auc, "tradeable": auc},
                                                              "folds_trading": {"10": {"S2": 1}}})}
        criteria = {"order_notional_usd": "1000", "pass_rule": "S2 must beat B1."}
        summary, body = runs._assemble(results, plan, market, dates, SYMBOLS, criteria, "x-criteria", wf, store,
                                       "x-test")
        assert len(store.runs("x-test")) == len(results)
        assert store.trade_events("x-test:S2:central:$10")
    assert summary["assessment"]["verdict"] == "fails"                         # a 4-day toy cannot reach 300 nights
    assert body.index("Selection bias") < body.index("## Verdict against the registered pass rule")
    for text in ("> S2 must beat B1.", "## Every strategy at $10 an order", "## Every strategy at $1000 an order",
                 "| S0 | central |", "| S2 | central |", "In B3"):
        assert text in body
