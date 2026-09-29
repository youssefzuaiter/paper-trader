"""Reports and the statistics behind them."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest
from conftest import ny

from backtest import leakage, metrics
from backtest.costs import LEVELS, FeeTable
from backtest.engine import Engine
from backtest.report import SELECTION_BIAS, document, reproduction_report, strategy_report, summarise


def test_stationary_bootstrap_resamples_whole_runs_of_days() -> None:
    idx = metrics.stationary_indices(1000, 5, np.random.default_rng(0))
    steps = np.diff(idx)
    assert set(np.unique(idx)) <= set(range(1000))
    assert 0.7 < np.mean((steps == 1) | (steps == -999)) < 0.9     # continuing a block 4 times in 5
    interval = metrics.block_bootstrap(np.random.default_rng(1).normal(0.001, 0.01, 500), np.mean, n=2000, seed=3)
    assert interval.low < interval.estimate < interval.high and interval.seed == 3


def test_bootstrap_auc_resamples_days_not_samples() -> None:
    rng = np.random.default_rng(2)
    day = np.repeat(np.arange(200), 10)
    y = np.repeat(rng.integers(0, 2, 200), 10)                       # a day's duplicated articles share
    prob = np.repeat(rng.random(200), 10) + rng.normal(0, 1e-3, 2000)   # their label and nearly their score
    by_day = metrics.bootstrap_auc(y, prob, day, n=500, seed=1)
    naive = metrics.bootstrap_auc(y, prob, np.arange(2000), n=500, seed=1, mean_block=1)  # samples as independent
    assert by_day.high - by_day.low > 1.5 * (naive.high - naive.low)       # 200 bets, not 2,000


def test_drawdown_and_its_duration() -> None:
    depth, days = metrics.max_drawdown(np.array([0.01, -0.02, -0.01, 0.005, 0.03, -0.001]))
    assert depth == pytest.approx(-0.03) and days == 3


def test_deflated_sharpe_falls_as_more_configurations_are_tried() -> None:
    few = metrics.deflated_sharpe(0.1, [0.02, 0.05], 500, 0.0, 3.0)
    many = metrics.deflated_sharpe(0.1, list(np.linspace(-0.05, 0.08, 50)), 500, 0.0, 3.0)
    assert many < few


def test_reproduction_report_compares_every_number() -> None:
    meta = {"version": "gbm-20260918-c97181eb", "n_samples": 3,
            "metrics": {"segments": {"train": {"n": 1, "span": "x", "up_rate": 0.4}},
                        "test": {k: 0.5 for k in ("auc_model", "auc_sentiment_only",
                                                  "auc_model_one_article_per_symbol_day", "auc_model_tradeable",
                                                  "auc_model_tradeable_one_per_symbol_day", "brier_model",
                                                  "brier_base_rate", "log_loss_model", "log_loss_base_rate")}},
            "move_table": {"edges": [0.4], "mean_return_pct": [0.1, 0.2]}}
    ours = {"metrics": meta["metrics"], "move_table": {"edges": [0.4], "mean_return_pct": [0.1, 0.25]},
            "n_samples": 3, "digest": "c97181eb"}
    check = leakage.Check("L4", False, {"n_differences": 1, "purge_s": 345600.0, "entry": "bisect_left"})
    body, counts = reproduction_report(ours, meta, [check], {"notes": ["a finding"]})
    assert counts == {"numbers": 17, "exact": 16}   # 12 metrics, 3 move-table entries, n_samples, digest
    assert body.index(SELECTION_BIAS) < body.index("## Every number")
    assert "| `move_table/mean_return_pct[1]` | 0.2 | 0.25 | ✗ |" in body and "- a finding" in body


def test_strategy_report_on_a_synthetic_replay() -> None:
    market, strategy, sessions = leakage.fixture_month(leakage.fixture_month.__globals__["Calendar"].weekday_rule(
        date(2025, 6, 1), date(2025, 6, 30)), date(2025, 6, 1))
    runs = {level: asyncio.run(Engine(market, strategy, LEVELS[level], FeeTable.load()).run(sessions))
            for level in ("optimistic", "central", "pessimistic")}
    s0 = {s.date: 0.0005 for s in sessions}
    summaries = [summarise(runs[level], Decimal("50"), {"S0": s0}, placebo_totals=[-0.01, 0.0, 0.01], resamples=500)
                 for level in runs]
    assert summaries[1].bets == {"days_with_a_position": len(sessions), "symbol_days": len(sessions),
                                 "trades": len(sessions)}
    assert summaries[2].costs["fees"] > summaries[1].costs["fees"]        # per-order rounding at pessimistic
    body = strategy_report(summaries, configuration=(2, 5), trial_sharpes=[0.1, -0.2, 0.3])
    assert body.startswith(f"# {strategy.name}") and SELECTION_BIAS in body and "Configuration 2 of 5" in body
    for section in ("Excess over the baselines", "Independent bets", "pessimistic", "Percentile in B3"):
        assert section in body
    assert document({"experiment": "x"}, body).startswith("<!-- experiment: x -->")


def test_timestamps_in_the_fixture_are_in_new_york() -> None:
    assert ny(date(2025, 6, 2), 9, 30) + timedelta(minutes=1) == ny(date(2025, 6, 2), 9, 31)
