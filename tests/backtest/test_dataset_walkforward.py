"""v2 labels (the live path's timing) and the walk-forward (P5, P6)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest
from conftest import HALF_DAY, make_calendar, ny, synthetic_samples

from backtest.calendar import Calendar
from backtest.data import MarketData
from backtest.dataset import Samples, build_v2
from backtest.events import Event
from backtest.lockbox import DEV_EVENTS_END
from backtest.walkforward import (
    THRESHOLD_GRID,
    Schedule,
    fit_fold,
    fold_bounds,
    prediction_rows,
    s2_nights,
    serving_models,
    threshold_rule,
)
from swarm.features import FEATURE_NAMES, completed_sessions, daily_features

SENT = np.array([0.6, 0.3, 0.1])


def event(n: int, at: datetime, symbols: tuple[str, ...] = ("AAPL",), headline: str = "AAPL news",
          story: str | None = None) -> Event:
    return Event(f"alpaca:{n}", str(n), "benzinga", at, None, at, headline, symbols, len(symbols), story)


def build(market: MarketData, events: list[Event]) -> tuple[Samples, dict[str, int]]:
    return build_v2(events, market, {e.headline: SENT for e in events})


def one(market: MarketData, at: datetime) -> Samples:
    samples, _ = build(market, [event(1, at)])
    assert len(samples) == 1
    return samples


def bar_start(market: MarketData, samples: Samples) -> datetime:
    return datetime.fromtimestamp(samples.entry_at[0], UTC)


# --- v2 labels ------------------------------------------------------------------------------

def test_session_event_enters_one_bar_after_the_decision_and_exits_at_the_close(market: MarketData) -> None:
    day = date(2025, 6, 30)
    s = one(market, ny(day, 10, 0, 30))                      # made 10:01:30, decided 10:02, fills from 10:03
    assert bar_start(market, s) == ny(day, 10, 3)
    series = market.minute["AAPL"]
    i = series.first_at_or_after(ny(day, 10, 3).timestamp())
    last = series.first_at_or_after(ny(day, 15, 59).timestamp())
    assert s.entry_price[0] == series.o[i] and s.exit_price[0] == series.c[last]
    assert s.fwd[0] == pytest.approx(series.c[last] / series.o[i] - 1, abs=0)
    assert s.y[0] == int(s.fwd[0] > 0.0025)
    assert s.resolved_at[0] == ny(day, 16).timestamp()
    assert (s.night[0], s.rolled[0], s.tradeable[0]) == (False, False, True)


def test_night_events_enter_at_the_schedulers_0932_bar(market: MarketData) -> None:
    s = one(market, ny(date(2025, 6, 30), 18))
    assert bar_start(market, s) == ny(date(2025, 7, 1), 9, 32) and s.night[0]
    weekend = one(market, ny(date(2025, 6, 28), 12))          # Saturday → Monday
    assert bar_start(market, weekend) == ny(date(2025, 6, 30), 9, 32)
    premarket = one(market, ny(date(2025, 7, 1), 9, 29, 30))
    assert bar_start(market, premarket) == ny(date(2025, 7, 1), 9, 32) and premarket.night[0]


def test_half_day_and_holiday(market: MarketData) -> None:
    during = one(market, ny(HALF_DAY, 11))
    assert during.resolved_at[0] == ny(HALF_DAY, 13).timestamp()        # the 13:00 close, not 16:00
    after = one(market, ny(HALF_DAY, 14))                                # after the half-day close
    assert after.night[0] and bar_start(market, after) == ny(date(2025, 7, 7), 9, 32)  # 07-04 is a holiday


def test_late_session_events_roll_and_are_not_tradeable(market: MarketData) -> None:
    day = date(2025, 6, 30)
    late = one(market, ny(day, 15, 58, 30))                  # made 15:59:30, fill bar 16:01: after the close
    assert (late.night[0], late.rolled[0], late.tradeable[0]) == (False, True, False)
    assert bar_start(market, late) == ny(date(2025, 7, 1), 9, 32)
    after_cutoff = one(market, ny(day, 15, 40))
    assert (after_cutoff.rolled[0], after_cutoff.tradeable[0]) == (False, False)


def test_features_are_as_of_made_at(market: MarketData) -> None:
    """16:14:00 + 60 s of latency = 16:15:00: that session's daily bar is published by then."""
    day = date(2025, 6, 30)
    s = one(market, ny(day, 16, 14))
    expected = daily_features(completed_sessions(market.daily["AAPL"], ny(day, 16, 15)))
    daily_cols = [FEATURE_NAMES.index(k) for k in expected]
    assert list(s.X[0, daily_cols]) == list(expected.values())
    assert completed_sessions(market.daily["AAPL"], ny(day, 16, 15))[-1].t.date() == day
    assert s.X[0, FEATURE_NAMES.index("regular_hours")] == 0.0  # published after the close


def test_filters_and_the_lockbox(market: MarketData) -> None:
    day = date(2025, 6, 30)
    events = [event(1, ny(day, 10), ("AAPL", "TSLA", "F", "GM")),     # roundup
              event(2, ny(day, 10), headline=""),
              event(3, ny(date(2025, 3, 4), 10)),                     # too little history
              event(4, DEV_EVENTS_END),                               # lock-box side
              event(5, ny(day, 10), ("AAPL", "TSLA", "F"))]           # two watched symbols
    samples, dropped = build(market, events)
    assert dropped == {"roundup": 1, "no_headline": 1, "history": 1, "not_development": 1}
    assert list(samples.symbol) == ["AAPL", "TSLA"]
    assert samples.X[0, FEATURE_NAMES.index("n_symbols")] == 3.0


# --- the walk-forward ----------------------------------------------------------------------------

def zero_cost(samples: Samples, idx: np.ndarray) -> np.ndarray:
    return np.zeros(len(idx))


def test_embargo_is_five_sessions_before_the_months_first_session() -> None:
    real = make_calendar(date(2025, 5, 1), date(2025, 7, 31))
    bounds = fold_bounds(date(2025, 7, 1), Schedule(), real)
    # sessions before Tue 07-01: 06-30, 27, 26, 25, 24 → the cut-off is 06-24's open
    assert bounds.embargo_cutoff == ny(date(2025, 6, 24), 9, 30)
    assert bounds.start == ny(date(2025, 7, 1), 0) and bounds.end == ny(date(2025, 8, 1), 0)


def test_fold_respects_p5_and_the_purge(long_calendar: Calendar) -> None:
    samples = synthetic_samples(long_calendar)
    schedule = Schedule()
    bounds = fold_bounds(date(2025, 3, 1), schedule, long_calendar)
    fold = fit_fold(samples, bounds, schedule, zero_cost)
    used = np.concatenate([fold.train, fold.calib])
    assert samples.resolved_at[used].max() < bounds.embargo_cutoff.timestamp()
    assert set(fold.train).isdisjoint(fold.calib)
    assert samples.made_at[fold.train].max() < samples.made_at[fold.calib].min() - schedule.purge.total_seconds()
    assert samples.resolved_at[fold.train].max() < samples.made_at[fold.calib].min()
    scored = samples.made_at[fold.scored]
    assert scored.min() >= bounds.start.timestamp() and scored.max() < bounds.end.timestamp()
    assert len(fold.calib) == pytest.approx(0.25 * len(used), rel=0.05)


def test_folds_are_deterministic(long_calendar: Calendar) -> None:
    samples = synthetic_samples(long_calendar)
    bounds = fold_bounds(date(2025, 6, 1), Schedule(), long_calendar)
    a, b = (fit_fold(samples, bounds, Schedule(), zero_cost) for _ in range(2))
    assert a.model_version == b.model_version and a.artifact_sha256 == b.artifact_sha256
    assert np.array_equal(a.prob_scored, b.prob_scored)


def test_threshold_rule_picks_the_lowest_qualifying_threshold() -> None:
    score = np.repeat([0.41, 0.45, 0.51], [40, 40, 40])
    ret = np.repeat([-0.004, 0.001, 0.004], [40, 40, 40])
    units = {"score": score, "ret": ret, "symbol": np.asarray(["A"] * 120), "session": np.arange(120)}
    chosen, grid = threshold_rule(units, np.full(120, 0.0009), per_symbol_day_first=False)
    # >= 0.40: all 120, mean +0.033% < 0.09% of cost; >= 0.42: 80 symbol-days, mean +0.25% > cost
    assert chosen == 0.42
    assert [g["threshold"] for g in grid] == list(THRESHOLD_GRID) and not any(g["selectable"] for g in grid)
    none, _ = threshold_rule(units, np.full(120, 0.01), per_symbol_day_first=False)
    assert none is None


def test_s3_counts_a_symbol_days_first_event_only() -> None:
    units = {"score": np.full(60, 0.5), "ret": np.full(60, 0.01), "symbol": np.asarray(["A"] * 60),
             "session": np.repeat(np.arange(20), 3), "made_at": np.arange(60.0)}
    _, grid = threshold_rule(units, np.zeros(60), per_symbol_day_first=True)
    assert grid[0]["symbol_days"] == 20  # 60 events, 20 symbol-days


def test_s2_scores_a_symbol_night_by_its_mean_and_collapses_stories(long_calendar: Calendar) -> None:
    samples = synthetic_samples(long_calendar, n=200)
    samples.night[:] = True
    samples.session[:4] = 5
    samples.symbol[:4] = "AAPL"
    samples.story_id[:4] = ["story:1", "story:1", "story:2", ""]
    samples.made_at[:4] = samples.made_at[0] + np.arange(4.0)  # in index order
    prob = np.linspace(0.3, 0.5, 200)
    idx = np.arange(200)
    full = s2_nights(samples, idx, prob, collapsed=False)
    k = int(np.flatnonzero((full["symbol"] == "AAPL") & (full["session"] == 5))[0])
    assert full["score"][k] == pytest.approx(prob[:4].mean())
    collapsed = s2_nights(samples, idx, prob, collapsed=True)
    k = int(np.flatnonzero((collapsed["symbol"] == "AAPL") & (collapsed["session"] == 5))[0])
    assert collapsed["score"][k] == pytest.approx(prob[[0, 2, 3]].mean())


def test_serving_and_stored_predictions(long_calendar: Calendar) -> None:
    samples = synthetic_samples(long_calendar)
    fold = fit_fold(samples, fold_bounds(date(2025, 6, 1), Schedule(), long_calendar), Schedule(), zero_cost)
    serving = serving_models(fold)
    table = fold.move_tables["S3"]
    assert serving["S3"].expected_move(0.0) == table["mean_return_pct"][0]
    assert serving["S3"].expected_move(1.0) == table["mean_return_pct"][-1]
    rows = prediction_rows(samples, [fold], experiment_id="x1")
    assert len(rows) == len(fold.scored) == len({r["prediction_id"] for r in rows})
    first = rows[0]
    assert first["inputs"]["rsi14"] == samples.X[fold.scored[0], FEATURE_NAMES.index("rsi14")]
    assert first["model_version"] == fold.model_version and first["horizon"] == "close"
    # Ids are stable within an experiment and distinct across experiments (clean re-runs, reproductions).
    assert [r["prediction_id"] for r in prediction_rows(samples, [fold], experiment_id="x1")] == \
        [r["prediction_id"] for r in rows]
    assert not {r["prediction_id"] for r in prediction_rows(samples, [fold], experiment_id="x2")} & \
        {r["prediction_id"] for r in rows}
