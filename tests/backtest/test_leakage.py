"""Leakage and consistency checks L1-L12 (design §9), each beside its broken twin.

Every check must pass on the real configuration and fail on its twin, which
breaks its rule on purpose: a check that cannot fail proves nothing. These
run on synthetic data; ``real_data`` tests repeat the ones that need the
cached market data and are skipped when it is absent.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    HALF_DAY,
    HOLIDAY,
    SYMBOLS,
    daily_bars,
    make_calendar,
    minute_arrays,
    ny,
    synthetic_samples,
)
from test_sim_engine import Scripted, custom_market, flat, sig

import tier0
from backtest import leakage, walkforward
from backtest.calendar import Calendar
from backtest.costs import LEVELS, FeeTable
from backtest.data import MarketData, regular_series
from backtest.dataset import build_v2
from backtest.engine import Engine, EngineParams
from backtest.events import Event
from backtest.walkforward import Schedule, fit_fold, fold_bounds
from swarm.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
REAL = (ROOT / ".cache" / "backtest" / "calendar.parquet").exists() and (ROOT / ".cache" / "training").exists()
real_data = pytest.mark.skipif(not REAL, reason="needs the fetched caches under .cache/")


def zero_cost(samples, idx) -> np.ndarray:
    return np.zeros(len(idx))


THREE_MONTHS = Schedule(first_month=date(2025, 6, 1), last_month=date(2025, 8, 1))


@pytest.fixture(scope="module")
def samples(long_calendar: Calendar):
    return synthetic_samples(long_calendar, n=30_000, seed=11)


@pytest.fixture(scope="module")
def folds(samples, long_calendar: Calendar):
    return walkforward.run(samples, THREE_MONTHS, long_calendar, zero_cost)


# --- L1 future canary ---------------------------------------------------------------------------

def test_l1_a_canary_known_30_days_later_does_not_help(samples, long_calendar, folds) -> None:
    check = leakage.future_canary(samples, THREE_MONTHS, long_calendar, zero_cost, base_folds=folds)
    assert check.passed, check.detail


def test_l1_twin_a_canary_joined_on_its_session_is_caught(samples, long_calendar, folds) -> None:
    check = leakage.future_canary(samples, THREE_MONTHS, long_calendar, zero_cost, join="session", base_folds=folds)
    assert not check.passed and check.detail["auc_with_canary"] > 0.9, check.detail


# --- L2 label shuffling --------------------------------------------------------------------------

def test_l2_shuffled_labels_leave_no_auc(samples, long_calendar) -> None:
    check = leakage.shuffled_auc(samples, THREE_MONTHS, long_calendar, zero_cost, seeds=range(5), bootstrap=300)
    assert check.passed, check.detail


def test_l2_shuffling_keeps_symbol_days_together(samples) -> None:
    shuffled = leakage.shuffle_labels(samples, seed=1)
    assert sorted(shuffled.y) != list(shuffled.y)  # not trivially sorted
    assert abs(shuffled.y.mean() - samples.y.mean()) < 0.02
    key = np.char.add(samples.symbol.astype(str), samples.session.astype(str))
    for k in np.unique(key)[:200]:
        block = shuffled.fwd[key == k]
        assert len(set(np.round(block, 12))) <= len(block)  # values come from one source block


SIX_MONTHS = Schedule(first_month=date(2025, 1, 1), last_month=date(2025, 6, 1))
EIGHT = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD")


@pytest.fixture(scope="module")
def shuffled_screen(long_calendar):
    """Eight symbols whose unconditional night return is negative: only look-ahead makes S2 positive."""
    data = leakage.shuffle_labels(synthetic_samples(long_calendar, n=40_000, seed=5, up=0.004, down=-0.004,
                                                    symbols=EIGHT), seed=2)
    folds = walkforward.run(data, SIX_MONTHS, long_calendar, zero_cost)
    returns: dict[tuple[str, int], list[float]] = {}
    for sym, s, r in zip(data.symbol, data.session, data.fwd, strict=True):
        returns.setdefault((str(sym), int(s)), []).append(float(r))
    return data, folds, {k: float(np.mean(v)) for k, v in returns.items()}


def test_l2_s2_on_shuffled_labels_stays_inside_the_placebo_band(shuffled_screen) -> None:
    data, folds, returns = shuffled_screen
    assert leakage.s2_screen(data, folds, returns).passed


def test_l2_twin_thresholds_chosen_on_the_scored_month_beat_the_band(shuffled_screen) -> None:
    data, folds, returns = shuffled_screen
    check = leakage.s2_screen(data, leakage.leaky_folds(data, folds, zero_cost), returns)
    assert not check.passed, check.detail


# --- L3 no bar after the decision ----------------------------------------------------------------------

def test_l3_point_in_time_view_matches_brute_force(market: MarketData) -> None:
    check = leakage.pit_oracle(market, n=3000, seed=4)
    assert check.passed, check.detail


def _replay(latency_bars: int) -> None:
    engine = Engine(custom_market(flat), Scripted([sig(1, ny(date(2025, 6, 30), 10))]), LEVELS["central"],
                    FeeTable.load(), EngineParams(latency_bars=latency_bars))
    asyncio.run(engine.run([engine.market.calendar.session(date(2025, 6, 30))]))


def test_l3_assertions_stay_silent_in_a_replay() -> None:
    assert leakage.assertions_silent(lambda: _replay(1)).passed


def test_l3_twin_zero_latency_trips_the_p4_assertion() -> None:
    check = leakage.assertions_silent(lambda: _replay(0))
    assert not check.passed and "FillTimingError" in check.detail["assertion"]


# --- L5 training cut-off, L6 story straddling -------------------------------------------------------------

def test_l5_every_fold_respects_the_embargo(samples, long_calendar, folds) -> None:
    assert leakage.training_cutoff(samples, folds, long_calendar).passed


def test_l5_twin_an_embargo_of_minus_one_session_is_caught(samples, long_calendar) -> None:
    broken = replace(THREE_MONTHS, embargo_sessions=-1)
    leaky = [fit_fold(samples, fold_bounds(m, broken, long_calendar), broken, zero_cost) for m in broken.months()]
    check = leakage.training_cutoff(samples, leaky, long_calendar)
    assert not check.passed and len(check.detail["violations"]) == 3


def _straddling_story(samples, calendar: Calendar):
    """One story from the last session of May into the first of June."""
    last_may, first_june = calendar.session(date(2025, 5, 30)), calendar.session(date(2025, 6, 2))
    data = replace(samples, story_id=samples.story_id.copy(), made_at=samples.made_at.copy(),
                   resolved_at=samples.resolved_at.copy(), night=samples.night.copy())
    for i, session in ((0, last_may), (1, first_june)):   # both in session, resolving at their close
        data.story_id[i] = "story:boundary"
        data.made_at[i] = (session.open_at + timedelta(hours=1)).timestamp()
        data.resolved_at[i] = session.close_at.timestamp()
        data.night[i] = False
    return data


def test_l6_no_story_crosses_from_training_into_its_scored_month(samples, long_calendar) -> None:
    data = _straddling_story(samples, long_calendar)
    schedule = replace(THREE_MONTHS, last_month=date(2025, 6, 1))
    folds = [fit_fold(data, fold_bounds(m, schedule, long_calendar), schedule, zero_cost) for m in schedule.months()]
    assert leakage.story_straddle(data, folds).passed


def test_l6_twin_without_an_embargo_a_story_straddles(samples, long_calendar) -> None:
    data = _straddling_story(samples, long_calendar)
    schedule = replace(THREE_MONTHS, last_month=date(2025, 6, 1), embargo_sessions=0)
    folds = [fit_fold(data, fold_bounds(m, schedule, long_calendar), schedule, zero_cost) for m in schedule.months()]
    check = leakage.story_straddle(data, folds)
    assert not check.passed and check.detail["straddles"][0]["stories"] == ["story:boundary"]


# --- L7 lock-box ----------------------------------------------------------------------------------------------

def _lockbox_market(unguarded: bool) -> MarketData:
    """Synthetic sessions on both sides of the lock-box line (2026-09-21)."""
    cal = make_calendar(date(2026, 8, 3), date(2026, 9, 30))
    minute, daily, raw = {}, {}, {}
    for k, s in enumerate(SYMBOLS):
        a = minute_arrays(s, cal, date(2026, 9, 1), seed=k + 1)
        minute[s] = regular_series(s, 60, a["t"].astype(np.int64), a["o"], a["h"], a["l"], a["c"], a["v"], cal)
        daily[s], raw[s] = daily_bars(s, cal, seed=k + 10)
    return (leakage.UnguardedMarketData if unguarded else MarketData)(cal, minute, daily, raw)


def test_l7_lockbox_data_is_unreadable() -> None:
    check = leakage.lockbox_guard(_lockbox_market(unguarded=False))
    assert check.passed, check.detail


def test_l7_twin_with_the_guard_off_the_lockbox_leaks() -> None:
    check = leakage.lockbox_guard(_lockbox_market(unguarded=True))
    assert not check.passed and len(check.detail["leaks"]) >= 3


# --- L8 feature parity ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stored_predictions(market: MarketData) -> list[dict]:
    rng = np.random.default_rng(0)
    events = []
    for n in range(300):
        day = market.calendar.sessions_between(date(2025, 6, 2), date(2025, 7, 30))[int(rng.integers(0, 40))]
        at = day.open_at + timedelta(minutes=int(rng.integers(-600, 700)))
        events.append(Event(f"alpaca:{n}", str(n), "benzinga", at, None, at, f"news {n}", ("AAPL",), 1))
    samples, _ = build_v2(events, market, {e.headline: np.array([0.5, 0.3, 0.2]) for e in events})
    return [{"prediction_id": f"p{i}", "symbol": str(samples.symbol[i]),
             "made_at": datetime.fromtimestamp(samples.made_at[i], ny(HOLIDAY, 0).tzinfo),
             "inputs": json.dumps(dict(zip(FEATURE_NAMES, map(float, samples.X[i]), strict=True)))}
            for i in range(len(samples))]


def test_l8_stored_inputs_equal_an_independent_recomputation(stored_predictions, market) -> None:
    check = leakage.feature_parity(stored_predictions, market, n=200)
    assert check.passed and check.detail["checked"] == 200


def test_l8_twin_features_cut_off_a_session_late_are_caught(stored_predictions, market) -> None:
    check = leakage.feature_parity(stored_predictions, market, n=200, late=True)
    assert not check.passed and check.detail["mismatches"] > 150


# --- L9 outcome isolation --------------------------------------------------------------------------------------

def test_l9_nothing_that_fits_or_chooses_reads_outcomes() -> None:
    check = leakage.outcome_isolation()
    assert check.passed and "backtest.walkforward" in check.detail["reached"]


def test_l9_twin_an_added_import_is_caught() -> None:
    source = (ROOT / "backtest" / "strategies.py").read_text() + "\nfrom backtest.outcomes import resolve\n"
    assert not leakage.outcome_isolation(overrides={"backtest.strategies": source}).passed
    indirect = (ROOT / "backtest" / "costs.py").read_text() + "\nfrom . import outcomes\n"
    assert not leakage.outcome_isolation(overrides={"backtest.costs": indirect}).passed


# --- L10 one engine ----------------------------------------------------------------------------------------------

def test_l10_the_engine_runs_the_production_router_and_limits() -> None:
    check = leakage.one_engine()
    assert check.passed, check.detail["problems"]
    assert set(check.detail["source_sha256"]) >= {"tier0.py", "risk_router/policy.py", "risk_router/gatekeeper.py"}


def test_l10_twin_a_monkeypatched_limit_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tier0, "MAX_NOTIONAL_USD", Decimal("1000"))
    check = leakage.one_engine()
    assert not check.passed and any("MAX_NOTIONAL_USD" in p for p in check.detail["problems"])


# --- L11 calendar -------------------------------------------------------------------------------------------------

def _calendar_run(calendar: Calendar | None) -> tuple:
    truth = custom_market(flat)
    market = truth if calendar is None else _rebuild(truth, calendar)
    signals = [sig(1, ny(HALF_DAY, 12)), sig(2, ny(HALF_DAY, 12, 31), "TSLA"), sig(3, ny(HOLIDAY, 11))]
    days = [s for s in market.calendar.sessions if date(2025, 7, 3) <= s.date <= date(2025, 7, 7)]
    engine = Engine(market, Scripted(signals), LEVELS["central"], FeeTable.load())
    return asyncio.run(engine.run(days)), truth.calendar


def _rebuild(truth: MarketData, calendar: Calendar) -> MarketData:
    """The same bars filed under another calendar."""
    minute = {s: regular_series(s, 60, b.start, b.o, b.h, b.l, b.c, b.v, calendar) for s, b in truth.minute.items()}
    return MarketData(calendar, minute, truth.daily, {s: {} for s in truth.minute})


def test_l11_half_days_and_holidays_follow_the_exchange_calendar() -> None:
    result, truth = _calendar_run(None)
    check = leakage.calendar_rules(result, truth, half_day=HALF_DAY, late_signal="sig-00000002")
    assert check.passed, check.detail


def test_l11_twin_a_weekday_rule_calendar_is_caught() -> None:
    naive = Calendar.weekday_rule(date(2025, 3, 3), date(2025, 7, 31))
    result, truth = _calendar_run(naive)
    check = leakage.calendar_rules(result, truth, half_day=HALF_DAY, late_signal="sig-00000002")
    assert not check.passed and len(check.detail["problems"]) >= 2, check.detail


# --- L12 minute skipping ---------------------------------------------------------------------------------------------

def _month_with_a_stop_and_a_target():
    stop_day, target_day = date(2025, 6, 11), date(2025, 6, 18)

    def path(day: date, k: int) -> tuple[float, float, float, float]:
        if day == stop_day and k == 100:
            return 100.0, 100.0, 94.0, 99.0
        if day == target_day and k >= 150:
            return 110.6, 110.8, 110.5, 110.7
        return flat(day, k)

    days = tuple(d for d in (date(2025, 6, 2) + timedelta(days=i) for i in range(28)) if d.weekday() < 5)
    market = custom_market(path, days=days)
    signals = [sig(n, ny(d, 10)) for n, d in enumerate(days, start=1)]
    return market, Scripted(signals), [market.calendar.session(d) for d in days], (stop_day, target_day)


def test_l12_skipping_idle_minutes_changes_nothing() -> None:
    market, strategy, sessions, _ = _month_with_a_stop_and_a_target()
    check = leakage.skipping_invariance(market, strategy, sessions)
    assert check.passed, check.detail
    assert check.detail["requests"][1] < check.detail["requests"][0] / 3   # and it is what makes replays fast


def test_l12_twin_skipping_the_exit_passes_is_caught() -> None:
    market, strategy, sessions, _ = _month_with_a_stop_and_a_target()
    assert not leakage.skipping_invariance(market, strategy, sessions, twin=True).passed


# --- the revision diagnostic ------------------------------------------------------------------------------------------

def test_revision_diagnostic_buckets_by_the_publication_to_revision_gap(samples, folds) -> None:
    idx = np.concatenate([f.scored for f in folds])
    updated = {}
    for k, i in enumerate(idx):
        published = datetime.fromtimestamp(samples.published_at[i], ny(HOLIDAY, 0).tzinfo)
        updated[str(samples.event_id[i])] = published + timedelta(seconds=[0, 30, 600, 7200, 200000][k % 5])
    rows = leakage.revision_auc(samples, folds, updated)
    assert [r["bucket"] for r in rows] == [b[0] for b in leakage.REVISION_BUCKETS]
    assert sum(r["n"] for r in rows) == len(idx) and all(r["auc"] is not None for r in rows)


# --- on the real caches --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_meta() -> dict:
    return json.loads((ROOT / "models" / "return_model" / "meta.json").read_text())


@real_data
def test_l4_reproduces_report_md_exactly(real_meta) -> None:
    check, _ = leakage.reproduce_report(real_meta, real_meta["symbols"], ROOT / ".cache" / "training")
    assert check.passed, check.detail


@real_data
@pytest.mark.parametrize("twin", [{"purge": timedelta(0)}, {"entry": "bisect_right"}])
def test_l4_twins_do_not_reproduce(real_meta, twin) -> None:
    check, _ = leakage.reproduce_report(real_meta, real_meta["symbols"], ROOT / ".cache" / "training", **twin)
    assert not check.passed and check.detail["n_differences"] > 0


@pytest.fixture(scope="module")
def real_market() -> MarketData:
    from backtest.fetch import Paths
    return MarketData.load(Paths(ROOT / ".cache" / "backtest"), symbols=tuple(json.loads(
        (ROOT / "models" / "return_model" / "meta.json").read_text())["symbols"]))


@pytest.fixture(scope="module")
def legacy(real_market, real_meta):
    from backtest.data import INTRADAY_CACHE, load_bar_cache
    from backtest.dataset import build_legacy
    from backtest.lockbox import LOCKBOX_SESSIONS_FROM
    samples = build_legacy(real_meta["symbols"], ROOT / ".cache" / "training")
    intraday = {s: [b for b in bars if b.t.astimezone(ny(HOLIDAY, 0).tzinfo).date() < LOCKBOX_SESSIONS_FROM]
                for s, bars in load_bar_cache(ROOT / ".cache" / "training" / INTRADAY_CACHE, real_meta["symbols"]).items()}
    return samples, leakage.legacy_series(intraday, real_market.calendar)


@real_data
def test_l4b_the_engine_agrees_with_every_legacy_label(real_market, legacy) -> None:
    samples, series = legacy
    check = leakage.engine_agrees_with_labels(samples, real_market, series)
    assert check.passed, check.detail


@real_data
def test_l4b_twin_filling_on_the_bar_that_contains_the_publication_is_caught(real_market, legacy) -> None:
    samples, series = legacy
    check = leakage.engine_agrees_with_labels(samples, real_market, series, twin=True)
    assert not check.passed and check.detail["over_tolerance"] > 1000


@real_data
def test_l3_real_point_in_time_view_matches_brute_force(real_market) -> None:
    assert leakage.pit_oracle(real_market, n=10_000, seed=1).passed


@real_data
def test_l7_real_lockbox_sessions_are_quarantined(real_market) -> None:
    check = leakage.lockbox_guard(real_market)
    assert check.passed, check.detail
