"""Strategy code on a synthetic market (no strategy is run on real data here)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest
from conftest import SPLIT, SYMBOLS, make_calendar, ny

from backtest.costs import LEVELS, FeeTable
from backtest.data import MarketData
from backtest.dataset import Samples, build_v2
from backtest.engine import Engine
from backtest.events import Event
from backtest.strategies import B1, B2, B3, CLOSED, S2, S3, raw_atr, run_hold
from backtest.walkforward import FoldBounds, FoldResult, prediction_rows
from swarm.features import FEATURE_NAMES, daily_features
from swarm.quant import atr_usd

D = Decimal
DAY, NEXT = date(2025, 6, 30), date(2025, 7, 1)
HEADLINES = {"aapl night one": 0.6, "aapl night two": 0.5, "tsla night": 0.3, "aapl day": 0.6,
             "tsla day low": 0.3, "aapl late": 0.7}


class ScoreIsSentiment:
    """A stand-in model: prob_up = the headline's positive-sentiment feature."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = X[:, FEATURE_NAMES.index("sent_pos")]
        return np.column_stack([1 - p, p])


def ev(n: int, at, symbol: str, headline: str, story: str | None = None) -> Event:
    return Event(f"alpaca:{n}", str(n), "benzinga", at, None, at, headline, (symbol,), 1, story)


@pytest.fixture(scope="module")
def samples(market: MarketData) -> Samples:
    events = [ev(1, ny(DAY, 18), "AAPL", "aapl night one", "story:1"),
              ev(2, ny(DAY, 19), "AAPL", "aapl night two", "story:1"),
              ev(3, ny(DAY, 20), "TSLA", "tsla night"),
              ev(4, ny(DAY, 10), "AAPL", "aapl day"),
              ev(5, ny(DAY, 10, 30), "TSLA", "tsla day low"),
              ev(6, ny(DAY, 15, 45), "AAPL", "aapl late")]
    sentiment = {h: np.array([p, 1 - p - 0.05, 0.05]) for h, p in HEADLINES.items()}
    built, _ = build_v2(events, market, sentiment)
    return built


def fold_over(samples: Samples, thresholds: dict[str, float | None]) -> FoldResult:
    start, end = ny(date(2025, 6, 1), 0), ny(date(2025, 8, 1), 0)
    scored = np.flatnonzero((samples.made_at >= start.timestamp()) & (samples.made_at < end.timestamp()))
    model = ScoreIsSentiment()
    table = {"edges": [0.5], "mean_return_pct": [-0.1, 0.3]}
    return FoldResult(bounds=FoldBounds(date(2025, 6, 1), start, end, start - timedelta(days=7)),
                      train=np.arange(0), calib=np.arange(0), scored=scored, model=model,
                      model_version="wf-2025-06-test", artifact_sha256="", prob_scored=model.predict_proba(
                          samples.X[scored])[:, 1], prob_calib=np.empty(0), thresholds=thresholds, grids={},
                      move_tables=dict.fromkeys(("S2", "S2_collapsed", "S3", "S3_collapsed"), table), calib_auc=None)


OPEN = {"S2": 0.45, "S2_collapsed": 0.45, "S3": 0.45, "S3_collapsed": 0.45}


def test_raw_atr_converts_the_live_adjusted_atr_to_traded_dollars(market: MarketData) -> None:
    at = ny(date(2025, 6, 13), 10)                        # before AAPL's 4-for-1 split: factor 1/4
    history = market.as_of(at).daily("AAPL")
    adjusted = atr_usd(daily_features(history), history[-1])
    assert float(raw_atr(market, "AAPL", at)) == pytest.approx(adjusted * SPLIT, abs=1e-4)
    after = ny(date(2025, 6, 17), 10)
    history = market.as_of(after).daily("AAPL")
    assert float(raw_atr(market, "AAPL", after)) == pytest.approx(atr_usd(daily_features(history), history[-1]),
                                                                  abs=1e-4)


def test_s2_scores_the_night_at_the_scheduler_and_gates_on_the_fold(market: MarketData, samples: Samples) -> None:
    session = market.calendar.session(NEXT)
    strategy = S2(samples, [fold_over(samples, OPEN)])
    (pending,) = strategy.signals_for(session, market)              # TSLA's night scores 0.3: no signal
    s = pending.signal
    assert (s.symbol, s.prob_up) == ("AAPL", pytest.approx(0.55))    # mean of 0.6 and 0.5 (D7)
    assert pending.emit_at == s.created_at == ny(NEXT, 9, 31)        # created by the scheduler, at open + 1 min
    assert s.predicted_move_pct == 0.3                               # the S2 move table, bin >= 0.5
    assert s.atr == raw_atr(market, "AAPL", ny(NEXT, 9, 31))
    assert set(pending.sources) == {"alpaca:1", "alpaca:2"}
    assert strategy.min_prob_up(session) == D("0.45")

    collapsed = S2(samples, [fold_over(samples, OPEN)], collapsed=True).signals_for(session, market)
    assert collapsed[0].signal.prob_up == pytest.approx(0.6)          # the story's first event only

    closed = S2(samples, [fold_over(samples, {**OPEN, "S2": None})])
    assert closed.signals_for(session, market) == [] and closed.min_prob_up(session) == CLOSED


def test_s3_signals_like_the_inference_agent(market: MarketData, samples: Samples) -> None:
    fold = fold_over(samples, OPEN)
    ids = {(r["event_id"], r["symbol"], r["model_version"]): r["prediction_id"]
           for r in prediction_rows(samples, [fold], experiment_id="x")}
    (pending,) = S3(samples, [fold], ids).signals_for(market.calendar.session(DAY), market)
    s = pending.signal
    assert s.symbol == "AAPL" and s.signal_id == "4-AAPL-wf-2025-06-test"  # the live agent's id scheme
    assert pending.emit_at == ny(DAY, 10, 1)                                # made_at = known + 60 s
    assert s.created_at == ny(DAY, 10)                                      # the article's time, as live
    assert pending.prediction_id == ids[("alpaca:4", "AAPL", "wf-2025-06-test")]
    # 'tsla day low' scores 0.3; 'aapl late' is known after close - 30 min: neither is signalled


def test_baselines_pass_the_gate_and_differ_only_in_selection(market: MarketData, samples: Samples) -> None:
    session = market.calendar.session(NEXT)
    b1 = B1(SYMBOLS, seed=3).signals_for(session, market)
    assert sorted(p.signal.symbol for p in b1) == sorted(SYMBOLS)
    assert [p.signal.symbol for p in b1] == [p.signal.symbol for p in B1(SYMBOLS, seed=3).signals_for(session,
                                                                                                        market)]
    assert all(p.signal.predicted_move_pct > 0 and p.emit_at == ny(NEXT, 9, 31) for p in b1)
    assert [p.signal.prob_up for p in b1] == sorted((p.signal.prob_up for p in b1), reverse=True)

    b2 = B2(S2(samples, [fold_over(samples, OPEN)])).signals_for(session, market)
    assert [p.signal.symbol for p in b2] == ["AAPL", "TSLA"]      # every night, best score first, no gate

    b3 = B3(SYMBOLS, {NEXT: 1}, seed=0).signals_for(session, market)
    assert len(b3) == 1 and B3(SYMBOLS, {}, seed=0).signals_for(session, market) == []


def test_s2_through_the_engine(market: MarketData, samples: Samples) -> None:
    engine = Engine(market, S2(samples, [fold_over(samples, OPEN)]), LEVELS["central"], FeeTable.load(),
                    run_id="s2-synthetic")
    result = asyncio.run(engine.run([market.calendar.session(NEXT)]))
    lifecycle = [r["state"] for r in result.rows if r["trade_id"] == "S2-20250701-AAPL"]
    assert lifecycle[:3] == ["signal", "submitted", "filled"] and lifecycle[-1] == "exit_filled"
    assert result.days[0].carried == [] and result.days[0].entries == 1


# --- S0 / S1 --------------------------------------------------------------------------------------------

def test_buy_and_hold_and_banded_rebalancing(market: MarketData) -> None:
    sessions = market.calendar.sessions_between(date(2025, 4, 1), date(2025, 7, 31))
    fees = FeeTable.load()
    s0 = run_hold(market, SYMBOLS, sessions, LEVELS["central"], fees)
    assert s0[0].costs > 0 and all(d.traded == 0 for d in s0[1:])       # bought once, held
    first = {s: next(b for b in market.daily[s] if b.t.date() == sessions[0].date) for s in SYMBOLS}
    last = {s: next(b for b in market.daily[s] if b.t.date() == sessions[-1].date) for s in SYMBOLS}
    ideal = sum(D("50000") * D(repr(last[s].c)) / D(repr(first[s].o)) for s in SYMBOLS)
    assert abs(s0[-1].value - ideal) / ideal < D("0.001")               # costs are a few basis points

    s1 = run_hold(market, SYMBOLS, sessions, LEVELS["central"], fees, band=D("0.05"))
    rebalanced = [d.date for d in s1[1:] if d.traded > 0]
    assert all(d.month != (d - timedelta(days=4)).month or d.day <= 4 for d in rebalanced)  # month starts only
    wide = run_hold(market, SYMBOLS, sessions, LEVELS["central"], fees, band=D("0.5"))
    assert all(d.traded == 0 for d in wide[1:])                          # a band nothing breaches


def test_a_calendar_without_sessions_trades_nothing() -> None:
    cal = make_calendar(date(2025, 7, 4), date(2025, 7, 4))              # the holiday alone
    assert cal.sessions == ()
