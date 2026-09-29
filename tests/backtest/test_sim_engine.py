"""The simulated broker and the replay through the production router."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import HALF_DAY, daily_bars, make_calendar, ny

from backtest.costs import LEVELS, FeeTable
from backtest.data import MarketData, regular_series
from backtest.engine import Engine, EngineParams, PendingSignal, RunResult
from backtest.sim_broker import FillTimingError, SimAlpaca, SimOrderRejected
from backtest.store import Store
from risk_router.gatekeeper import buy_order_payload
from risk_router.policy import PortfolioSnapshot
from risk_router.schemas import TradeSignal
from tier0 import plan_buy

D = Decimal
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "alpaca_paper.json").read_text())
DAY = date(2025, 6, 30)
NEXT = date(2025, 7, 1)
Path_ = Callable[[date, int], tuple[float, float, float, float]]


def flat(_: date, __: int) -> tuple[float, float, float, float]:
    return 100.0, 100.02, 99.95, 100.0


def custom_market(path: Path_, days: tuple[date, ...] = (DAY, NEXT, HALF_DAY), symbols=("AAPL",)) -> MarketData:
    calendar = make_calendar(date(2025, 3, 3), date(2025, 7, 31))
    minute = {}
    for symbol in symbols:
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
        minute[symbol] = regular_series(symbol, 60, a["t"].astype(np.int64), a["o"], a["h"], a["l"], a["c"], a["v"],
                                        calendar)
    daily = {s: daily_bars(s, calendar, seed=1)[0] for s in symbols}
    raw = {s: daily_bars(s, calendar, seed=1)[1] for s in symbols}
    return MarketData(calendar, minute, daily, raw)


class Scripted:
    """Signals fixed in advance, for testing the engine rather than a strategy."""

    name = "scripted"

    def __init__(self, signals: list[PendingSignal], min_prob: str = "0.5") -> None:
        self._signals = signals
        self._min = D(min_prob)

    def min_prob_up(self, session) -> Decimal:
        return self._min

    def signals_for(self, session, market) -> list[PendingSignal]:
        return [p for p in self._signals if session.open_at <= p.emit_at <= session.close_at]


def sig(n: int, at: datetime, symbol: str = "AAPL", prob: float = 0.7, move: float = 0.5,
        created_at: datetime | None = None) -> PendingSignal:
    return PendingSignal(at, TradeSignal(signal_id=f"sig-{n:08d}", symbol=symbol, source="test", model_name="m",
                                         prob_up=prob, predicted_move_pct=move, confidence=0.5,
                                         created_at=created_at or at))


def replay(market: MarketData, signals: list[PendingSignal], level: str = "central", days=(DAY,),
           **params: Any) -> RunResult:
    engine = Engine(market, Scripted(signals), LEVELS[level], FeeTable.load(), EngineParams(**params))
    return asyncio.run(engine.run([market.calendar.session(d) for d in days]))


def states(result: RunResult, trade: str = "sig-00000001") -> list[tuple[str, str | None]]:
    return [(r["state"], r.get("reason_code")) for r in result.rows if r["trade_id"] == trade]


def row(result: RunResult, state: str, trade: str = "sig-00000001") -> dict[str, Any]:
    return next(r for r in result.rows if r["trade_id"] == trade and r["state"] == state)


# --- the contract with Alpaca ----------------------------------------------------------------

def _types(body: dict[str, Any]) -> dict[str, str]:
    return {k: type(v).__name__ for k, v in body.items()}


def test_bodies_match_recorded_paper_responses() -> None:
    market = custom_market(flat)
    sim = SimAlpaca(market, LEVELS["central"], FeeTable.load())
    sim.set_time(ny(DAY, 10))
    plan = plan_buy(symbol="AAPL", bid=D("99.98"), ask=D("100.02"), atr=None, client_order_id="c1")
    assert sim.submit(buy_order_payload(plan))[0] == 200
    sim.advance(ny(DAY, 10, 2))
    (order,) = sim.orders_body("all", None, 50, "desc")
    (position,) = sim.positions_body()
    for ours, theirs in ((sim.account_body(), FIXTURE["account"]), (position, FIXTURE["position"]),
                         (order, FIXTURE["order_filled_limit_buy"]), (sim.clock_body(), FIXTURE["clock"]),
                         (sim.quote_body("AAPL")["quote"], FIXTURE["latest_quote"]["quote"])):
        assert _types(ours) == _types(theirs) | {k: _types(ours)[k] for k in ("ap", "bp") if k in theirs}
    for body in (position, order):
        for value in body.values():
            if isinstance(value, str) and value[:1].isdigit() and "T" not in value and "-" not in value:
                Decimal(value)  # every number Alpaca sends as a string parses exactly
    for stamp in (order["submitted_at"], order["filled_at"], FIXTURE["order_filled_limit_buy"]["filled_at"],
                  FIXTURE["clock"]["timestamp"], FIXTURE["latest_quote"]["quote"]["t"]):
        assert datetime.fromisoformat(stamp).utcoffset() is not None


def test_the_production_client_and_policy_read_the_simulation() -> None:
    market = custom_market(flat)
    sim = SimAlpaca(market, LEVELS["central"], FeeTable.load())
    sim.set_time(ny(DAY, 10))

    async def go() -> PortfolioSnapshot:
        client = sim.client()
        try:
            plan = plan_buy(symbol="AAPL", bid=D("99.98"), ask=D("100.02"), atr=None, client_order_id="c1")
            await client.submit_order(buy_order_payload(plan))
            quote = await client.latest_quote("AAPL")
            assert (quote.bid, quote.ask, quote.timestamp) == (D("99.98"), D("100.02"), ny(DAY, 10))
            return PortfolioSnapshot.from_alpaca(positions=await client.positions(),
                                                 open_orders=await client.orders(status="open"),
                                                 recent_orders=await client.orders(status="all", after=ny(DAY, 0)),
                                                 day_start=ny(DAY, 0))
        finally:
            await client.aclose()

    snapshot = asyncio.run(go())
    plan = plan_buy(symbol="AAPL", bid=D("99.98"), ask=D("100.02"), atr=None)
    assert snapshot.buys_today == 1 and snapshot.working_buys["AAPL"] == plan.quantity * plan.limit_price


# --- fills ----------------------------------------------------------------------------------------

@pytest.mark.parametrize(("level", "price"), [("central", D("100.0350")), ("optimistic", D("100.0050")),
                                              ("pessimistic", D("100.30"))])
def test_entry_fills_one_bar_after_the_decision_at_the_levels_price(level: str, price: Decimal) -> None:
    result = replay(custom_market(flat), [sig(1, ny(DAY, 10, 0, 30))], level)
    assert [s for s, _ in states(result)][:3] == ["signal", "submitted", "filled"]
    filled = row(result, "filled")
    assert filled["detail"]["bar_start"] == ny(DAY, 10, 2).isoformat()  # decided 10:01, fills from 10:02
    assert filled["occurred_at"] == ny(DAY, 10, 3) and filled["price"] == price


def test_everything_is_sold_ten_minutes_before_the_close() -> None:
    result = replay(custom_market(flat), [sig(1, ny(DAY, 10))])
    exit_ = row(result, "exit_submitted")
    assert (exit_["occurred_at"], exit_["reason_code"]) == (ny(DAY, 15, 50), "session_close")
    sold = row(result, "exit_filled")
    assert sold["detail"]["bar_start"] == ny(DAY, 15, 51).isoformat()
    assert sold["price"] == D("99.9650")               # 100 x (1 - 1.5 bps - 2 bps), rounded down
    assert result.days[0].carried == [] and len(result.days) == 1


def test_a_stop_fires_on_a_touch_and_a_target_only_on_a_close() -> None:
    def path(day: date, k: int) -> tuple[float, float, float, float]:
        if k == 90:                 # 11:00: one bar trades 5.1% down but closes flat
            return 100.0, 100.0, 94.90, 100.0
        return flat(day, k)

    stopped = replay(custom_market(path), [sig(1, ny(DAY, 10))])
    assert row(stopped, "exit_submitted")["reason_code"] == "stop_loss"
    assert row(stopped, "exit_submitted")["occurred_at"] == ny(DAY, 11, 1)   # at that bar's end

    def spike(day: date, k: int) -> tuple[float, float, float, float]:
        if k == 90:                 # trades 11% up, closes 9% up: no take-profit
            return 100.0, 111.0, 100.0, 109.0
        if k == 120:                # closes 10.5% up: take-profit
            return 109.0, 110.6, 109.0, 110.5
        return (109.0, 109.02, 108.95, 109.0) if k > 90 else flat(day, k)

    target = replay(custom_market(spike), [sig(1, ny(DAY, 10))])
    assert row(target, "exit_submitted")["reason_code"] == "take_profit"
    assert row(target, "exit_submitted")["occurred_at"] == ny(DAY, 11, 31)


def test_half_day_entry_cutoff_and_exit() -> None:
    result = replay(custom_market(flat), [sig(1, ny(HALF_DAY, 12)), sig(2, ny(HALF_DAY, 12, 31), "AAPL")],
                    days=(HALF_DAY,))
    assert row(result, "rejected", "sig-00000002")["reason_code"] == "entry_window_closed"
    assert row(result, "exit_submitted")["occurred_at"] == ny(HALF_DAY, 12, 50)


def test_rejections_are_recorded_with_their_codes() -> None:
    signals = [sig(1, ny(DAY, 10)), sig(2, ny(DAY, 10, 5)),                      # duplicate accumulation
               sig(3, ny(DAY, 9, 30), "AAPL"),                                   # the open: last quote is yesterday's
               sig(4, ny(DAY, 11), prob=0.45),                                    # below the gate
               sig(5, ny(DAY, 12), created_at=ny(DAY, 11, 57))]                   # stale by the decision
    result = replay(custom_market(flat), signals)
    assert dict(result.rejections) == {"duplicate_accumulation": 1, "stale_quote": 1, "below_min_prob": 1,
                                       "stale_signal": 1}


def test_p4_is_asserted_on_every_fill() -> None:
    market = custom_market(flat)
    sim = SimAlpaca(market, LEVELS["central"], FeeTable.load(), latency_bars=0)
    sim.set_time(ny(DAY, 10))
    plan = plan_buy(symbol="AAPL", bid=D("99.98"), ask=D("100.02"), atr=None, client_order_id="c1")
    sim.submit(buy_order_payload(plan))
    with pytest.raises(FillTimingError):
        sim.advance(ny(DAY, 10, 1))


def test_native_stop_orders_are_refused_loudly() -> None:
    sim = SimAlpaca(custom_market(flat), LEVELS["central"], FeeTable.load())
    sim.set_time(ny(DAY, 10))
    with pytest.raises(SimOrderRejected):
        sim.submit({"symbol": "AAPL", "qty": "2", "side": "buy", "type": "limit", "time_in_force": "day",
                    "limit_price": "5", "client_order_id": "c", "order_class": "oto",
                    "stop_loss": {"stop_price": "4.75"}})


def test_unfilled_day_orders_expire_at_the_close() -> None:
    def gap_up(day: date, k: int) -> tuple[float, float, float, float]:
        return (110.0, 110.5, 109.8, 110.0) if k > 30 else flat(day, k)   # never back below the limit

    result = replay(custom_market(gap_up), [sig(1, ny(DAY, 10, 0, 30))])
    assert [s for s, _ in states(result)] == ["signal", "submitted", "expired"]
    assert row(result, "expired")["occurred_at"] == ny(DAY, 16)


def test_rows_load_into_the_store_and_the_trade_view_matches_the_account(tmp_path: Path) -> None:
    result = replay(custom_market(flat), [sig(1, ny(DAY, 10))], "pessimistic")
    with Store(tmp_path / "s.duckdb") as store:
        store.append_trade_events(result.rows)
        (trade,) = store.trades("run")
    day = result.days[0]
    assert trade["net_pnl"] == day.pnl                  # cash legs and per-order fees, to the cent
    assert trade["fees"] == D("0.03")                   # 2025-06-30, per order: CAT buy, CAT + TAF sell; SEC at $0
    assert trade["gross_pnl"] == 0                      # flat prices: every cent lost is a cost
