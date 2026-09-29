"""The calendar and the point-in-time view (P2, P3, P7)."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import pytest
from conftest import HALF_DAY, HOLIDAY, SPLIT, make_market, ny

from backtest.calendar import Calendar
from backtest.data import MarketData
from backtest.lockbox import LOCKBOX_SESSIONS_FROM, LockBoxError, issue_key
from risk_router.gatekeeper import session_close
from swarm.features import completed_sessions


def test_clock_speaks_alpaca_and_the_router_reads_half_days(calendar: Calendar) -> None:
    during = calendar.clock(ny(HALF_DAY, 11))
    assert during["is_open"] is True
    assert session_close(during) == ny(HALF_DAY, 13)  # the router's own parser
    assert during["next_close"].endswith("-04:00")    # New York offset, as Alpaca sends it

    holiday = calendar.clock(ny(HOLIDAY, 11))
    assert holiday["is_open"] is False
    assert datetime.fromisoformat(holiday["next_open"]) == ny(date(2025, 7, 7), 9, 30)

    assert calendar.clock(ny(HALF_DAY, 13))["is_open"] is False  # the close is exclusive


def test_session_lookups(calendar: Calendar) -> None:
    friday_close = ny(date(2025, 6, 27), 16)
    assert calendar.session_at(friday_close) is None
    assert calendar.next_session(friday_close).date == date(2025, 6, 30)
    assert calendar.previous_session(friday_close).date == date(2025, 6, 27)
    assert calendar.session_at(ny(date(2025, 6, 30), 9, 30)).date == date(2025, 6, 30)
    assert calendar.shift(calendar.session(date(2025, 7, 7)), -2).date == date(2025, 7, 2)


def test_the_weekday_rule_is_wrong_where_it_matters() -> None:
    naive = Calendar.weekday_rule(date(2025, 6, 30), date(2025, 7, 7))
    assert naive.session(HOLIDAY) is not None
    assert naive.session(HALF_DAY).close_at == ny(HALF_DAY, 16)


def test_daily_view_is_exactly_completed_sessions(market: MarketData) -> None:
    rng = random.Random(3)
    bars = market.daily["AAPL"]
    for _ in range(300):
        t = ny(date(2025, 4, 1), 0) + timedelta(minutes=rng.randrange(0, 120 * 24 * 60))
        assert market.as_of(t).daily("AAPL") == completed_sessions(bars, t)


def test_a_minute_bar_is_a_quote_only_from_its_end(market: MarketData) -> None:
    day = date(2025, 6, 30)
    assert market.as_of(ny(day, 9, 30)).quote_bar("AAPL").start < ny(day, 9, 30)  # yesterday's last bar
    at_931 = market.as_of(ny(day, 9, 31)).quote_bar("AAPL")
    assert (at_931.start, at_931.end) == (ny(day, 9, 30), ny(day, 9, 31))
    assert market.as_of(ny(day, 9, 31) - timedelta(microseconds=1)).quote_bar("AAPL").start < ny(day, 9, 30)


def test_pre_market_bars_are_never_used(market: MarketData) -> None:
    series = market.minute["AAPL"]
    assert all(market.calendar.sessions[s].contains(datetime.fromtimestamp(int(t), ny(HOLIDAY, 0).tzinfo))
               for t, s in zip(series.start[:2000], series.session[:2000], strict=True))


def test_model_inputs_lag_by_the_sip_delay(market: MarketData) -> None:
    t = ny(date(2025, 6, 30), 11)
    bars = market.as_of(t).input_bars("AAPL", since=ny(date(2025, 6, 30), 9, 30))
    assert bars[-1].end <= t - timedelta(minutes=16) and bars[-1].end > t - timedelta(minutes=17)


def test_factor_converts_raw_to_adjusted(market: MarketData) -> None:
    assert market.factor("AAPL", date(2025, 6, 13)) == pytest.approx(1 / SPLIT)
    assert market.factor("AAPL", date(2025, 6, 16)) == pytest.approx(1.0)


def test_lockbox_sessions_are_not_even_loaded() -> None:
    from conftest import make_calendar
    cal = make_calendar(date(2026, 8, 3), date(2026, 9, 30))
    closed = make_market(cal, minute_from=date(2026, 9, 1))
    assert max(closed.calendar.sessions[s].date for s in closed.minute["AAPL"].session) < LOCKBOX_SESSIONS_FROM
    with pytest.raises(LockBoxError):
        closed.as_of(ny(LOCKBOX_SESSIONS_FROM, 10))
    assert all(b.t.date() < LOCKBOX_SESSIONS_FROM for b in closed.daily["AAPL"])

    opened = MarketData(cal, closed.minute, closed.daily, {}, lockbox=issue_key("x"))
    opened.as_of(ny(LOCKBOX_SESSIONS_FROM, 10))  # a registered lock-box run may
