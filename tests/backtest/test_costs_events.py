"""The fill and cost model (design §3, D1), fees, and story clustering."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from backtest import costs
from backtest.costs import FRICTIONLESS, LEVELS, DailyFees, FeeTable
from backtest.events import Event, cluster, from_articles, jaccard, normalise, store_rows

D = Decimal
CENTRAL, PESSIMISTIC, OPTIMISTIC = LEVELS["central"], LEVELS["pessimistic"], LEVELS["optimistic"]


# --- quotes and fills ------------------------------------------------------------------

def test_quote_rounds_outward_to_the_cent() -> None:
    q = costs.quote(D("187.15"), "AAPL", CENTRAL, D(1))      # h = 1.5 bps
    assert (q.bid, q.ask) == (D("187.12"), D("187.18"))
    wide = costs.quote(D("187.15"), "TSLA", CENTRAL, D(2))   # 2.5 bps, doubled at the open
    assert (wide.bid, wide.ask) == (D("187.05"), D("187.25"))


def test_market_orders_pay_spread_and_slippage_rounded_against_us() -> None:
    buy = costs.market_fill("buy", D("100.00"), "AAPL", CENTRAL, D(1))    # 1.5 + 2 bps
    sell = costs.market_fill("sell", D("100.00"), "AAPL", CENTRAL, D(1))
    assert (buy.price, sell.price) == (D("100.0350"), D("99.9650"))
    assert (buy.spread_cost, buy.slippage_cost) == (D("0.0150"), D("0.0200"))
    odd = costs.market_fill("buy", D("33.33"), "AAPL", CENTRAL, D(1))     # 33.3416655 → up
    assert odd.price == D("33.3417")


def test_limit_buys_fill_only_through_the_limit() -> None:
    limit = D("100.30")
    assert costs.limit_buy_fill(limit, D("100.10"), D("100.30"), "AAPL", CENTRAL, D(1)) is None  # a touch
    assert costs.limit_buy_fill(limit, D("100.40"), D("100.35"), "AAPL", CENTRAL, D(1)) is None  # never below


def test_d1_the_levels_bracket_reality() -> None:
    limit, open_, low = D("100.30"), D("100.00"), D("99.90")
    assert costs.limit_buy_fill(limit, open_, low, "AAPL", PESSIMISTIC, D(1)).price == limit  # pays the limit
    central = costs.limit_buy_fill(limit, open_, low, "AAPL", CENTRAL, D(1))
    assert central.price == D("100.0350")                                                     # o(1 + hm + σ)
    optimistic = costs.limit_buy_fill(limit, open_, low, "AAPL", OPTIMISTIC, D(1))
    assert optimistic.price == D("100.0050")
    # a bar that opens above the limit and trades down through it fills at the limit at every level
    assert costs.limit_buy_fill(limit, D("100.50"), D("100.00"), "AAPL", CENTRAL, D(1)).price == limit


def test_opening_multiplier_covers_the_first_five_minutes() -> None:
    open_at = datetime(2025, 6, 30, 13, 30, tzinfo=UTC)
    assert costs.multiplier(CENTRAL, open_at + timedelta(minutes=4), open_at) == D(2)
    assert costs.multiplier(CENTRAL, open_at + timedelta(minutes=5), open_at) == D(1)


def test_cash_debits_round_up_and_credits_down() -> None:
    assert costs.cash_debit(D("0.053"), D("188.1234"), CENTRAL) == D("9.98")   # 9.9705...
    assert costs.cash_credit(D("0.053"), D("188.1234"), CENTRAL) == D("9.97")
    assert costs.cash_debit(D("0.053"), D("188.1234"), FRICTIONLESS) == D("0.053") * D("188.1234")


# --- fees ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fees() -> FeeTable:
    return FeeTable.load()


@pytest.mark.parametrize(("day", "sec_per_million"), [
    (date(2024, 5, 21), D("8")), (date(2024, 5, 22), D("27.8")), (date(2025, 5, 13), D("27.8")),
    (date(2025, 5, 14), D("0")), (date(2026, 4, 3), D("0")), (date(2026, 4, 4), D("20.6")),
])
def test_sec_fee_follows_the_advisories(fees: FeeTable, day: date, sec_per_million: Decimal) -> None:
    exact = fees.exact(day, "sell", D("1000"), D("1000"))   # $1,000,000 of principal
    assert exact["sec"] == sec_per_million


def test_taf_by_year_and_its_cap(fees: FeeTable) -> None:
    assert fees.exact(date(2025, 12, 31), "sell", D("100"), D("10"))["taf"] == D("0.0166")
    assert fees.exact(date(2026, 1, 2), "sell", D("100"), D("10"))["taf"] == D("0.0195")
    assert fees.exact(date(2026, 1, 2), "sell", D("100000"), D("10"))["taf"] == D("9.79")
    assert set(fees.exact(date(2026, 1, 2), "buy", D("1"), D("10"))) == {"cat"}  # buys pay CAT only


def test_the_fee_table_is_alpacas_schedule_of_2026_09_17(fees: FeeTable) -> None:
    """Owner-confirmed (D12): SEC 0.0000206 x trade value on sells, TAF 0.000195 a share on sells
    (max 9.79 a trade), CAT 0.000003 a share on buys and sells; still in force after 2026."""
    for day in (date(2026, 9, 17), date(2027, 6, 1)):
        assert fees.exact(day, "sell", D("100"), D("250")) == {
            "sec": D("0.0000206") * D("25000"), "taf": D("0.000195") * 100, "cat": D("0.000003") * 100}
        assert fees.exact(day, "buy", D("100"), D("250")) == {"cat": D("0.000003") * 100}
        assert fees.exact(day, "sell", D("60000"), D("1"))["taf"] == D("9.79")
    assert {k: v.fee_rounding for k, v in LEVELS.items()} == {
        "optimistic": "daily", "central": "daily", "pessimistic": "per_order"}


def test_a_fee_table_with_a_gap_is_refused(tmp_path) -> None:
    import json

    from backtest.costs import FEES_PATH
    doc = json.loads(FEES_PATH.read_text())
    gap = [r for r in doc["fees"] if not (r["fee"] == "taf" and r["from"] == "2024-01-01")]
    gap.append({"fee": "taf", "side": "sell", "basis": "shares", "rate": "0.000166", "cap": "8.30",
                "from": "2024-01-01", "to": "2025-11-30"})                          # December 2025 missing
    closed = [{**r, "to": "2026-12-31"} if r["fee"] == "cat" else r for r in doc["fees"]]
    for rows in (gap, closed):
        path = tmp_path / "fees.json"
        path.write_text(json.dumps({**doc, "fees": rows}))
        with pytest.raises(ValueError):
            FeeTable.load(path)


def test_per_order_rounding_is_what_makes_small_orders_expensive(fees: FeeTable) -> None:
    """A $10 sale: exact fees are fractions of a cent; per-order rounding makes each a full cent."""
    day, qty, price = date(2026, 5, 4), D("0.05"), D("200")
    per_order = fees.order_fees(day, "sell", qty, price, "per_order")
    assert per_order == {"sec": D("0.01"), "taf": D("0.01"), "cat": D("0.01")}
    exact = fees.order_fees(day, "sell", qty, price, "daily")
    assert sum(exact.values()) < D("0.001")


def test_daily_rounding_charges_at_most_a_cent_per_fee_type_per_day(fees: FeeTable) -> None:
    day, daily = date(2026, 5, 4), DailyFees()
    for _ in range(5):
        daily.add(day, fees.order_fees(day, "sell", D("0.05"), D("200"), "daily"))
    exact_total = sum(sum(fees.exact(day, "sell", D("0.05"), D("200")).values()) for _ in range(5))
    assert D("0") < daily.rounding_charge(day) <= D("0.03")
    assert (exact_total + daily.rounding_charge(day)) % D("0.01") == 0


def test_round_trip_cost_matches_the_design_table(fees: FeeTable) -> None:
    """§3: ≈ 9 bps before fees at central with D1's entry (tight group, opening entry), ≈ 46 pessimistic."""
    kw = {"day": date(2025, 6, 2), "price": D("200"), "notional": D("100000"), "entry_in_opening_window": True}
    central = costs.round_trip_fraction("AAPL", CENTRAL, fees, **kw)
    pessimistic = costs.round_trip_fraction("AAPL", PESSIMISTIC, fees, **kw)
    assert 0.00085 <= central <= 0.0009
    assert 0.0045 <= pessimistic <= 0.0047
    # $10 per order, per-order rounding: whole cents. SEC's rate was $0 from 2025-05-14 to 2026-04-03,
    # so TAF and CAT on both legs make three cents (30 bps); from 2026-04-04 the SEC fee adds a fourth.
    at_ten = costs.round_trip_fraction("AAPL", PESSIMISTIC, fees, **{**kw, "notional": D("10")})
    assert at_ten - pessimistic == pytest.approx(0.003, abs=2e-6)
    at_ten_2026 = costs.round_trip_fraction("AAPL", PESSIMISTIC, fees,
                                            **{**kw, "day": date(2026, 5, 4), "notional": D("10")})
    assert at_ten_2026 - at_ten == pytest.approx(0.001, abs=1e-12)


# --- events and stories -------------------------------------------------------------------------

T = datetime(2025, 3, 3, 14, 0, tzinfo=UTC)


def ev(n: int, minutes: float, headline: str, symbols: tuple[str, ...] = ("AAPL",)) -> Event:
    at = T + timedelta(minutes=minutes)
    return Event(f"alpaca:{n}", str(n), "benzinga", at, None, at, headline, symbols, len(symbols))


def test_normalisation_strips_case_digits_and_punctuation() -> None:
    assert normalise("Apple's Q3: $94.9B revenue, up 5%!") == frozenset({"apple", "s", "q", "b", "revenue", "up"})
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0


def test_near_duplicates_within_six_hours_are_one_story() -> None:
    events = [
        ev(1, 0, "Apple Shares Rise After Strong iPhone Sales Report"),
        ev(2, 30, "Apple shares rise after strong iPhone sales report (update)"),     # near-duplicate
        ev(3, 60, "Apple shares rise after strong iPhone sales report", ("TSLA",)),   # other symbol
        ev(4, 7 * 60, "Apple shares rise after strong iPhone sales report"),          # too late for 1, not for 2?
        ev(5, 90, "Tesla recalls Model Y over seat belt issue"),                      # different text
    ]
    stories = cluster(events, frozenset({"AAPL", "TSLA"}))
    assert stories["alpaca:1"] == stories["alpaca:2"] == "story:1"
    assert stories["alpaca:3"] == "story:3"
    assert stories["alpaca:4"] == "story:4"  # 6h30 after event 2, 7h after event 1
    assert stories["alpaca:5"] == "story:5"


def test_single_linkage_chains_a_running_story() -> None:
    events = [ev(1, 0, "apple wins patent case against rival"),
              ev(2, 5 * 60, "apple wins patent case against rival firm"),
              ev(3, 10 * 60, "apple wins patent case against rival firm today")]
    stories = cluster(events, frozenset({"AAPL"}))
    assert len(set(stories.values())) == 1  # 1~2 and 2~3, although 1 and 3 are ten hours apart


def test_events_from_the_cache_keep_their_headline_and_gain_updated_at() -> None:
    updated = datetime(2025, 3, 4, tzinfo=UTC)
    cached = [{"id": 7, "headline": " AAPL beats ", "symbols": ["AAPL", "F"], "created_at": "2025-03-03T14:00:00Z",
               "source": "benzinga"}]
    fetched = [{"id": 7, "headline": "AAPL beats (revised)", "symbols": "AAPL,F",
                "created_at": datetime(2025, 3, 3, 14, tzinfo=UTC), "updated_at": updated, "source": "benzinga"}]
    (e,), counts = from_articles(fetched, cached)
    assert (e.headline, e.updated_at, e.n_symbols) == ("AAPL beats", updated, 2)
    assert counts["headline_changed"] == 1 and counts["cached_only"] == 0
    rows, symbols = store_rows([e])
    assert rows[0]["known_at_basis"] == "published" and len(symbols) == 2
