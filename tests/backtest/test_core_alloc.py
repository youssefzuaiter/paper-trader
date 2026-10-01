"""core_alloc: weights, rules, rebalance plans, lots and the raise-cash plan (core design §3, §6).

Property tests are seeded loops (no hypothesis dependency): each draws random portfolios and checks
the design's §6.2 invariants on every one.
"""

from __future__ import annotations

import random
from datetime import date
from decimal import Decimal

import pytest

import core_alloc as ca

D = Decimal


def test_weights_quantise_to_a_millionth_and_sum_to_exactly_one() -> None:
    rng = random.Random(1)
    for _ in range(500):
        raw = {f"S{i}": rng.random() + 1e-9 for i in range(rng.randint(1, 9))}
        w = ca.quantise_weights(raw)
        assert sum(w.values()) == 1
        assert all(v == v.quantize(ca.WEIGHT_QUANTUM) for v in w.values())
    assert ca.quantise_weights({"a": 1, "b": 1, "c": 1}) == {"a": D("0.333334"), "b": D("0.333333"), "c": D("0.333333")}
    with pytest.raises(ValueError):
        ca.quantise_weights({"a": -1.0, "b": 2.0})


def test_inverse_volatility_weights_favour_the_calmer_asset() -> None:
    calm = [100 * (1.001 if i % 2 else 0.999) ** 1 for i in range(64)]
    wild = [100 * (1.02 if i % 2 else 0.98) for i in range(64)]
    w = ca.inverse_vol_weights({"BND": calm, "VTI": wild}, 63)
    assert w["BND"] > D("0.9") and sum(w.values()) == 1
    # A fixed sleeve takes its weight first; the rest is shared by inverse volatility.
    w8 = ca.inverse_vol_weights({"BND": calm, "VTI": wild}, 63, {"BTC/USD": D("0.05")})
    assert w8["BTC/USD"] == D("0.05") and sum(w8.values()) == 1
    assert float(w8["BND"] / w8["VTI"]) == pytest.approx(float(w["BND"] / w["VTI"]), rel=1e-4)
    with pytest.raises(ValueError):  # exactly lookback + 1 closes: the window is not allowed to drift
        ca.inverse_vol_weights({"BND": calm[:-1], "VTI": wild[:-1]}, 63)


def test_inverse_volatility_is_scale_invariant() -> None:
    """C5's premise: multiplying a whole price history by k changes no weight."""
    rng = random.Random(2)
    series = {s: [100.0] for s in ("A", "B", "C")}
    for s, closes in series.items():
        for _ in range(63):
            closes.append(closes[-1] * (1 + rng.gauss(0, 0.01 * (1 + "ABC".index(s)))))
    base = ca.inverse_vol_weights(series, 63)
    scaled = ca.inverse_vol_weights({s: [x * 37.5 for x in c] for s, c in series.items()}, 63)
    assert base == scaled


def test_period_ends_follow_the_calendar_not_the_weekday() -> None:
    assert ca.period_ends(date(2025, 1, 31), date(2025, 2, 3)) == {"month"}
    assert ca.period_ends(date(2025, 3, 31), date(2025, 4, 1)) == {"month", "quarter"}
    assert ca.period_ends(date(2024, 12, 31), date(2025, 1, 2)) == {"month", "quarter", "year"}
    assert ca.period_ends(date(2025, 5, 29), date(2025, 5, 30)) == frozenset()


def test_the_eight_rules() -> None:
    targets = {"A": D("0.5"), "B": D("0.5")}
    near, far = {"A": D("0.53"), "B": D("0.47")}, {"A": D("0.56"), "B": D("0.44")}
    month, none = frozenset({"month"}), frozenset()
    r = ca.RULES
    assert len(r) == 8
    assert not r["none"].due(frozenset({"month", "quarter", "year"}), far, targets)
    assert r["monthly"].due(month, near, targets) and not r["monthly"].due(none, far, targets)
    assert not r["quarterly"].due(month, far, targets)
    assert r["band5"].due(none, far, targets) and not r["band5"].due(none, near, targets)
    assert not r["band10"].due(none, far, targets)
    assert r["monthly_band5"].due(month, far, targets) and not r["monthly_band5"].due(none, far, targets)
    assert not r["monthly_band5"].due(month, near, targets)
    assert r["quarterly_band5"].due(frozenset({"month", "quarter"}), far, targets)


def test_a_rebalance_plan_goes_fully_back_to_target_and_skips_tiny_orders() -> None:
    qty = {"A": D("10"), "B": D("0")}
    prices = {"A": D("100"), "B": D("50")}
    plan = ca.plan_rebalance(date(2025, 1, 31), qty, prices, D("0"), {"A": D("0.5"), "B": D("0.5")},
                             min_order_usd=D("1"))
    assert plan.sells == {"A": D("5")} and plan.buys == {"B": D("500.00")}
    tiny = ca.plan_rebalance(date(2025, 1, 31), {"A": D("5"), "B": D("10.01")}, prices, D("0"),
                             {"A": D("0.5"), "B": D("0.5")}, min_order_usd=D("1"))
    assert tiny.empty  # 25 cents of drift is left alone
    leaving = ca.plan_rebalance(date(2025, 1, 31), {"A": D("3.123456789"), "B": D("1")}, prices, D("0"),
                                {"B": D("1")}, min_order_usd=D("1"))
    assert leaving.sells["A"] == D("3.123456789")  # an asset leaving the mix is sold entirely, no dust


def test_buys_are_scaled_to_the_cash_available() -> None:
    buys = {"A": D("600"), "B": D("400")}
    rate = {"A": D("0.001"), "B": D("0.001")}
    assert ca.scale_buys(buys, D("2000"), rate) == buys
    scaled = ca.scale_buys(buys, D("500"), rate)
    assert sum(n * (1 + rate[s]) for s, n in scaled.items()) <= D("500")
    assert scaled["A"] / scaled["B"] == pytest.approx(1.5, rel=1e-3)
    assert ca.scale_buys(buys, D("-5"), rate) == {"A": D("0.00"), "B": D("0.00")}


def test_lot_methods() -> None:
    def holding() -> ca.Holding:
        h = ca.Holding()
        h.buy(D("1"), D("100"), date(2020, 1, 2))
        h.buy(D("1"), D("300"), date(2021, 1, 4))
        h.buy(D("1"), D("200"), date(2022, 1, 3))
        return h

    assert holding().sell(D("1"), "average") == D("200")
    assert holding().sell(D("1"), "fifo") == D("100")
    assert holding().sell(D("1"), "hifo") == D("300")  # the highest basis goes first
    h = holding()
    h.sell(D("1.5"), "hifo")  # the $300 lot, then half the $200 lot
    assert h.qty == D("1.5") and h.cost == D("200")
    with pytest.raises(ValueError):
        holding().sell(D("4"), "fifo")


def test_tax_nets_a_year_and_carries_losses_forward() -> None:
    t = ca.TaxYear(D("0.15"))
    t.realise(2020, D("-1000"))
    assert t.due(2020) == 0 and t.carried_loss == D("1000")
    t.realise(2021, D("1500"))
    assert t.due(2021) == D("75.00") and t.carried_loss == 0


# --- raise cash (§6.2) ------------------------------------------------------------------------------------

def _random_portfolio(rng: random.Random) -> tuple[dict, dict, dict, dict]:
    symbols = ["VTI", "VXUS", "BND", "IAU", "VNQ", "BIL"][: rng.randint(2, 6)]
    prices = {s: D(str(round(rng.uniform(20, 400), 2))) for s in symbols}
    qty = {s: D(str(round(rng.uniform(0, 200), 6))) for s in symbols}
    targets = ca.quantise_weights({s: rng.random() + 0.05 for s in symbols})
    rates = {s: D(str(round(rng.uniform(0, 0.002), 6))) for s in symbols}
    return qty, prices, targets, rates


def _net(plan: ca.Plan, prices: dict, rates: dict) -> Decimal:
    return sum((q * prices[s] * (1 - rates[s]) for s, q in plan.sells.items()), D(0))


def test_raise_cash_properties_hold_on_random_portfolios() -> None:
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        qty, prices, targets, rates = _random_portfolio(rng)
        cash = D(str(round(rng.uniform(0, 500), 2)))
        value = cash + sum(qty[s] * prices[s] for s in qty)
        need = (value * D(str(rng.uniform(0.01, 0.5)))).quantize(D("0.01"))
        plan = ca.plan_raise_cash(date(2025, 3, 3), qty, prices, cash, need, targets, cost_rate=rates)
        raised = min(cash, need) + _net(plan, prices, rates)
        steps = len(plan.sells)
        assert raised >= need - D("0.01"), (need, raised)                       # net ≥ X
        assert raised <= need + D("0.01") + sum(prices.values()) * D("1e-9") * 2 + steps * D("1.01")  # < X + small
        assert all(0 <= plan.sells[s] <= qty[s] for s in plan.sells)              # nothing below zero
        assert not plan.buys
        # No under-weight asset is sold while an over-weight one remains unsold (after the withdrawal).
        after = value - need
        excess = {s: qty[s] * prices[s] - targets[s] * after for s in qty}
        sold_under = [s for s in plan.sells if excess[s] < 0]
        if sold_under:
            assert all(qty[s] * prices[s] - plan.sells.get(s, 0) * prices[s] - targets[s] * after <= D("1.01")
                       for s in qty if excess[s] > 0)
        checked += 1
    assert checked == 400


def test_raise_cash_uses_free_cash_then_the_buffer_first() -> None:
    qty = {"VTI": D("100"), "BIL": D("100")}
    prices = {"VTI": D("200"), "BIL": D("90")}
    targets = {"VTI": D("0.95"), "BIL": D("0.05")}
    rates = {"VTI": D("0"), "BIL": D("0")}
    plan = ca.plan_raise_cash(date(2025, 3, 3), qty, prices, D("1000"), D("3000"), targets, cost_rate=rates,
                              buffer_symbol="BIL")
    assert set(plan.sells) == {"BIL"} and plan.sells["BIL"] * prices["BIL"] >= D("2000")
    assert ca.plan_raise_cash(date(2025, 3, 3), qty, prices, D("5000"), D("3000"), targets, cost_rate=rates,
                              buffer_symbol="BIL").empty


def test_with_zero_costs_raise_cash_is_pure_water_filling() -> None:
    qty = {"A": D("60"), "B": D("30"), "C": D("10")}
    prices = dict.fromkeys(qty, D("10"))  # values 600 / 300 / 100 against equal targets
    targets = {"A": D("0.333334"), "B": D("0.333333"), "C": D("0.333333")}
    plan = ca.plan_raise_cash(date(2025, 3, 3), qty, prices, D("0"), D("200"), targets,
                              cost_rate=dict.fromkeys(qty, D("0")))
    assert set(plan.sells) == {"A"} and plan.sells["A"] == D("20")  # only the largest over-weight
    plan = ca.plan_raise_cash(date(2025, 3, 3), qty, prices, D("0"), D("600"), targets,
                              cost_rate=dict.fromkeys(qty, D("0")))
    # A down to B's level (300 sold), then both together: A and B end level, C untouched.
    assert "C" not in plan.sells and abs(plan.sells["A"] - plan.sells["B"] - D("30")) < D("0.0001")


def test_raise_cash_refuses_more_than_the_portfolio() -> None:
    with pytest.raises(ValueError, match="exceeds_portfolio"):
        ca.plan_raise_cash(date(2025, 3, 3), {"A": D("1")}, {"A": D("100")}, D("0"), D("100"), {"A": D("1")},
                           cost_rate={"A": D("0.01")})


# --- compound metrics (core design §5.1) ------------------------------------------------------------------------

def test_the_shared_index_matrix_is_block_bootstraps_own_resamples() -> None:
    import numpy as np

    from backtest import metrics as m

    rng = np.random.default_rng(0)
    assert (m.index_matrix(300, resamples=40, seed=0) == np.array([m.stationary_indices(300, 5, rng)
                                                                   for _ in range(40)])).all()
    x = np.random.default_rng(1).normal(0.0004, 0.01, 300)
    phase1 = m.block_bootstrap(x, np.mean, n=500, seed=3)
    shared = m.interval_from(float(np.mean(x)), m.resampled(lambda r: r.mean(axis=-1),
                                                            m.index_matrix(300, resamples=500, seed=3), x),
                             resamples=500, seed=3)
    assert (shared.low, shared.high) == (phase1.low, phase1.high)


def test_drawdown_is_compound_with_dates_and_the_longest_spell() -> None:
    import numpy as np

    from backtest import metrics as m

    r = np.array([0.10, -0.50, 0.20, 0.30, 1.0, -0.01])
    d = m.drawdown_detail(r)
    assert d.depth == pytest.approx(-0.5) and (d.peak, d.trough, d.recovery) == (1, 2, 5)
    assert (d.longest_under, d.longest_under_from, d.longest_under_to, d.open_at_end) == (3, 1, 5, False)
    # Additive would say -0.5 + 0.2 + ... ; compound says the path halved: -50%.
    assert float(m.compound_drawdown(r)) == pytest.approx(-0.5)
    open_end = m.drawdown_detail(np.array([0.1, -0.2, 0.05, 0.01]))
    assert open_end.open_at_end and open_end.recovery is None and open_end.longest_under == 3
    assert m.year_returns([0.1, 0.1, -0.5], [2020, 2020, 2021]) == {2020: pytest.approx(0.21), 2021: -0.5}
    assert float(m.cagr(np.full(252, (1.07) ** (1 / 252) - 1))) == pytest.approx(0.07)
