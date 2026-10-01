"""C1-C12 on real data: each check passes, and each fails on its broken twin (core design §7)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from backtest import core_grid as G
from backtest import leakage_core as L

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / ".cache" / "backtest" / "core" / "bars_1day_all.parquet"
PHASE1 = (ROOT / ".cache" / "backtest" / "calendar.parquet").exists() and (ROOT / ".cache" / "training").exists()
pytestmark = pytest.mark.skipif(not CORE.exists(), reason="needs the core's fetched data")


@pytest.fixture(scope="module")
def market():
    from backtest.core_fetch import BTC, ETFS, CorePaths
    from backtest.daily import DailyMarket
    return DailyMarket.load(CorePaths.under(ROOT / ".cache" / "backtest"), [*ETFS, BTC], last=G.WINDOWS["full"][1])


@pytest.fixture(scope="module")
def paths():
    from backtest.core_fetch import CorePaths
    return CorePaths.under(ROOT / ".cache" / "backtest")


def test_c1(market) -> None:
    assert L.c1_m4_canary(market, instants=60, seed=1).passed
    assert not L.c1_m4_canary(market, instants=10, seed=1, leaky=True).passed


def test_c2(market) -> None:
    assert L.c2_pit_oracle(market, instants=1500, seed=1).passed
    assert not L.c2_pit_oracle(market, instants=1500, seed=1, delay=timedelta(0)).passed


def test_c3(market) -> None:
    assert L.c3_execution_lag(market).passed
    assert not L.c3_execution_lag(market, fill_at_decision_close=True).passed


def test_c4(market) -> None:
    assert L.c4_band_trigger(market).passed
    assert not L.c4_band_trigger(market, trigger_reads_next_open=True).passed


@pytest.mark.parametrize("seed", [1, 2])
def test_c5(market, seed: int) -> None:
    assert L.c5_adjustment_invariance(market, seed=seed).passed
    assert not L.c5_adjustment_invariance(market, seed=seed, price_differences=True).passed


def test_c6(market, paths) -> None:
    check = L.c6_crypto_cutoff(market, paths)
    assert check.passed and check.detail["half_days"] > 0 and check.detail["dst_weeks"] > 0
    assert not L.c6_crypto_cutoff(market, paths, sampler="bar_after_close").passed
    assert not L.c6_crypto_cutoff(market, paths, sampler="utc_2100").passed


@pytest.mark.skipif(not PHASE1, reason="needs phase 1's caches")
def test_c7_twins() -> None:
    assert not L._c7_cash_safe(ROOT).passed
    assert not L.c7_phase1(ROOT, None, None, fee_rounding="daily").passed
    assert L.c7_phase1(ROOT, None, None).passed  # the golden file (the registry rows need the real store)


def test_c8(market) -> None:
    assert L.c8_accounting(market, [c for c in G.grid() if c.family == "core" and c.size == G.OWNER_SIZE]).passed
    assert not L.c8_accounting(market, list(L.C3_CONFIGS), uncapped_buys=True).passed


def test_c9(market) -> None:
    check = L.c9_reference(market)
    assert check.passed and check.detail["worst_relative_difference"] < 1e-10
    assert not L.c9_reference(market, shift=1).passed


def test_c10() -> None:
    assert L.c10_registration_lock().passed
    assert not L.c10_registration_lock(guard=False).passed


def test_c11(market, paths) -> None:
    assert L.c11_completeness(market, paths).passed
    assert not L.c11_completeness(market, paths, forward_fill_gap=True).passed


def test_c12() -> None:
    assert L.c12_one_implementation().passed
    assert not L.c12_one_implementation(overrides={"backtest/portfolio.py": "def plan_rebalance():\n    pass\n"}).passed
