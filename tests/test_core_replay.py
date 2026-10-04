"""Paper mode against the registered engine, on real history.

The unit tests prove each piece of the paper router; ``C12`` proves it calls the engine's own functions. This
proves the *whole pipeline over time* does what the evidence assumed: the real ``CoreRouter`` and the shared
decision are driven session by session through two years of cached market data (2022-2023, which includes the
year stocks and bonds fell together), and its wealth path and rebalance dates are compared with
``portfolio.simulate``, the engine the registered study ran, at the frictionless level on the same prices.

Both sides see one price series (adjusted prices stand in for "raw" ones, so there are no cash dividends to
differ on), the same calendar (holidays included, so quarter-end detection is tested on real days), and no costs.
What is left to differ is the router's own mechanics: cent-rounded buy notionals, the cash it holds back, orders
sent as limit orders, and the plan being approved and tracked through its states. Needs the fetched caches
(``.cache/backtest/core``); skipped without them, as the other real-data tests are.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from test_core_paper import SESSIONS, Clock, FakeBroker, make_router, policy, proposal

from backtest import core_fetch, core_runs, portfolio
from backtest import core_grid as G
from backtest.costs import FRICTIONLESS, FeeTable

D = Decimal
ROOT = Path(__file__).resolve().parents[1]
REAL = not core_fetch.missing(core_runs.core_paths(ROOT))
WINDOW = (date(2022, 1, 3), date(2023, 12, 29))
#: The router keeps a $1 reserve and a few cents of allowance idle and sizes buys to the cent, and the engine at the
#: frictionless level does none of that: a few cents on $10,000 (about 3e-5). Anything bigger is a real difference.
TOLERANCE = 1e-4
#: The engine run without the loss breaker: FakeBroker's equity never moves, so the router's breaker never fires either.
NO_BREAKER = portfolio.Profile("replay-no-breaker", "core", None, False, False, True, True)


@pytest.mark.skipif(not REAL, reason="needs the fetched core caches under .cache/backtest/core")
@pytest.mark.asyncio
async def test_the_paper_router_replays_the_registered_engine_on_two_years_of_real_history(tmp_path: Path) -> None:
    market = core_runs.load_market(ROOT)
    first, last = market.index(WINDOW[0]), market.index(WINDOW[1])
    spec = G.Config("buffer", "full", "M3", "quarterly", "central", G.OWNER_SIZE, buffer=D("0.05")).spec()
    engine = portfolio.simulate(market, spec, first, last, FRICTIONLESS, FeeTable.load(), NO_BREAKER)
    assert not any(day.deferred for day in engine), "the engine's breaker would differ from the fake account's"

    # the policy and the registered configuration agree on everything the two sides share
    pol = replace(policy(), effective_from=date(2000, 1, 1))
    assert pol.mix == spec.weights(None)
    assert pol.cash_reserve_usd == spec.cash_reserve_usd and pol.min_order_usd == spec.min_order_usd

    symbols = list(pol.symbols)
    backup = list(SESSIONS)
    SESSIONS[:] = [market.dates[k].isoformat() for k in range(first - 1, min(last + 4, len(market.dates)))]
    try:
        broker, clock = FakeBroker(cash=str(G.OWNER_SIZE)), Clock(datetime(2022, 1, 1, tzinfo=UTC))
        router = make_router(tmp_path, broker, pol=pol, clock=clock)
        replay: dict[date, Decimal] = {}
        executed: set[date] = set()
        for k in range(first - 1, last):
            decided, executes = market.dates[k], market.dates[k + 1]
            broker.closes = {s: market.close(s, k) for s in symbols}                    # the close the Allocator reads
            clock.set(decided.isoformat(), 16, 30)
            answer = await router.receive_plan(await proposal(router, decided.isoformat()))
            if answer["status"] == "awaiting_approval":
                await router.approve(answer["plan_id"], fund=router.store.data.plan["kind"] == "initial")
            if router.store.data.plan is not None:
                assert router.store.data.plan["execute_on"] == executes.isoformat()
                broker.closes = {s: market.open(s, k + 1) for s in symbols}              # orders fill at the open
                clock.set(executes.isoformat(), 9, 22)
                await router.tick()
                broker.fill_all("sell")
                clock.set(executes.isoformat(), 9, 40)
                await router.tick()
                broker.fill_all("buy")
                clock.set(executes.isoformat(), 9, 50)
                await router.tick()
                assert router.store.data.plan is None, f"the plan decided {decided} did not finish"
                executed.add(executes)
            replay[executes] = broker.cash + sum((q * market.close(s, k + 1) for s, q in broker.qty.items()), D(0))
    finally:
        SESSIONS[:] = backup

    worst = max(abs(replay[day.date] / day.value - 1) for day in engine)
    assert worst < TOLERANCE, f"the paper router's wealth path leaves the engine's by {worst:.2e} of the portfolio"
    assert executed == {day.date for day in engine if day.rebalanced}, "different days were traded"
    assert len(executed) >= 8, "the window should hold the initial build and at least seven quarter-ends"
    print(f"\nreplayed {len(replay)} sessions, {len(executed)} trading days; worst daily gap {worst:.2e}")
