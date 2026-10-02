"""The shekel translation: rate alignment on NYSE sessions, and the shekel return arithmetic."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from backtest.core_fx import ROWS, rates_on, shekel_returns


def test_each_session_uses_its_own_rate_else_the_last_one_published_before_it() -> None:
    published = {date(2025, 4, 10): 3.70, date(2025, 4, 11): 3.72, date(2025, 4, 14): 3.75}
    sessions = [date(2025, 4, 10), date(2025, 4, 11), date(2025, 4, 14), date(2025, 4, 15)]
    rates, carried = rates_on(sessions, published)
    assert rates == [3.70, 3.72, 3.75, 3.75] and carried == [date(2025, 4, 15)]
    with pytest.raises(ValueError):
        rates_on([date(2025, 4, 9)], published)  # nothing published yet: never back-filled from the future


def test_shekel_returns_combine_the_portfolio_and_the_currency() -> None:
    usd = [Decimal(10100), Decimal(10100), Decimal(9090)]
    r = shekel_returns(usd, Decimal(10000), 3.0, [3.0, 3.3, 3.3])
    assert r == pytest.approx(np.array([0.01, 0.10, -0.10]))
    assert float(np.prod(1 + r)) == pytest.approx(9090 * 3.3 / 30000)


def test_the_rows_are_the_registered_configurations() -> None:
    assert len(ROWS) == 16 and ("M3", "monthly_band5", Decimal(0)) in ROWS
