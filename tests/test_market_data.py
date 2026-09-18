"""broker.get_latest_prices — the batched IEX mark behind ``/control/quotes``.

A fake data client stands in for alpaca-py: what's under test is the
selection (only symbols Alpaca answered for, only usable prices), the
normalisation (uppercase, de-duplicated, Decimal via str) and the
"one unknown symbol never fails the batch" contract — not HTTP.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from alpaca.data.enums import DataFeed

from broker import get_latest_prices


class FakeDataClient:
    def __init__(self, trades: dict[str, Any]) -> None:
        self.trades = trades
        self.requests: list[Any] = []

    def get_stock_latest_trade(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {symbol: trade for symbol, trade in self.trades.items() if symbol in request.symbol_or_symbols}


def _trade(price: float | None, ts: datetime | None = datetime(2026, 9, 18, 15, 30, tzinfo=UTC)) -> Any:
    return SimpleNamespace(price=price, timestamp=ts)


def test_returns_only_symbols_alpaca_answered_for_with_decimal_prices() -> None:
    client = FakeDataClient({"TSLA": _trade(362.25), "GOOGL": _trade(165.4)})
    prices = get_latest_prices(["TSLA", "GOOGL", "ZZZZ"], client=client)

    assert set(prices) == {"TSLA", "GOOGL"}  # ZZZZ simply absent, never an error
    assert prices["TSLA"].price == Decimal("362.25")
    assert prices["TSLA"].timestamp == datetime(2026, 9, 18, 15, 30, tzinfo=UTC)


def test_normalises_case_deduplicates_and_requests_the_iex_feed_once() -> None:
    client = FakeDataClient({"TSLA": _trade(362.25)})
    prices = get_latest_prices(["tsla", "TSLA", " ", ""], client=client)

    assert list(prices) == ["TSLA"]
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.symbol_or_symbols == ["TSLA"]
    assert request.feed == DataFeed.IEX


def test_drops_unusable_prices_and_missing_timestamps() -> None:
    client = FakeDataClient({"A": _trade(0), "B": _trade(-1.5), "C": _trade(None), "D": _trade(10, ts=None), "E": _trade(10)})
    prices = get_latest_prices(["A", "B", "C", "D", "E"], client=client)
    assert list(prices) == ["E"]


def test_empty_input_makes_no_request() -> None:
    client = FakeDataClient({"TSLA": _trade(362.25)})
    assert get_latest_prices([], client=client) == {}
    assert client.requests == []
