"""Alpaca broker adapter — sandbox (paper) execution only.

Sandbox enforcement is structural, not configurable:

* ``PAPER_ONLY`` is a module-level ``Final[bool]`` set to ``True``.
* It is not a function parameter, not an environment variable, and not
  reachable from any request payload.
* ``_assert_sandbox`` re-checks the *resolved* client after construction, so
  a future edit that flips the flag fails loudly at startup instead of
  quietly routing real money.

There is no code path in this project that builds a live ``TradingClient``.
"""

from __future__ import annotations

import logging
import ssl
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Final

import certifi
from alpaca.common.enums import BaseURL
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.models import TradeAccount
from alpaca.trading.stream import TradingStream

from config import PAPER_BASE_URL, get_broker_settings

logger = logging.getLogger(__name__)

#: Hardcoded sandbox enforcement. Do not parameterise. Do not read from env.
PAPER_ONLY: Final[bool] = True

#: The only host this process is permitted to talk to. Cross-checked
#: against alpaca-py's own enum at import time, so a library change that
#: moved the paper host would fail here rather than silently routing
#: elsewhere.
_EXPECTED_BASE_URL: Final[str] = PAPER_BASE_URL
assert _EXPECTED_BASE_URL == BaseURL.TRADING_PAPER.value, (
    f"config.PAPER_BASE_URL ({PAPER_BASE_URL}) no longer matches alpaca-py's "
    f"BaseURL.TRADING_PAPER ({BaseURL.TRADING_PAPER.value})"
)


class SandboxViolationError(RuntimeError):
    """Raised when the resolved broker client is not pointed at the sandbox."""


def _assert_sandbox(client: TradingClient) -> None:
    """Verify the constructed client resolved to the paper host.

    Defence in depth against a future refactor silently flipping ``PAPER_ONLY``
    or an ``url_override`` sneaking in.
    """
    raw_base_url = getattr(client, "_base_url", "")
    base_url = raw_base_url.value if isinstance(raw_base_url, BaseURL) else str(raw_base_url or "")
    sandbox_flag = bool(getattr(client, "_sandbox", False))

    if not base_url.startswith(_EXPECTED_BASE_URL) or not sandbox_flag:
        raise SandboxViolationError(
            "Refusing to trade: resolved Alpaca client is not the paper sandbox "
            f"(base_url={base_url!r}, sandbox={sandbox_flag}). "
            f"Expected base_url={_EXPECTED_BASE_URL!r}, sandbox=True."
        )


@lru_cache(maxsize=1)
def get_trading_client() -> TradingClient:
    """Return the process-wide paper ``TradingClient``.

    Cached: the client is a thin, thread-safe HTTP wrapper, and rebuilding it
    per request would add a TLS handshake to every order.
    """
    settings = get_broker_settings()

    client = TradingClient(
        api_key=settings.api_key_id,
        secret_key=settings.api_secret_key,
        paper=PAPER_ONLY,  # hardcoded sandbox — see module docstring
    )

    _assert_sandbox(client)
    logger.info("Alpaca paper TradingClient ready (base_url=%s)", _EXPECTED_BASE_URL)
    return client


def get_account() -> TradeAccount:
    """Fetch the paper account snapshot (buying power, equity, status)."""
    account = get_trading_client().get_account()
    if getattr(account, "trading_blocked", False):
        logger.warning("Paper account has trading_blocked=True; orders will fail.")
    return account


def get_daily_pnl_pct() -> Decimal:
    """Today's account P&L as a percentage of yesterday's closing equity.

    ``(equity - last_equity) / last_equity * 100``, in ``Decimal``
    throughout — this app never lets money-adjacent arithmetic touch
    ``float`` (execution.py's own module docstring: "Money never touches
    float"), and a portfolio-level circuit breaker is exactly that kind
    of arithmetic. Both fields are typed ``Optional[str]`` by alpaca-py
    itself (confirmed against ``TradeAccount.model_fields``, not
    assumed) — ``None`` or a non-positive ``last_equity`` raises rather
    than dividing by zero or silently returning a meaningless 0%, since
    this value directly gates whether the autonomous loop keeps trading.

    Used by ``scheduler.trading_loop``'s circuit breaker check, against
    the hardcoded threshold in ``execution.CIRCUIT_BREAKER_DAILY_PNL_PCT``.
    """
    account = get_account()
    if account.equity is None or account.last_equity is None:
        raise ValueError(
            f"Cannot compute daily P&L: equity={account.equity!r}, last_equity={account.last_equity!r}"
        )

    equity = Decimal(account.equity)
    last_equity = Decimal(account.last_equity)
    if last_equity <= 0:
        raise ValueError(f"Cannot compute daily P&L: last_equity={last_equity!r} is not positive")

    return ((equity - last_equity) / last_equity) * Decimal(100)


#: The only WebSocket host this process is permitted to stream from —
#: same cross-check discipline as _EXPECTED_BASE_URL above.
_EXPECTED_STREAM_URL: Final[str] = BaseURL.TRADING_STREAM_PAPER.value


@lru_cache(maxsize=1)
def get_trading_stream() -> TradingStream:
    """Return the process-wide ``trade_updates`` WebSocket stream client.

    Same paper credentials as ``get_trading_client()``, same hardcoded
    ``paper=PAPER_ONLY`` sandbox enforcement. Verified against alpaca-py's
    own resolved endpoint after construction — the same defense-in-depth
    ``_assert_sandbox`` gives the REST client, reading the private
    ``_endpoint`` attribute since ``TradingStream`` (unlike
    ``TradingClient``) exposes no public accessor for it. Cached: this
    object holds the live WebSocket connection state once started, so a
    second call must return the SAME instance, never a second stream
    racing the first for Alpaca's single-connection-per-account slot.
    """
    settings = get_broker_settings()

    # This machine's Python `ssl` module can't find a local CA bundle for
    # raw TLS verification (verified live: connecting without this raises
    # `SSLCertVerificationError: unable to get local issuer certificate`)
    # — unlike httpx (used by get_trading_client()/get_market_data_client()
    # and webhook.py), which bundles/locates certifi automatically, the
    # `websockets` library TradingStream uses relies on the interpreter's
    # own default SSL context. Building one explicitly from certifi's CA
    # bundle fixes this portably, without requiring every developer to
    # separately run a one-off "Install Certificates.command"-style fix
    # on their own machine. `websocket_params`, if passed at all, REPLACES
    # the class's own default dict entirely (confirmed by reading
    # TradingStream.__init__) — so ping_interval/ping_timeout/max_queue
    # are restated here rather than only adding "ssl", or the protocol-
    # level heartbeat this module's own docstring relies on would be lost.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    stream = TradingStream(
        settings.api_key_id,
        settings.api_secret_key,
        paper=PAPER_ONLY,
        websocket_params={
            "ping_interval": 10,
            "ping_timeout": 180,
            "max_queue": 1024,
            "ssl": ssl_context,
        },
    )

    # NOT a plain str(...) cast — verified live that alpaca-py's BaseURL,
    # despite being a (str, Enum) mixin, still returns its qualified name
    # ("BaseURL.TRADING_STREAM_PAPER") from str(), not its value, unless
    # explicitly unwrapped — the exact bug _assert_sandbox above already
    # hit once for _base_url; same fix, same reason, applied here too.
    raw_endpoint = getattr(stream, "_endpoint", "")
    endpoint = raw_endpoint.value if isinstance(raw_endpoint, BaseURL) else str(raw_endpoint or "")
    if endpoint != _EXPECTED_STREAM_URL:
        raise SandboxViolationError(
            "Refusing to stream: resolved Alpaca TradingStream is not the paper "
            f"sandbox (endpoint={endpoint!r}). Expected endpoint={_EXPECTED_STREAM_URL!r}."
        )

    logger.info("Alpaca trade_updates TradingStream ready (endpoint=%s)", _EXPECTED_STREAM_URL)
    return stream


def is_market_open() -> bool:
    """True when the US equity market is currently open.

    Not used as an execution gate: Alpaca queues DAY orders placed outside
    regular hours, which is the behaviour we want for a paper agent.
    """
    return bool(get_trading_client().get_clock().is_open)


class RealQuoteUnavailable(RuntimeError):
    """Raised when Alpaca has no usable real quote for a symbol.

    Callers (``inference.fetch_quote``) catch this — alongside
    ``config.ConfigError`` when the broker isn't configured at all — and
    fall back to a fully synthetic quote rather than letting a single
    symbol's data gap crash a Tier-0 cycle.
    """


@lru_cache(maxsize=1)
def get_market_data_client() -> StockHistoricalDataClient:
    """Real-market quote client.

    Same paper-account API keys as ``get_trading_client()`` — Alpaca's
    free IEX feed needs no separate paid (SIP) market-data subscription.
    Used to price Tier-0 orders against the REAL market instead of a
    synthetic one, so a submitted limit order is actually marketable: a
    limit computed from a fake price can sit at half the real market ask
    and simply never fill (verified live — see the module docstring in
    ``inference.py`` for the incident this fixed).
    """
    settings = get_broker_settings()
    return StockHistoricalDataClient(settings.api_key_id, settings.api_secret_key)


def get_real_quote(ticker: str) -> tuple[Decimal, Decimal, datetime]:
    """Fetch Alpaca's real latest ``(bid, ask, timestamp)`` for ``ticker``.

    Raises:
        ConfigError: via ``get_market_data_client()``, if broker
            credentials aren't configured.
        RealQuoteUnavailable: if Alpaca has no quote for this symbol, or
            the request fails, or returns a non-positive/crossed quote.
    """
    client = get_market_data_client()
    try:
        response = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
        quote = response[ticker]
    except Exception as exc:  # network error, unknown symbol, malformed response
        raise RealQuoteUnavailable(f"No real quote available for {ticker!r}: {exc}") from exc

    bid, ask = quote.bid_price, quote.ask_price
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        raise RealQuoteUnavailable(f"Alpaca returned an unusable quote for {ticker!r}: bid={bid} ask={ask}")

    return Decimal(str(bid)), Decimal(str(ask)), quote.timestamp


class AtrUnavailable(RuntimeError):
    """Raised when Alpaca has no usable historical bars to compute an
    ATR from for a symbol.

    Callers (``inference.fetch_quote``) catch this alongside
    ``config.ConfigError`` and fall back to no-ATR — volatility-parity
    sizing in ``execution.py`` then degrades to the notional-cap-only
    figure — rather than let a single symbol's data gap crash a Tier-0
    cycle, the same graceful-degradation posture ``RealQuoteUnavailable``
    already has for the quote path.
    """


#: The 14-day window this feature is named for (ad hoc, Phase 4).
_ATR_PERIOD_DAYS: Final[int] = 14

#: Calendar days of history to request — always noticeably more than
#: _ATR_PERIOD_DAYS since weekends/holidays mean calendar days
#: outnumber TRADING days; 30 comfortably covers 15+ trading days even
#: across a long weekend.
_ATR_LOOKBACK_CALENDAR_DAYS: Final[int] = 30


def get_atr(ticker: str, period: int = _ATR_PERIOD_DAYS) -> Decimal:
    """Fetch real daily bars and compute the `period`-day Average True
    Range for ``ticker``.

    True Range for one trading day is
    ``max(high-low, |high-prevClose|, |low-prevClose|)`` — the largest of
    three ways a day's range can be measured, which is what correctly
    captures a gap up/down from the prior close, not just that day's own
    high-low spread. ATR here is a plain arithmetic mean of the last
    `period` daily True Range values — a stated simplification: Wilder's
    original smoothing (an exponential moving average) is a real
    refinement not implemented here, not silently assumed equivalent.

    Uses the free-tier IEX feed explicitly (``DataFeed.IEX``) — the same
    feed ``get_real_quote`` already relies on; verified live that
    omitting this raises ``"subscription does not permit querying recent
    SIP data"`` on this account's plan, the same class of feed-selection
    trap ``get_real_quote``'s own history already established.

    Raises:
        ConfigError: via ``get_market_data_client()``, if broker
            credentials aren't configured.
        AtrUnavailable: if Alpaca has fewer than `period` + 1 bars for
            this symbol (need `period` + 1 closes to derive `period` true
            ranges), or the request fails outright.
    """
    client = get_market_data_client()
    end = datetime.now(UTC)
    start = end - timedelta(days=_ATR_LOOKBACK_CALENDAR_DAYS)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = list(client.get_stock_bars(request)[ticker])
    except Exception as exc:  # network error, unknown symbol, malformed response
        raise AtrUnavailable(f"No historical bars available for {ticker!r}: {exc}") from exc

    if len(bars) < period + 1:
        raise AtrUnavailable(
            f"Only {len(bars)} bar(s) available for {ticker!r} in the last "
            f"{_ATR_LOOKBACK_CALENDAR_DAYS} days — need at least {period + 1} "
            f"to compute a {period}-day ATR"
        )

    true_ranges: list[Decimal] = []
    for i in range(1, len(bars)):
        high = Decimal(str(bars[i].high))
        low = Decimal(str(bars[i].low))
        prev_close = Decimal(str(bars[i - 1].close))
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    last_n = true_ranges[-period:]
    return sum(last_n) / Decimal(len(last_n))
