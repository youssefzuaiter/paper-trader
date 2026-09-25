"""Non-blocking Alpaca REST client for the Risk & Routing agent — paper only.

Why not alpaca-py here: its ``TradingClient`` is built on ``requests`` and
blocks the calling thread for the whole round-trip. The monolith hides that
behind ``asyncio.to_thread``, which works but spends a worker thread per
in-flight call and cannot be cancelled mid-request. This client speaks
Alpaca's REST API directly through one pooled ``httpx.AsyncClient``, so an
order submission is a genuine ``await`` on a socket: the event loop keeps
serving signals, health checks and the exit monitor while Alpaca answers.

Three properties this module guarantees:

1. **Paper only.** The trading host is ``config.PAPER_BASE_URL``, a
   constant, and the constructor refuses any client that resolved
   elsewhere. There is no parameter that selects the live host.
2. **Exact numbers.** Responses are parsed with ``parse_float=Decimal`` and
   quantities/prices go out as decimal strings, so no price or quantity
   is ever a binary float — not even at the API boundary.
3. **At-most-once submission.** A timeout does not mean the order failed:
   the request may have reached Alpaca. Every retry is preceded by a
   lookup on the order's ``client_order_id``; an order that already exists
   is returned instead of being placed twice.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Final
from urllib.parse import urlsplit

import httpx

from config import PAPER_BASE_URL, get_broker_settings

logger = logging.getLogger("risk_router.alpaca")

DATA_BASE_URL: Final[str] = "https://data.alpaca.markets"
_PAPER_HOST: Final[str] = urlsplit(PAPER_BASE_URL).hostname or ""

_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(10.0, connect=3.0)
#: Alpaca's free plan allows 200 requests/minute; the router needs a few
#: per signal. A small pool bounds concurrency without queuing much.
_LIMITS: Final[httpx.Limits] = httpx.Limits(max_connections=10, max_keepalive_connections=5)

_SUBMIT_ATTEMPTS: Final[int] = 3
_RETRY_BASE_SECONDS: Final[float] = 0.5


class SandboxViolationError(RuntimeError):
    """The client resolved to a host other than the Alpaca paper sandbox."""


class AlpacaError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OrderRejected(AlpacaError):
    """Alpaca refused the order (a 4xx other than 429). Never retried."""


class OrderOutcomeUnknown(AlpacaError):
    """Every attempt failed in transit AND no order with this
    ``client_order_id`` could be found. The order was probably not
    placed, but the caller must not assume so: reconciliation will find
    it if it was."""


@dataclass(frozen=True, slots=True)
class LatestQuote:
    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: datetime


Sleep = Callable[[float], Awaitable[None]]


class AsyncAlpaca:
    """One pooled connection per Alpaca host, for the process lifetime.

    ``transport`` exists for tests (``httpx.MockTransport``); production
    never passes it.
    """

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        common: dict[str, Any] = {
            "headers": headers, "timeout": _TIMEOUT, "limits": _LIMITS, "transport": transport,
        }
        self._trading = httpx.AsyncClient(base_url=PAPER_BASE_URL, **common)
        self._data = httpx.AsyncClient(base_url=DATA_BASE_URL, **common)
        self._sleep = sleep

        # Same defence in depth as broker._assert_sandbox: check what the
        # client actually resolved to, not what we meant to pass it.
        if self._trading.base_url.host != _PAPER_HOST:
            raise SandboxViolationError(
                f"Refusing to trade: client resolved to {self._trading.base_url!s}, "
                f"expected the paper host {_PAPER_HOST}"
            )

    @classmethod
    def from_settings(cls) -> AsyncAlpaca:
        settings = get_broker_settings()  # raises ConfigError when unset
        return cls(settings.api_key_id, settings.api_secret_key)

    async def aclose(self) -> None:
        await asyncio.gather(self._trading.aclose(), self._data.aclose())

    # --- reads -------------------------------------------------------------

    async def _get(self, client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await client.get(path, params=params)
        if response.is_error:
            raise AlpacaError(
                f"GET {path} → HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        return response.json(parse_float=Decimal)

    async def account(self) -> dict[str, Any]:
        return await self._get(self._trading, "/v2/account")

    async def clock(self) -> dict[str, Any]:
        return await self._get(self._trading, "/v2/clock")

    async def positions(self) -> list[dict[str, Any]]:
        return await self._get(self._trading, "/v2/positions")

    async def orders(self, *, status: str, after: datetime | None = None, limit: int = 500) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"status": status, "limit": limit, "direction": "desc"}
        if after is not None:
            params["after"] = after.isoformat()
        return await self._get(self._trading, "/v2/orders", params)

    async def order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        response = await self._trading.get(
            "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id},
        )
        if response.status_code == 404:
            return None
        if response.is_error:
            raise AlpacaError(
                f"order lookup → HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        return response.json(parse_float=Decimal)

    async def latest_quote(self, symbol: str) -> LatestQuote:
        """Latest IEX quote — the free feed the paper keys already include."""
        body = await self._get(self._data, f"/v2/stocks/{symbol}/quotes/latest", {"feed": "iex"})
        quote = body["quote"]
        return LatestQuote(
            symbol=symbol,
            bid=Decimal(str(quote["bp"])),
            ask=Decimal(str(quote["ap"])),
            timestamp=datetime.fromisoformat(quote["t"]),
        )

    # --- writes ------------------------------------------------------------

    async def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one order, at most once, without blocking the event loop.

        Retries transport errors, 429 and 5xx with backoff — but before
        each retry, and on a duplicate-id 422, looks the order up by its
        ``client_order_id`` and returns it if Alpaca already has it. A
        lost response therefore never becomes a second order.

        Raises:
            OrderRejected: a 4xx other than 429 (bad request, insufficient
                buying power, …). Retrying would not change the answer.
            OrderOutcomeUnknown: every attempt failed and the lookup
                found nothing.
        """
        client_order_id = payload["client_order_id"]
        last_error = "not attempted"

        for attempt in range(1, _SUBMIT_ATTEMPTS + 1):
            if attempt > 1:
                # The previous attempt may have landed. Ask before re-sending.
                existing = await self._lookup_quietly(client_order_id)
                if existing is not None:
                    logger.warning("Order %s was placed by an earlier attempt; not re-sending", client_order_id)
                    return existing
                await self._sleep(_RETRY_BASE_SECONDS * 2 ** (attempt - 2))

            try:
                response = await self._trading.post("/v2/orders", json=payload)
            except httpx.TransportError as exc:  # timeouts included
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Order %s attempt %d/%d failed in transit: %s",
                               client_order_id, attempt, _SUBMIT_ATTEMPTS, last_error)
                continue

            if response.is_success:
                return response.json(parse_float=Decimal)

            body = response.text[:300]
            if response.status_code == 422 and "client_order_id" in body:
                existing = await self._lookup_quietly(client_order_id)
                if existing is not None:
                    return existing
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {body}"
                continue
            raise OrderRejected(f"Alpaca rejected order {client_order_id}: HTTP {response.status_code}: {body}",
                                status_code=response.status_code)

        existing = await self._lookup_quietly(client_order_id)
        if existing is not None:
            return existing
        raise OrderOutcomeUnknown(
            f"Order {client_order_id} not confirmed after {_SUBMIT_ATTEMPTS} attempts: {last_error}"
        )

    async def _lookup_quietly(self, client_order_id: str) -> dict[str, Any] | None:
        try:
            return await self.order_by_client_id(client_order_id)
        except (httpx.TransportError, AlpacaError) as exc:
            logger.warning("Lookup of order %s failed: %s", client_order_id, exc)
            return None

    async def cancel_all_orders(self) -> int:
        response = await self._trading.delete("/v2/orders")
        if response.is_error:
            raise AlpacaError(f"cancel all → HTTP {response.status_code}: {response.text[:200]}",
                              status_code=response.status_code)
        return len(response.json()) if response.content else 0
