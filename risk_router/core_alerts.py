"""Push alerts for the long-term core: optional, quick to set up, and unable to hurt the router.

The core is built to run unattended, which only works if it can reach its owner. Two kinds of event need a
human: something went differently from the plan (a plan abandoned, a mismatch, a refused order, a breaker
that blocked the buys), and something is waiting for the owner (the initial build and every large or
raise-cash plan need a signed approval before they can execute).

Set ``CORE_ALERT_WEBHOOK_URL`` and the journal's notable events are pushed there. Two formats:

* ``json`` (default): ``{"title", "message", "text", "content"}``, which Slack, Discord and Mattermost
  incoming webhooks all accept.
* ``ntfy``: the plain-text body and ``Title`` header that https://ntfy.sh topics take. ntfy needs no account:
  pick a topic name nobody will guess and subscribe to it in the phone app.

Nothing here can raise into, or wait on, the router: a send is a fire-and-forget task with a short timeout,
and a failed one is logged and dropped. The journal stays the record; this only makes sure someone looks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, Final

import httpx

logger = logging.getLogger("risk_router.core_alerts")

#: Journal events that always warrant a push. (``plan_accepted`` and ``plan_done`` are conditional.)
ALERT_EVENTS: Final[frozenset[str]] = frozenset({
    "plan_mismatch", "plan_rejected", "plan_abandoned", "plan_expired", "plan_deferred", "plan_halted",
    "order_rejected", "buy_blocked", "sells_cancelled_at_deadline", "buys_cancelled_at_deadline",
    "tick_failed", "journal_invalid",
})

TITLES: Final[dict[str, str]] = {
    "plan_mismatch": "Core: the router disagrees with the Allocator",
    "plan_rejected": "Core: a plan broke a limit",
    "plan_abandoned": "Core: a rebalance was abandoned part-way",
    "plan_expired": "Core: a plan missed its session",
    "plan_deferred": "Core: the breaker deferred a plan",
    "plan_halted": "Core: the kill switch stopped a plan",
    "plan_done": "Core: a plan finished with unfilled orders",
    "order_rejected": "Core: Alpaca refused an order",
    "buy_blocked": "Core: the buys were blocked after the open",
    "sells_cancelled_at_deadline": "Core: sells did not fill in time",
    "buys_cancelled_at_deadline": "Core: buys did not fill in time",
    "tick_failed": "Core: the execution loop keeps failing",
    "journal_invalid": "Core: the journal failed its integrity check",
    "plan_accepted": "Core: a plan needs your approval",
}

#: The only fields of a journal entry that go into a message: no payloads, no order bodies, nothing secret.
_SHOWN: Final[tuple[str, ...]] = ("plan_id", "kind", "execute_on", "session", "reason", "code", "detail",
                                  "problems", "error", "consecutive", "symbol", "inputs_match")
_MAX_MESSAGE: Final[int] = 600


def should_alert(event: str, detail: Mapping[str, Any]) -> bool:
    if event in ALERT_EVENTS:
        return True
    if event == "plan_accepted":
        return detail.get("status") == "awaiting_approval"
    if event == "plan_done":      # a rebalance that finished but did not fully fill: drift will be re-decided
        return any(o.get("status") != "filled" for o in (detail.get("orders") or {}).values())
    return False


def render(event: str, detail: Mapping[str, Any]) -> tuple[str, str]:
    """A short title and message for one journal entry, from a fixed list of safe fields."""
    title = TITLES.get(event, f"Core: {event.replace('_', ' ')}")
    lines = [f"{key}: {detail[key]}" for key in _SHOWN if detail.get(key) not in (None, "", [], {})]
    if event == "plan_done":
        orders = detail.get("orders") or {}
        unfilled = sum(1 for o in orders.values() if o.get("status") != "filled")
        lines.append(f"{unfilled} of {len(orders)} orders did not fill")
    if event == "plan_accepted":
        lines.append("approve it with the signed control call (see docs/core-runbook.md)")
    return title, "\n".join(lines)[:_MAX_MESSAGE] or title


class Alerter:
    """Sends a message to the owner's webhook. ``Alerter(None)`` is a valid, silent alerter."""

    def __init__(self, url: str | None, fmt: str = "json", *, timeout: float = 5.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = url or None
        self.fmt = fmt if fmt in ("json", "ntfy") else "json"
        self.timeout = timeout
        self._transport = transport
        self._tasks: set[asyncio.Task[bool]] = set()   # strong references: a bare task can be collected mid-send
        if self.url and not self.url.startswith("https://") and "localhost" not in self.url:
            logger.warning("CORE_ALERT_WEBHOOK_URL is not https: alerts will cross the network unencrypted")

    @classmethod
    def from_env(cls) -> Alerter:
        return cls(os.getenv("CORE_ALERT_WEBHOOK_URL", "").strip() or None,
                   os.getenv("CORE_ALERT_FORMAT", "json").strip().lower())

    async def send(self, title: str, message: str) -> bool:
        if not self.url:
            return False
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                if self.fmt == "ntfy":
                    response = await client.post(self.url, content=message.encode("utf-8"), headers={"Title": title})
                else:
                    response = await client.post(self.url, json={"title": title, "message": message,
                                                                 "text": f"{title}\n{message}",
                                                                 "content": f"**{title}**\n{message}"})
                response.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001 - an alert must never be the thing that breaks the router
            logger.warning("core alert not delivered (%s): %s", title, exc)
            return False

    def notify(self, title: str, message: str) -> None:
        """Fire and forget. A no-op without a URL or outside a running event loop."""
        if not self.url:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.send(title, message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
