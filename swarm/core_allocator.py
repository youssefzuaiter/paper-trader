"""The long-term core's Allocator (core design §8.4, system design §7.2): proposes, never trades.

Once per session, after the close (``tier0_core.DECIDE_AFTER``, 16:20 New York, from published bars):

1. ask the core router for the facts only it knows (was the last plan deferred? the last targets?);
2. read the core account and the session's closes with its own connection;
3. decide with ``core_paper.decide``, the backtest's ``core_alloc`` arithmetic;
4. sign the proposal (``CORE_PLAN_SECRET``) and send it. The router recomputes it independently and
   rejects any difference; this process has no way to place an order.

Run it once per session, after 16:20 New York, e.g. a Kubernetes CronJob::

    python -m swarm.core_allocator               # today's session
    python -m swarm.core_allocator --session 2026-10-05
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx

import core_paper
import tier0_core
import webhook
from risk_router.alpaca_async import AsyncAlpaca

logger = logging.getLogger("swarm.core_allocator")

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")
DEFAULT_POLICY: Final[Path] = Path(__file__).resolve().parent.parent / "policy" / "core.toml"


def signed_headers(body: bytes, secret: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    return {webhook.TIMESTAMP_HEADER: timestamp, webhook.SIGNATURE_HEADER: webhook.sign(body, timestamp, secret),
            "Content-Type": "application/json"}


async def propose(alpaca: Any, http: httpx.AsyncClient, policy: tier0_core.CorePolicy, session: date,
                  secret: str) -> dict[str, Any]:
    """One decision for ``session``; returns the router's answer."""
    state = await http.get("/v1/core/state", headers=signed_headers(b"", secret))
    state.raise_for_status()
    facts = state.json()
    snap = await core_paper.read_snapshot(alpaca, policy, session, forced=bool(facts["forced"]),
                                          targets={s: Decimal(w) for s, w in facts["targets"].items()})
    plan = core_paper.decide(policy, snap)
    body = json.dumps({"session": session.isoformat(), "inputs": snap.digest(),
                       "plan": core_paper.plan_to_json(plan) if plan else None}, sort_keys=True).encode()
    response = await http.post("/v1/core/plans", content=body, headers=signed_headers(body, secret))
    answer = {"http_status": response.status_code, **(response.json() if response.content else {})}
    logger.info("session %s: %s", session, answer)
    return answer


def default_session(now: datetime) -> date:
    local = now.astimezone(NEW_YORK)
    if local.time() < tier0_core.DECIDE_AFTER:
        raise SystemExit(f"too early: decisions use published closes, after {tier0_core.DECIDE_AFTER} New York")
    return local.date()


async def main_async(session: date | None) -> int:
    policy = tier0_core.load_policy(Path(os.getenv("CORE_POLICY_PATH", str(DEFAULT_POLICY))))
    problems = tier0_core.violations(policy)
    if problems:
        logger.error("policy outside the tier-0 limits, not proposing: %s", problems)
        return 2
    secret = os.environ["CORE_PLAN_SECRET"]
    alpaca = AsyncAlpaca(os.environ["CORE_ALPACA_KEY_ID"], os.environ["CORE_ALPACA_SECRET_KEY"])
    try:
        async with httpx.AsyncClient(base_url=os.environ["CORE_ROUTER_URL"], timeout=60.0) as http:
            answer = await propose(alpaca, http, policy, session or default_session(datetime.now(UTC)), secret)
    finally:
        await alpaca.aclose()
    print(json.dumps(answer, indent=1, default=str))
    return 0 if answer["http_status"] == 200 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m swarm.core_allocator", description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", type=date.fromisoformat, default=None)
    return asyncio.run(main_async(parser.parse_args(argv).session))


if __name__ == "__main__":
    sys.exit(main())
