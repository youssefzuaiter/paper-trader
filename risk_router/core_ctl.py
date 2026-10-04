"""The owner's console for the long-term core: status, approvals, raising cash, and the kill switch.

    ./.venv/bin/python -m risk_router.core_ctl status
    ./.venv/bin/python -m risk_router.core_ctl approve [PLAN_ID] [--fund]
    ./.venv/bin/python -m risk_router.core_ctl raise-cash --amount 500 --need-by 2026-11-15
    ./.venv/bin/python -m risk_router.core_ctl halt
    ./.venv/bin/python -m risk_router.core_ctl test-alert

The core asks for a human in three places: the initial build (a signed *fund* command), any plan that trades more
than the policy's approval threshold, and every raise-cash plan. Those are signed HTTP calls, and this is the
tool that makes them, so the approval gate is something a person can actually use.

Each command signs its request with ``WEBHOOK_SECRET`` (the owner's secret: the Allocator's ``CORE_PLAN_SECRET``
is different and cannot sign these) and sends it to ``CORE_ROUTER_URL``. A command that changes anything shows
what it will do first and asks, unless ``--yes`` is given. Secrets are read from the environment (``.env``) and
never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

import webhook
from risk_router.core_alerts import Alerter


def signed_headers(body: bytes, secret: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    return {webhook.TIMESTAMP_HEADER: timestamp, webhook.SIGNATURE_HEADER: webhook.sign(body, timestamp, secret),
            "Content-Type": "application/json"}


def _explain(response: httpx.Response) -> str:
    """The router's own words for why it said no."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text[:300]
    if isinstance(detail, dict):
        return f"{detail.get('code', '')}: {detail.get('detail', '')}".strip(": ")
    return str(detail)


def describe(plan: dict[str, Any]) -> list[str]:
    lines = [f"plan {plan['id']}  ·  {plan['kind']}  ·  {plan['status']}  ·  decided {plan['decided_on']}, "
             f"executes {plan['execute_on']} at the open" + ("  ·  re-decision" if plan.get("redecision") else "")]
    lines += [f"    sell {qty} {symbol}" for symbol, qty in sorted(plan["sells"].items())]
    lines += [f"    buy  ${notional} of {symbol}" for symbol, notional in sorted(plan["buys"].items())]
    for order in plan.get("orders", {}).values():
        lines.append(f"    {order['side']} {order['symbol']}: {order['status']}")
    return lines


async def _signed_get(http: httpx.AsyncClient, path: str, secret: str) -> httpx.Response:
    return await http.get(path, headers=signed_headers(b"", secret))


async def _signed_post(http: httpx.AsyncClient, path: str, secret: str, payload: dict[str, Any] | None) -> httpx.Response:
    body = json.dumps(payload, sort_keys=True).encode() if payload is not None else b""
    return await http.post(path, content=body, headers=signed_headers(body, secret))


def _rejected_signature(response: httpx.Response) -> bool:
    return response.status_code == 403


async def execute(args: argparse.Namespace, http: httpx.AsyncClient, secret: str, confirm: Callable[[str], bool],
                  out: Callable[[str], None] = print) -> int:
    """Run one command against the router; the process's exit status."""
    if args.command == "status":
        health = (await http.get("/health")).json()
        out(f"status: {health['status']}  ·  trading {'enabled' if health['trading_enabled'] else 'DISABLED'}  ·  "
            f"{'HALTED' if health['halted'] else 'not halted'}  ·  policy {str(health['policy_sha256'])[:12]}…")
        for line in health["attention"]:
            out(f"  ! {line}")
        shown = await _signed_get(http, "/v1/core/plan", secret)
        if _rejected_signature(shown):
            out("the router rejected the signature: WEBHOOK_SECRET here must equal the router's")
            return 1
        if shown.status_code == 200 and shown.json()["plan"]:
            for line in describe(shown.json()["plan"]):
                out(line)
        elif shown.status_code == 200:
            out("no plan open" + ("; the next close will re-decide the last one" if shown.json()["forced"] else ""))
        return 0

    if args.command == "approve":
        shown = await _signed_get(http, "/v1/core/plan", secret)
        if _rejected_signature(shown):
            out("the router rejected the signature: WEBHOOK_SECRET here must equal the router's")
            return 1
        if shown.status_code != 200:
            out(f"the router cannot show the plan: {_explain(shown)}")
            return 1
        plan = shown.json()["plan"]
        if plan is None or plan["status"] != "awaiting_approval":
            out("nothing is waiting for your approval" + (f" (plan {plan['id']} is {plan['status']})" if plan else ""))
            return 1
        if args.plan_id and args.plan_id != plan["id"]:
            out(f"the plan waiting for approval is {plan['id']}, not {args.plan_id}")
            return 1
        for line in describe(plan):
            out(line)
        if plan["kind"] == "initial" and not args.fund:
            out("this is the initial build. It spends the account's cash: add --fund to sign the fund command")
            return 1
        if not (args.yes or confirm(f"Approve plan {plan['id']}? [y/N] ")):
            out("not approved")
            return 1
        response = await _signed_post(http, f"/v1/core/plans/{plan['id']}/approve", secret, {"fund": bool(args.fund)})
        out(f"approved: it will execute at the open on {plan['execute_on']}" if response.status_code == 200
            else f"not approved: {_explain(response)}")
        return 0 if response.status_code == 200 else 1

    if args.command == "raise-cash":
        try:
            amount = Decimal(args.amount)
        except InvalidOperation:
            out(f"--amount must be a number of dollars, not {args.amount!r}")
            return 1
        response = await _signed_post(http, "/v1/core/raise-cash", secret,
                                      {"amount": str(amount), "need_by": args.need_by})
        if response.status_code != 200:
            out(f"refused: {_explain(response)}")
            return 1
        out(f"a plan to raise ${amount} by {args.need_by} was made and is waiting for your approval; "
            f"run `approve` to see its orders and sign it")
        return 0

    if args.command == "halt":
        if not (args.yes or confirm("Stop ALL core trading and cancel open orders? [y/N] ")):
            out("not halted")
            return 1
        response = await _signed_post(http, "/v1/control/halt", secret, None)
        out(f"halted; {response.json().get('orders_canceled', 0)} open order(s) cancelled. To resume, delete "
            f"core-router-state.json on the router's host and restart it" if response.status_code == 200
            else f"the halt FAILED: {_explain(response)}")
        return 0 if response.status_code == 200 else 1

    if args.command == "test-alert":      # local: needs no router and no signature
        alerter = Alerter.from_env()
        if not alerter.url:
            out("CORE_ALERT_WEBHOOK_URL is not set: there is nowhere to send an alert")
            return 1
        delivered = await alerter.send("Core: test alert", "If you can read this, alerts reach you.")
        out("delivered: check your phone or channel" if delivered
            else f"NOT delivered to {alerter.url.split('//')[-1].split('/')[0]}: see the log line above for why")
        return 0 if delivered else 1

    out(f"unknown command {args.command!r}")
    return 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m risk_router.core_ctl", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="what the router is doing, and what is waiting for you")
    approve = sub.add_parser("approve", help="sign off the plan that is waiting for approval")
    approve.add_argument("plan_id", nargs="?", help="check that this is the plan waiting (optional)")
    approve.add_argument("--fund", action="store_true", help="sign the fund command (required for the initial build)")
    approve.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    cash = sub.add_parser("raise-cash", help="ask for a sells-only plan that raises cash by a date")
    cash.add_argument("--amount", required=True, help="dollars to raise, net of costs")
    cash.add_argument("--need-by", required=True, help="the date the cash is needed, YYYY-MM-DD")
    halt = sub.add_parser("halt", help="the kill switch: stop all core trading and cancel open orders")
    halt.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    sub.add_parser("test-alert", help="send a test message to CORE_ALERT_WEBHOOK_URL (needs no router)")
    return p


def _ask(question: str) -> bool:
    return input(question).strip().lower() in ("y", "yes")


async def main_async(argv: list[str] | None) -> int:
    args = parser().parse_args(argv)
    if args.command == "test-alert":
        return await execute(args, None, "", _ask)      # type: ignore[arg-type]  # local: no router involved
    secret = os.getenv("WEBHOOK_SECRET", "").strip()
    url = os.getenv("CORE_ROUTER_URL", "").strip()
    if len(secret) < 32 or not url:
        print("WEBHOOK_SECRET (32+ characters) and CORE_ROUTER_URL must be set, in the environment or in .env")
        return 1
    try:
        async with httpx.AsyncClient(base_url=url, timeout=30.0) as http:
            return await execute(args, http, secret, _ask)
    except httpx.HTTPError as exc:
        print(f"could not reach the core router at {url}: {exc}")
        return 1


def main() -> int:
    return asyncio.run(main_async(None))


if __name__ == "__main__":
    sys.exit(main())
