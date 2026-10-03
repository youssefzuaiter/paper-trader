"""Pre-flight check for the long-term core: is everything in place for the first funded session?

    ./.venv/bin/python -m risk_router.core_preflight

Run it before the first evening, and again after any change to the policy, the keys or the host. It reads the
policy, the environment, the account, the calendar and the last published closes, and it prints the plan the
Allocator would propose for the latest close. It is **read-only**: it never submits an order, never changes the
account, and only reads the journal and the state files. The one thing it writes is a throwaway file, created and
deleted at once, to prove the state directory is writable. It never prints a secret, only the *name* of one that
is missing, too short or reused.

Exit status 0 means no check FAILed (WARNs are advice); 1 means something would stop the core from trading.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import httpx

import core_paper
import tier0_core
from risk_router.alpaca_async import AlpacaError, AsyncAlpaca
from risk_router.core_gatekeeper import CoreStore, Journal
from risk_router.state import StateStore

NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")
PASS: Final[str] = "PASS"
WARN: Final[str] = "WARN"
FAIL: Final[str] = "FAIL"
MIN_SECRET_CHARS: Final[int] = 32
DEFAULT_POLICY: Final[Path] = Path(__file__).resolve().parent.parent / "policy" / "core.toml"


@dataclass(frozen=True)
class Check:
    name: str
    status: str      # PASS | WARN | FAIL
    detail: str


def _secrets(env: Mapping[str, str]) -> list[Check]:
    """Names only, never values."""
    out: list[Check] = []
    missing = [k for k in ("CORE_ALPACA_KEY_ID", "CORE_ALPACA_SECRET_KEY") if not env.get(k, "").strip()]
    out.append(Check("alpaca keys", FAIL if missing else PASS,
                     f"not set: {', '.join(missing)}" if missing else "the core account's key id and secret are set"))
    plan = env.get("CORE_PLAN_SECRET", "").strip()
    control = env.get("WEBHOOK_SECRET", "").strip()
    short = [name for name, value in (("CORE_PLAN_SECRET", plan), ("WEBHOOK_SECRET", control))
             if len(value) < MIN_SECRET_CHARS]
    out.append(Check("signing secrets", FAIL if short else PASS,
                     f"missing or shorter than {MIN_SECRET_CHARS} characters: {', '.join(short)}" if short
                     else "CORE_PLAN_SECRET and WEBHOOK_SECRET are both set and long enough"))
    if plan and plan == control:
        out.append(Check("secrets are distinct", FAIL,
                         "CORE_PLAN_SECRET equals WEBHOOK_SECRET: the Allocator could then sign approvals and halts "
                         "as well as proposals. Use two different random values"))
    elif plan and control:
        out.append(Check("secrets are distinct", PASS, "the Allocator's secret cannot sign the owner's approvals"))
    if not env.get("CORE_ROUTER_URL", "").strip():
        out.append(Check("allocator → router url", WARN,
                         "CORE_ROUTER_URL is not set. The router does not need it; the Allocator does"))
    else:
        out.append(Check("allocator → router url", PASS, "CORE_ROUTER_URL is set"))
    if env.get("CORE_ALERT_WEBHOOK_URL", "").strip():
        out.append(Check("alerts", PASS, "CORE_ALERT_WEBHOOK_URL is set: problems and approvals will be pushed"))
    else:
        out.append(Check("alerts", WARN, "CORE_ALERT_WEBHOOK_URL is not set: nothing will tell you a plan is waiting "
                                         "for your approval, or that one failed. Set it (see docs/core-runbook.md)"))
    return out


def _state(state_dir: Path) -> list[Check]:
    out: list[Check] = []
    probe = state_dir
    while not probe.exists() and probe != probe.parent:    # a directory not made yet needs a writable parent
        probe = probe.parent
    try:
        with tempfile.NamedTemporaryFile(dir=probe, prefix=".preflight-"):
            pass
        out.append(Check("state directory", PASS, f"{state_dir} {'exists' if state_dir.exists() else 'will be created'} "
                                                  f"and is writable"))
    except OSError as exc:
        out.append(Check("state directory", FAIL, f"{state_dir} is not writable: {exc}"))
        return out
    journal_path = state_dir / "core-journal.jsonl"
    if journal_path.exists():
        journal = Journal(journal_path, CoreStore(state_dir / "core-plan-state.json"), "preflight")
        status = journal.status(repair=False)
        out.append(Check("journal", PASS if status["ok"] else FAIL,
                         f"{status['entries']} entries, chain intact" if status["ok"]
                         else f"integrity check failed: {status['reason']}"))
    else:
        out.append(Check("journal", PASS, "no journal yet (a fresh start)"))
    plan = CoreStore(state_dir / "core-plan-state.json").data.plan
    if plan:
        out.append(Check("plan in flight", WARN, f"plan {plan['id']} is {plan['status']} for {plan['execute_on']}"))
    if StateStore(state_dir / "core-router-state.json").halted:
        out.append(Check("kill switch", WARN, "the kill switch is ON: nothing will trade until it is cleared "
                                              "(delete core-router-state.json, then restart)"))
    else:
        out.append(Check("kill switch", PASS, "off"))
    return out


def _describe(plan: Any) -> str:
    parts = [f"sell {q} {s}" for s, q in sorted(plan.sells.items())] + \
            [f"buy ${n} {s}" for s, n in sorted(plan.buys.items())]
    return f"{plan.reason}: " + ", ".join(parts)


async def run_checks(broker: Any, policy: tier0_core.CorePolicy, *, env: Mapping[str, str], state_dir: Path,
                     now: datetime) -> list[Check]:
    """Every check, in the order a person would worry about them. ``broker`` is read from only."""
    checks: list[Check] = []
    problems = tier0_core.violations(policy)
    checks.append(Check("policy", FAIL if problems else PASS,
                        "; ".join(problems) if problems else
                        f"inside every tier-0 limit · rule {policy.rule.name} · sha256 {policy.sha256[:12]}… · "
                        f"effective {policy.effective_from}"))
    checks.extend(_secrets(env))
    checks.extend(_state(state_dir))
    if any(c.status == FAIL and c.name == "alpaca keys" for c in checks):
        return checks                                   # nothing below can run without the keys

    try:
        await _network(checks, broker, policy, now)
    except (AlpacaError, OSError, httpx.HTTPError) as exc:
        hint = " The key or secret is wrong, or revoked." if getattr(exc, "status_code", None) in (401, 403) else ""
        checks.append(Check("alpaca", FAIL, f"an Alpaca call failed: {exc}.{hint}"))
    return checks


async def _network(checks: list[Check], broker: Any, policy: tier0_core.CorePolicy, now: datetime) -> None:
    """The checks that need Alpaca. Appends as it goes, so a failure part-way keeps what already passed."""
    account = await broker.account()
    number = str(account.get("account_number", ""))
    if number != policy.account:
        checks.append(Check("account", FAIL, f"the keys belong to {number!r}; the policy names {policy.account!r}"))
    else:
        blocked = [k for k in ("trading_blocked", "account_blocked") if account.get(k)]
        inactive = account.get("status") not in (None, "ACTIVE")
        checks.append(Check("account", FAIL if blocked or inactive else PASS,
                            f"{number} is blocked or inactive ({blocked or account.get('status')})" if blocked or inactive
                            else f"{number} matches the policy and is tradable"))

    positions = await broker.positions()
    held = {p["symbol"]: Decimal(str(p["qty"])) for p in positions if Decimal(str(p["qty"])) != 0}
    foreign = sorted(set(held) - set(policy.mix))
    cash = Decimal(str(account.get("cash", "0")))
    if foreign:
        checks.append(Check("holdings", FAIL, f"positions outside the policy: {', '.join(foreign)}"))
    elif not held:
        if cash > policy.funded_amount_cap_usd:
            checks.append(Check("funding", FAIL, f"the account holds ${cash} but the policy caps the first build at "
                                                 f"${policy.funded_amount_cap_usd}: reset the paper account to that"))
        else:
            checks.append(Check("funding", PASS, f"empty account, ${cash} cash ≤ the ${policy.funded_amount_cap_usd} cap "
                                                 f"(the initial build will need your signed fund command)"))
    else:
        checks.append(Check("holdings", PASS, f"{len(held)} policy instruments held, ${cash} cash"))

    now_ny = now.astimezone(NEW_YORK)
    today = now_ny.date()
    calendar = await broker.calendar((today - timedelta(days=14)).isoformat(), (today + timedelta(days=14)).isoformat())
    days = sorted(date.fromisoformat(s["date"]) for s in calendar)
    upcoming = [d for d in days if d > today or (d == today and now_ny.time() < tier0_core.SUBMIT_FROM)]
    if upcoming:
        first = max(upcoming[0], policy.effective_from)
        early = upcoming[0] < policy.effective_from
        checks.append(Check("calendar", WARN if early else PASS,
                            f"next session {upcoming[0]}; the first order the policy may place is on {first}"
                            + (" (effective_from is later than the next session)" if early else "")))
    last = core_paper.last_published(days, now)
    if last is None:
        checks.append(Check("closes", FAIL, "no published session close in the last two weeks"))
        return
    try:
        snap = await core_paper.read_snapshot(broker, policy, last, forced=False, targets={})
    except core_paper.SnapshotError as exc:
        checks.append(Check("closes", FAIL, f"cannot read the {last} closes for every instrument ({exc.code}: "
                                            f"{exc.detail}). The Allocator needs SIP daily bars: check the data "
                                            f"entitlement"))
        return
    checks.append(Check("closes", PASS, f"{last} close published for all {len(policy.symbols)} instruments"))

    plan = core_paper.decide(policy, snap)
    if plan is None:
        reason = core_paper.skip_reason(policy, snap)
        checks.append(Check("dry run", PASS, f"for the {last} close the Allocator would propose no plan"
                                             + (f" ({reason})" if reason else " (nothing is due)")))
        return
    equity = snap.cash + sum((q * snap.closes[s] for s, q in snap.qty.items()), Decimal(0))
    rejected = tier0_core.plan_violations(plan, policy, prices=snap.closes, equity=equity, history=[], today=last,
                                          initial=snap.initial, funded=snap.initial)
    summary = f"if the Allocator ran for {last} it would propose → {_describe(plan)}"
    if rejected:
        checks.append(Check("dry run", FAIL, f"{summary}. The router would REJECT it: {'; '.join(rejected)}"))
    else:
        needs = " (waits for your signed approval)" if snap.initial or tier0_core.needs_approval(plan, policy, snap.closes) else ""
        checks.append(Check("dry run", PASS, summary + needs))


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{c.status:<4}  {c.name:<{width}}  {c.detail}" for c in checks]
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    lines.append("")
    lines.append("READY: nothing would stop the core from trading." if not failed
                 else f"NOT READY: {failed} check(s) FAILed.")
    if warned:
        lines.append(f"{warned} warning(s): advice, not blockers.")
    return "\n".join(lines)


async def main_async() -> int:
    env = dict(os.environ)       # config.py has already loaded .env (override=False) by the time it is imported
    policy_path = Path(env.get("CORE_POLICY_PATH", str(DEFAULT_POLICY)))
    try:
        policy = tier0_core.load_policy(policy_path)
    except (tier0_core.PolicyError, OSError) as exc:
        print(render([Check("policy", FAIL, f"{policy_path}: {exc}")]))
        return 1
    state_dir = Path(env.get("CORE_STATE_DIR", "core-state"))
    key, secret = env.get("CORE_ALPACA_KEY_ID", "").strip(), env.get("CORE_ALPACA_SECRET_KEY", "").strip()
    if not (key and secret):
        print(render(await run_checks(None, policy, env=env, state_dir=state_dir, now=datetime.now(UTC))))
        return 1
    alpaca = AsyncAlpaca(key, secret)
    try:
        checks = await run_checks(alpaca, policy, env=env, state_dir=state_dir, now=datetime.now(UTC))
    finally:
        await alpaca.aclose()
    print(render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
