"""Tier-0 hard limits for the long-term core's paper mode (core design §8.3), and its policy file.

The core's counterpart of ``tier0.py``, which stays untouched for news trading. Every limit here is a
hardcoded module ``Final``: not read from ``.env``, not a function parameter, not reachable from a
request. The owner's policy file (``policy/core.toml``) may only choose values *inside* these limits;
a policy beyond any of them disables core trading (fail closed).

Unchanged and imported by the core router, never copied: the kill switch and daily loss breaker
(``risk_router.guards``), the paper-only lock (``risk_router.alpaca_async``), the outbox.

What is deliberately *not* here: stop-losses, take-profits and session exits. Core positions are
held for years; the news router's exit monitor never touches this account.

Pure standard library, no I/O beyond reading the policy file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

import tomllib

import core_alloc

# ===========================================================================
# TIER-0 CORE LIMITS — do not parameterise, do not move to .env
# ===========================================================================

#: The registered instruments only (core design §2): the primaries, their substitutes, BTC/USD.
ALLOWED_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"VTI", "VXUS", "BND", "IAU", "VNQ", "BIL", "SPY", "IEF", "TLT", "GLD", "SGOV", "BTC/USD"})
CRYPTO_SYMBOLS: Final[frozenset[str]] = frozenset({"BTC/USD"})
MAX_CRYPTO_WEIGHT_PCT: Final[Decimal] = Decimal(10)
#: One-way traded notional of one rebalance, as a share of equity. The initial build is exempt only
#: through the owner's signed ``fund`` command.
MAX_TURNOVER_PER_REBALANCE_PCT: Final[Decimal] = Decimal(25)
MAX_ORDER_NOTIONAL_USD: Final[Decimal] = Decimal(100000)
MAX_FUNDED_AMOUNT_USD: Final[Decimal] = Decimal(100000)
#: Accepted rebalance plans per calendar month; a re-decision after a breaker deferral is not a new plan.
MAX_REBALANCE_PLANS_PER_MONTH: Final[int] = 2
#: System design §7.4's turnover cap: trades and traded value per rolling seven days.
MAX_TRADES_PER_WEEK: Final[int] = 40
MAX_TRADE_VALUE_PCT_PER_WEEK: Final[Decimal] = Decimal(100)
MAX_APPROVAL_THRESHOLD_USD: Final[Decimal] = Decimal(100000)
#: Pre-open limit orders sit at the previous close ± this: wide enough to fill at the opening print on
#: nearly every day, narrow enough to refuse a wild print (system design §7.4: limit orders only).
LIMIT_COLLAR_PCT: Final[Decimal] = Decimal(3)
#: Buys placed after the sells fill: a marketable limit at the ask + this.
MARKETABLE_BUFFER_PCT: Final[Decimal] = Decimal("0.5")
#: New York time. Orders reaching Alpaca before 09:28 are filled at the official opening price.
SUBMIT_FROM: Final[time] = time(9, 20)
SUBMIT_UNTIL: Final[time] = time(9, 27)
#: New York time. The Allocator decides from published closes (the backtest's close + 15 min).
DECIDE_AFTER: Final[time] = time(16, 20)
MIN_ORDER_USD: Final[Decimal] = Decimal(1)  # Alpaca's fractional minimum (docs, checked 2026-10-02)
LADDER_SETTINGS: Final[frozenset[str]] = frozenset({"off"})  # "on" is designed in the system design, not built


def max_orders_per_day(n_symbols: int) -> int:
    """At most one sell and one buy per instrument per day."""
    return 2 * n_symbols


class PolicyError(ValueError):
    """The policy file is malformed or outside the tier-0 limits."""


@dataclass(frozen=True)
class CorePolicy:
    account: str
    registration: str
    evidence_run: str
    evidence_config: str
    effective_from: date
    funded_amount_cap_usd: Decimal
    mix: dict[str, Decimal]
    rule: core_alloc.Rule
    buffer_symbol: str | None
    buffer_target: Decimal
    max_order_usd: Decimal
    max_turnover_pct: Decimal
    min_order_usd: Decimal
    cash_reserve_usd: Decimal
    max_trades_per_week: int
    max_trade_value_pct_per_week: Decimal
    approval_needed_above_usd: Decimal
    drawdown_ladder: str
    receipts_to_pfw: bool
    sha256: str

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.mix))


def _dec(raw: Mapping[str, Any], key: str) -> Decimal:
    try:
        value = Decimal(str(raw[key]))
    except (KeyError, InvalidOperation) as exc:
        raise PolicyError(f"policy: {key!r} is missing or not a number") from exc
    if not value.is_finite():
        raise PolicyError(f"policy: {key!r} is not finite")
    return value


def load_policy(path: Path) -> CorePolicy:
    """Parse ``policy/core.toml``. Malformed input raises ``PolicyError``; bounds are ``violations``'s job."""
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PolicyError(f"policy: not valid TOML: {exc}") from exc
    limits, buffer = raw.get("limits", {}), raw.get("buffer", {})
    rule_name = str(raw.get("rule", {}).get("name", ""))
    if rule_name not in core_alloc.RULES:
        raise PolicyError(f"policy: unknown rule {rule_name!r}")
    try:
        effective = date.fromisoformat(str(raw["effective_from"]))
    except (KeyError, ValueError) as exc:
        raise PolicyError("policy: 'effective_from' must be an ISO date") from exc
    mix = {str(s): _dec(raw.get("mix", {}), s) for s in raw.get("mix", {})}
    if not mix:
        raise PolicyError("policy: [mix] is empty")
    return CorePolicy(
        account=str(raw.get("account", "")), registration=str(raw.get("registration", "")),
        evidence_run=str(raw.get("evidence_run", "")), evidence_config=str(raw.get("evidence_config", "")),
        effective_from=effective, funded_amount_cap_usd=_dec(raw, "funded_amount_cap_usd"), mix=mix,
        rule=core_alloc.RULES[rule_name], buffer_symbol=buffer.get("symbol"),
        buffer_target=_dec(buffer, "target") if "target" in buffer else Decimal(0),
        max_order_usd=_dec(limits, "max_order_usd"), max_turnover_pct=_dec(limits, "max_turnover_pct_per_rebalance"),
        min_order_usd=_dec(limits, "min_order_usd"), cash_reserve_usd=_dec(limits, "cash_reserve_usd"),
        max_trades_per_week=int(limits.get("max_trades_per_week", 0)),
        max_trade_value_pct_per_week=_dec(limits, "max_trade_value_pct_per_week"),
        approval_needed_above_usd=_dec(raw, "approval_needed_above_usd"),
        drawdown_ladder=str(raw.get("drawdown_ladder", "")), receipts_to_pfw=bool(raw.get("receipts_to_pfw", False)),
        sha256=hashlib.sha256(data).hexdigest())


def violations(policy: CorePolicy) -> list[str]:
    """Every way ``policy`` breaks a tier-0 limit; empty means the router may trade it."""
    out: list[str] = []
    if not policy.account or policy.account == "UNSET" or not policy.account.startswith("PA"):
        out.append("account is not set to a paper account number (PA…)")
    if any(w < 0 for w in policy.mix.values()):
        out.append("a mix weight is negative")
    if sum(policy.mix.values(), Decimal(0)) != 1:
        out.append(f"mix weights sum to {sum(policy.mix.values(), Decimal(0))}, not exactly 1")
    unknown = set(policy.mix) - ALLOWED_SYMBOLS
    if unknown:
        out.append(f"symbols outside the registered instruments: {sorted(unknown)}")
    crypto = sum((w for s, w in policy.mix.items() if s in CRYPTO_SYMBOLS), Decimal(0)) * 100
    if crypto > MAX_CRYPTO_WEIGHT_PCT:
        out.append(f"crypto weight {crypto}% exceeds {MAX_CRYPTO_WEIGHT_PCT}%")
    if policy.buffer_symbol is not None and policy.mix.get(policy.buffer_symbol) != policy.buffer_target:
        out.append(f"buffer {policy.buffer_symbol} is {policy.mix.get(policy.buffer_symbol)}, not its target "
                   f"{policy.buffer_target}")
    for name, value, ceiling in (
            ("max_order_usd", policy.max_order_usd, MAX_ORDER_NOTIONAL_USD),
            ("max_turnover_pct_per_rebalance", policy.max_turnover_pct, MAX_TURNOVER_PER_REBALANCE_PCT),
            ("funded_amount_cap_usd", policy.funded_amount_cap_usd, MAX_FUNDED_AMOUNT_USD),
            ("max_trade_value_pct_per_week", policy.max_trade_value_pct_per_week, MAX_TRADE_VALUE_PCT_PER_WEEK),
            ("approval_needed_above_usd", policy.approval_needed_above_usd, MAX_APPROVAL_THRESHOLD_USD)):
        if not 0 < value <= ceiling:
            out.append(f"{name} {value} is outside (0, {ceiling}]")
    if not 0 < policy.max_trades_per_week <= MAX_TRADES_PER_WEEK:
        out.append(f"max_trades_per_week {policy.max_trades_per_week} is outside (0, {MAX_TRADES_PER_WEEK}]")
    if policy.min_order_usd < MIN_ORDER_USD:
        out.append(f"min_order_usd {policy.min_order_usd} is below Alpaca's {MIN_ORDER_USD}")
    if policy.cash_reserve_usd < 0:
        out.append("cash_reserve_usd is negative")
    if policy.drawdown_ladder not in LADDER_SETTINGS:
        out.append(f"drawdown_ladder {policy.drawdown_ladder!r} is not built (allowed: {sorted(LADDER_SETTINGS)})")
    return out


# --- the per-plan limits -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Executed:
    """One accepted plan in the router's history, for the monthly and weekly caps."""
    day: date
    orders: int
    traded_usd: Decimal
    redecision: bool = False
    initial: bool = False   # the one-off build: not churn, so outside the weekly and monthly caps


def plan_violations(plan: core_alloc.Plan, policy: CorePolicy, *, prices: Mapping[str, Decimal], equity: Decimal,
                    history: Iterable[Executed], today: date, initial: bool = False,
                    funded: bool = False) -> list[str]:
    """Every way a plan breaks a limit. ``initial``: the first build, allowed past the turnover cap only
    when ``funded`` (the owner's signed fund command)."""
    out: list[str] = []
    history = [h for h in history if not h.initial]
    symbols = set(plan.sells) | set(plan.buys)
    outside = symbols - set(policy.mix)
    if outside:
        out.append(f"orders for symbols outside the policy: {sorted(outside)}")
    if symbols - ALLOWED_SYMBOLS:
        out.append(f"orders for symbols outside the registered instruments: {sorted(symbols - ALLOWED_SYMBOLS)}")
    ceiling = min(policy.max_order_usd, MAX_ORDER_NOTIONAL_USD)
    sells_usd = {s: q * prices[s] for s, q in plan.sells.items()}
    for s, usd in (*sells_usd.items(), *plan.buys.items()):
        if usd > ceiling:
            out.append(f"{s}: order ${usd:.2f} exceeds the ${ceiling} per-order ceiling")
    if equity <= 0:
        out.append("equity is not positive")
        return out
    one_way = max(sum(sells_usd.values(), Decimal(0)), sum(plan.buys.values(), Decimal(0)))
    turnover_cap = min(policy.max_turnover_pct, MAX_TURNOVER_PER_REBALANCE_PCT)
    if one_way / equity * 100 > turnover_cap and not (initial and funded):
        out.append(f"one-way turnover {one_way / equity * 100:.1f}% exceeds {turnover_cap}%"
                   + (" (the initial build needs the owner's signed fund command)" if initial else ""))
    orders = len(plan.sells) + len(plan.buys)
    if orders > max_orders_per_day(len(policy.mix)):
        out.append(f"{orders} orders exceed {max_orders_per_day(len(policy.mix))} a day")
    month = [h for h in history if (h.day.year, h.day.month) == (today.year, today.month) and not h.redecision]
    if not initial and len(month) >= MAX_REBALANCE_PLANS_PER_MONTH:
        out.append(f"{len(month)} plans already accepted this month (limit {MAX_REBALANCE_PLANS_PER_MONTH})")
    week = [h for h in history if today - timedelta(days=7) < h.day <= today]
    trades_cap = min(policy.max_trades_per_week, MAX_TRADES_PER_WEEK)
    if sum(h.orders for h in week) + orders > trades_cap:
        out.append(f"trades in seven days would reach {sum(h.orders for h in week) + orders} (limit {trades_cap})")
    value_cap = min(policy.max_trade_value_pct_per_week, MAX_TRADE_VALUE_PCT_PER_WEEK)
    week_value = sum((h.traded_usd for h in week), Decimal(0)) + sum(sells_usd.values(), Decimal(0)) \
        + sum(plan.buys.values(), Decimal(0))
    if not (initial and funded) and week_value / equity * 100 > value_cap:
        out.append(f"traded value in seven days would reach {week_value / equity * 100:.1f}% of equity "
                   f"(limit {value_cap}%)")
    crypto_target = sum((w for s, w in plan.targets.items() if s in CRYPTO_SYMBOLS), Decimal(0)) * 100
    if crypto_target > MAX_CRYPTO_WEIGHT_PCT:
        out.append(f"crypto target {crypto_target}% exceeds {MAX_CRYPTO_WEIGHT_PCT}%")
    return out


def needs_approval(plan: core_alloc.Plan, policy: CorePolicy, prices: Mapping[str, Decimal]) -> bool:
    traded = sum((q * prices[s] for s, q in plan.sells.items()), Decimal(0)) + sum(plan.buys.values(), Decimal(0))
    return traded > policy.approval_needed_above_usd
