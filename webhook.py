"""Phase 4 — HMAC-SHA256 signed trade receipts for the Next.js dashboard.

Wire contract
-------------
``POST {WEBHOOK_URL}``

===========================  ==============================================
Header                       Value
===========================  ==============================================
``Content-Type``             ``application/json``
``X-Signature-Timestamp``    Unix seconds, as a string
``X-Signature-256``          ``sha256=<hex digest>``
``X-Idempotency-Key``        Mirrors ``idempotency_key`` in the body
===========================  ==============================================

The signed material is ``f"{timestamp}.".encode() + body`` — Stripe's scheme.
Binding the timestamp into the MAC is what makes a captured receipt
un-replayable: the receiver rejects a stale timestamp, and an attacker cannot
move the timestamp forward without invalidating the digest.

Two invariants this module exists to protect:

1. **Sign the bytes you send.** The canonical body is serialised once and
   passed to httpx as ``content=``, never ``json=``. Handing httpx a dict
   would let it re-serialise with different key order or spacing, and the
   digest would no longer describe the transmitted bytes.
2. **Money crosses the wire as integers.** Amounts are minor units
   (cents / agorot) as JSON integers, and share quantities are decimal
   *strings*. Nothing monetary is ever a JSON float — PFW stores these as
   ``BigInt`` and ``Decimal(30, 18)``, and a float round-trip would corrupt
   both.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Final

import httpx

from config import get_webhook_settings

if TYPE_CHECKING:  # avoid a circular import at runtime
    from execution import ExecutionPlan
    from inference import InferenceSignal

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 1
SIGNATURE_HEADER: Final[str] = "X-Signature-256"
TIMESTAMP_HEADER: Final[str] = "X-Signature-Timestamp"
IDEMPOTENCY_HEADER: Final[str] = "X-Idempotency-Key"
SIGNATURE_PREFIX: Final[str] = "sha256="

_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(5.0, connect=2.0)
_MAX_ATTEMPTS: Final[int] = 3
_BACKOFF_BASE_SECONDS: Final[float] = 0.5


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialise deterministically: sorted keys, no whitespace, UTF-8.

    Determinism matters because the digest is computed over these exact bytes.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sign(body: bytes, timestamp: str, secret: str) -> str:
    """Return the hex HMAC-SHA256 over ``"{timestamp}.".encode() + body``."""
    mac = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    )
    return mac.hexdigest()


def verify(body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    """Constant-time verification of an inbound HMAC-signed request.

    Mirrors what PFW's own webhook routes do for THIS process's outbound
    receipts — used in the reverse direction here, by
    ``main.py``'s ``POST /control/halt``, to verify a request arriving
    INTO this service (from PFW's server, which signs it with the same
    shared ``WEBHOOK_SECRET`` before forwarding). Previously removed as
    unused dead code in an earlier audit pass, since nothing called it at
    the time — restored because the emergency-halt endpoint is a genuine
    caller now, not speculative.
    """
    expected = sign(body, timestamp, secret)
    provided = signature.removeprefix(SIGNATURE_PREFIX)
    return hmac.compare_digest(expected, provided)


def _to_minor_units(amount: Decimal) -> int:
    """Exact major-unit -> minor-unit conversion (USD -> cents, ILS -> agorot)."""
    return int((amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def build_receipt(
    *,
    plan: ExecutionPlan,
    signal: InferenceSignal,
    order: Any,
    executed_at: datetime,
) -> dict[str, Any]:
    """Assemble a "pending" receipt body, fired the instant Alpaca ACCEPTS
    an order — before anything has actually happened at the real market.

    Field names and units mirror PFW's ``Trade`` model so the Next.js route can
    persist without a translation layer: ``*_agorot`` and ``native_*_amount``
    are ``BigInt``-safe integers, ``quantity`` and ``exchange_rate_at_entry``
    are decimal strings.

    Two-phase settlement (ad hoc, extends the original single-shot
    design): PFW creates only a PENDING ``Trade`` row from this receipt —
    no envelope/ledger effect yet, since ``plan.limit_price`` is not a
    confirmed fill. See ``build_settlement_receipt`` for the "settled"
    counterpart, sent once ``scheduler.settlement_loop`` observes a real
    fill, carrying the ACTUAL fill price instead.
    """
    rate = get_webhook_settings().usd_ils_rate

    price_usd = plan.limit_price
    total_usd = plan.notional_usd
    price_ils = price_usd * rate
    total_ils = total_usd * rate

    return {
        "schema_version": SCHEMA_VERSION,
        # Mirrors Trade.idempotencyKey — makes webhook *redelivery* safe.
        "idempotency_key": plan.client_order_id,
        "status": "pending",
        "broker": "alpaca",
        "environment": "paper",
        "order_id": str(order.id),
        "client_order_id": order.client_order_id,
        "order_status": str(order.status),
        "symbol": plan.symbol,
        "side": plan.side.value,
        "quantity": str(plan.quantity),
        "currency": "USD",
        "native_price_amount": _to_minor_units(price_usd),
        "native_total_amount": _to_minor_units(total_usd),
        "exchange_rate_at_entry": str(rate),
        "price_agorot": _to_minor_units(price_ils),
        "total_agorot": _to_minor_units(total_ils),
        "limit_price": str(price_usd),
        "stop_price": str(plan.stop_price),
        "stop_loss_kind": plan.stop_loss_kind.value,
        "executed_at": executed_at.isoformat().replace("+00:00", "Z"),
        "signal": {
            "model_name": signal.model_name,
            "predicted_move_pct": signal.predicted_move_pct,
            "confidence": signal.confidence,
            "headline": signal.headline,
        },
    }


def build_settlement_receipt(order: Any) -> dict[str, Any]:
    """Assemble a "settled" receipt from a REALLY-FILLED Alpaca order.

    Built independently of ``build_receipt``'s plan+signal shape: by the
    time ``scheduler.settlement_loop`` polls a fill, the original
    ``ExecutionPlan``/``InferenceSignal`` objects from that earlier HTTP
    request are long gone — only the order's own fields (and the current
    exchange rate) are available, which is also exactly why this receipt
    correctly has no ``signal`` block (that field is optional in PFW's
    schema for precisely this reason).

    Uses ``order.filled_avg_price``/``order.filled_qty`` — the REAL fill —
    never ``order.limit_price``, which is what caused every paper trade
    before this fix to be booked in PFW's ledger despite never actually
    filling at Alpaca.
    """
    rate = get_webhook_settings().usd_ils_rate

    filled_price_usd = Decimal(str(order.filled_avg_price))
    filled_qty = Decimal(str(order.filled_qty))
    total_usd = filled_price_usd * filled_qty
    price_ils = filled_price_usd * rate
    total_ils = total_usd * rate

    side = str(order.side).rsplit(".", maxsplit=1)[-1].lower()
    filled_at = order.filled_at or order.updated_at

    return {
        "schema_version": SCHEMA_VERSION,
        "idempotency_key": order.client_order_id,
        "status": "settled",
        "broker": "alpaca",
        "environment": "paper",
        "order_id": str(order.id),
        "client_order_id": order.client_order_id,
        "order_status": str(order.status),
        "symbol": order.symbol,
        "side": side,
        "quantity": str(filled_qty),
        "currency": "USD",
        "native_price_amount": _to_minor_units(filled_price_usd),
        "native_total_amount": _to_minor_units(total_usd),
        "exchange_rate_at_entry": str(rate),
        "price_agorot": _to_minor_units(price_ils),
        "total_agorot": _to_minor_units(total_ils),
        "executed_at": filled_at.isoformat().replace("+00:00", "Z"),
    }


async def _sign_and_send(receipt: dict[str, Any], url: str) -> dict[str, Any]:
    """Sign and POST a receipt (pending, settled, OR scenario-metrics) to
    ``url``. Never raises.

    Shared by ``send_trade_receipt``/``send_settlement_receipt`` (both
    pass ``get_webhook_settings().url``) and ``send_scenario_metrics``
    (passes ``get_webhook_settings().metrics_url``, a genuinely different
    endpoint/table on the PFW side) — the HMAC signing, retry/backoff,
    and structured result shape are identical across all three; only how
    the payload gets built, and where it's sent, differs.

    The event this receipt describes has already happened by the time
    this runs, so a delivery failure must not unwind it or surface as a
    500 on the caller's own endpoint/loop. Failures are logged and
    returned as structured status for the caller to record.

    Retries are safe: the body, timestamp and signature are computed once
    and replayed byte-for-byte, and ``idempotency_key`` lets the receiver
    dedupe.
    """
    settings = get_webhook_settings()
    body = canonical_json(receipt)
    timestamp = str(int(time.time()))
    signature = sign(body, timestamp, settings.secret)

    headers = {
        "Content-Type": "application/json",
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: f"{SIGNATURE_PREFIX}{signature}",
        IDEMPOTENCY_HEADER: receipt["idempotency_key"],
        "User-Agent": "tier0-paper-trader/1.0",
    }

    last_error = "not attempted"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await client.post(
                    url, content=body, headers=headers
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Receipt POST attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, last_error,
                )
            else:
                if response.is_success:
                    logger.info(
                        "Receipt %s (%s) delivered (HTTP %d)",
                        receipt["idempotency_key"], receipt["status"], response.status_code,
                    )
                    return {
                        "delivered": True,
                        "status_code": response.status_code,
                        "attempts": attempt,
                        "idempotency_key": receipt["idempotency_key"],
                    }

                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                # 4xx is a contract problem (bad signature, bad schema) and will
                # not fix itself. Only 5xx and transport errors are retried.
                if response.status_code < 500:
                    logger.error(
                        "Receipt %s rejected, not retrying: %s",
                        receipt["idempotency_key"], last_error,
                    )
                    break
                logger.warning(
                    "Receipt POST attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, last_error,
                )

            if attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    logger.error(
        "Receipt %s (%s) undelivered after %d attempt(s): %s",
        receipt["idempotency_key"], receipt["status"], _MAX_ATTEMPTS, last_error,
    )
    return {
        "delivered": False,
        "error": last_error,
        "attempts": _MAX_ATTEMPTS,
        "idempotency_key": receipt["idempotency_key"],
        # Returned so a failed receipt can be replayed by hand without
        # reconstructing (and re-signing) it.
        "receipt": receipt,
    }


async def send_trade_receipt(
    *,
    plan: ExecutionPlan,
    signal: InferenceSignal,
    order: Any,
    executed_at: datetime,
) -> dict[str, Any]:
    """Sign and POST a "pending" receipt. See ``build_receipt``."""
    receipt = build_receipt(plan=plan, signal=signal, order=order, executed_at=executed_at)
    return await _sign_and_send(receipt, get_webhook_settings().url)


async def send_settlement_receipt(order: Any) -> dict[str, Any]:
    """Sign and POST a "settled" receipt for a really-filled order. See
    ``build_settlement_receipt``.
    """
    receipt = build_settlement_receipt(order)
    return await _sign_and_send(receipt, get_webhook_settings().url)


def build_scenario_metrics(
    *,
    ticker: str,
    headline: str,
    predicted_move_pct: float,
    decision: str,
    actual_fill_price: Decimal | None = None,
    shadow_predicted_move_pct: float | None = None,
    shadow_decision: str | None = None,
) -> dict[str, Any]:
    """Assemble one scenario-evaluation metrics payload.

    ``hash`` is a SHA-256 hex digest of ``"{ticker}|{headline}"`` — the
    same input shape ``inference._embed()`` already hashes for its
    pseudo-embedding — so PFW's ``ScenarioMetrics.hash`` column can group
    repeated evaluations of the SAME scenario (this agent replays a fixed
    ``inference.MARKET_SCENARIOS`` set, not unique headlines every time)
    without this side needing to coordinate a shared id scheme. The SAME
    hash covers both the primary and shadow verdicts on this row — both
    models evaluated the identical (ticker, headline) scenario.

    ``decision`` is either ``"executed"`` or one of
    ``execution.RejectionCode``'s values — a plain string PFW stores
    as-is, not validated against a fixed enum on either side (see
    ``ScenarioMetrics``'s own schema doc comment for why). ``shadow_decision``
    is either ``"would_execute"`` or ``"would_reject"`` (see
    ``inference._shadow_evaluate``) — a deliberately different vocabulary
    from the primary ``decision``, since the shadow model never actually
    executes anything; conflating the two into one shared set of literal
    values would misrepresent a hypothetical verdict as a real one.

    ``actual_fill_price`` is ``None`` when called from
    ``scheduler.trading_loop`` (this feature's only caller, per its own
    scope) — evaluation happens before any order is even submitted, let
    alone filled, so there is genuinely no fill price to report yet.

    ``shadow_predicted_move_pct``/``shadow_decision`` are both optional
    (``None`` when unset) purely for backward compatibility with any
    caller that hasn't computed a shadow verdict — ``scheduler.py``'s own
    real caller (Phase 3, ad hoc) always supplies both together.
    """
    scenario_hash = hashlib.sha256(f"{ticker}|{headline}".encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        # Informational only here — unlike the trade-receipt endpoints,
        # PFW's metrics route enforces no idempotency-key dedup (a
        # duplicate analytics row on retry is an acceptable, low-severity
        # cost; this is telemetry, not money).
        "idempotency_key": scenario_hash,
        "status": "metrics",
        "ticker": ticker,
        "hash": scenario_hash,
        "predicted_move_pct": predicted_move_pct,
        "actual_fill_price": str(actual_fill_price) if actual_fill_price is not None else None,
        "decision": decision,
        "shadow_predicted_move_pct": shadow_predicted_move_pct,
        "shadow_decision": shadow_decision,
    }


async def send_scenario_metrics(
    *,
    ticker: str,
    headline: str,
    predicted_move_pct: float,
    decision: str,
    actual_fill_price: Decimal | None = None,
    shadow_predicted_move_pct: float | None = None,
    shadow_decision: str | None = None,
) -> dict[str, Any]:
    """Sign and POST one scenario-evaluation metrics event. Never raises
    — a metrics-delivery failure must not interrupt the trading loop that
    calls this. See ``build_scenario_metrics``.
    """
    payload = build_scenario_metrics(
        ticker=ticker,
        headline=headline,
        predicted_move_pct=predicted_move_pct,
        decision=decision,
        actual_fill_price=actual_fill_price,
        shadow_predicted_move_pct=shadow_predicted_move_pct,
        shadow_decision=shadow_decision,
    )
    return await _sign_and_send(payload, get_webhook_settings().metrics_url)
