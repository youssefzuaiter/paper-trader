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
import uuid
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Final

import httpx

from config import get_webhook_settings
from outbox import ReceiptKind, SendOutcome, default_outbox

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


async def send_once(url: str, receipt: dict[str, Any]) -> SendOutcome:
    """Exactly one signed POST of ``receipt`` to ``url``. Never raises.

    The single primitive both the inline retry loop (``_deliver``) and the
    durable outbox's replay loop are built on. Signs with a FRESH
    timestamp every call — PFW rejects a timestamp older than its
    300s replay window, so a replay hours later cannot reuse the headers
    of the original attempt. The body bytes are identical every time
    (``canonical_json`` is deterministic), so PFW's idempotency dedupe
    still sees the same receipt.

    ``permanent`` is True for a 4xx other than 429: a contract problem
    (bad signature, bad schema, a business rejection) that will not fix
    itself, so neither the inline loop nor the outbox retries it.
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

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(url, content=body, headers=headers)
    except httpx.HTTPError as exc:
        return SendOutcome(delivered=False, permanent=False, error=f"{type(exc).__name__}: {exc}")

    if response.is_success:
        return SendOutcome(delivered=True, permanent=False, status_code=response.status_code)

    error = f"HTTP {response.status_code}: {response.text[:200]}"
    permanent = 400 <= response.status_code < 500 and response.status_code != 429
    return SendOutcome(delivered=False, permanent=permanent, error=error, status_code=response.status_code)


async def _deliver(receipt: dict[str, Any], url: str, kind: ReceiptKind) -> dict[str, Any]:
    """Sign and POST a receipt (pending, settled, OR scenario-metrics) to
    ``url``, retrying briefly inline and then handing off to the durable
    outbox. Never raises.

    Shared by ``send_trade_receipt``/``send_settlement_receipt`` (both
    pass ``get_webhook_settings().url``) and ``send_scenario_metrics``
    (passes ``get_webhook_settings().metrics_url``, a genuinely different
    endpoint/table on the PFW side) — the signing, retry/backoff, outbox
    hand-off and structured result shape are identical across all three;
    only how the payload gets built, and where it's sent, differs.

    The event this receipt describes has already happened by the time
    this runs, so a delivery failure must not unwind it or surface as a
    500 on the caller's own endpoint/loop. Failures are logged and
    returned as structured status for the caller to record.

    Three quick inline attempts cover a blip (a PFW redeploy, a dropped
    connection). Anything longer than ~1.5s used to mean the receipt was
    simply lost — for an order that is ALREADY at the broker. Now a
    retryable exhaustion is queued in ``outbox`` and replayed with
    backoff until PFW acknowledges it; the result carries
    ``"queued": True`` so a caller can say "queued for replay" rather
    than "dropped". A permanent (4xx) rejection is still not retried —
    it will not fix itself — and is reported the same way it always was.
    """
    last_outcome = SendOutcome(delivered=False, permanent=False, error="not attempted")

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        last_outcome = await send_once(url, receipt)
        if last_outcome.delivered:
            logger.info(
                "Receipt %s (%s) delivered (HTTP %d)",
                receipt["idempotency_key"], receipt["status"], last_outcome.status_code,
            )
            return {
                "delivered": True,
                "status_code": last_outcome.status_code,
                "attempts": attempt,
                "idempotency_key": receipt["idempotency_key"],
            }
        if last_outcome.permanent:
            logger.error(
                "Receipt %s rejected, not retrying: %s",
                receipt["idempotency_key"], last_outcome.error,
            )
            return {
                "delivered": False,
                "queued": False,
                "error": last_outcome.error,
                "attempts": attempt,
                "idempotency_key": receipt["idempotency_key"],
                "receipt": receipt,
            }
        logger.warning(
            "Receipt POST attempt %d/%d failed: %s",
            attempt, _MAX_ATTEMPTS, last_outcome.error,
        )
        if attempt < _MAX_ATTEMPTS:
            await asyncio.sleep(_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))

    await default_outbox.enqueue(kind, url, receipt, last_outcome.error)
    logger.error(
        "Receipt %s (%s) undelivered after %d attempt(s), queued for replay: %s",
        receipt["idempotency_key"], receipt["status"], _MAX_ATTEMPTS, last_outcome.error,
    )
    return {
        "delivered": False,
        "queued": True,
        "error": last_outcome.error,
        "attempts": _MAX_ATTEMPTS,
        "idempotency_key": receipt["idempotency_key"],
        # Returned so a failed receipt can be inspected without
        # reconstructing (and re-signing) it; the outbox holds the copy
        # that will actually be replayed.
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
    return await _deliver(receipt, get_webhook_settings().url, "trade")


async def send_settlement_receipt(order: Any) -> dict[str, Any]:
    """Sign and POST a "settled" receipt for a really-filled order. See
    ``build_settlement_receipt``.
    """
    receipt = build_settlement_receipt(order)
    return await _deliver(receipt, get_webhook_settings().url, "settlement")


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
        # One key PER EVALUATION EVENT, minted here and carried unchanged
        # through the outbox, so a replay of THIS delivery dedupes on the
        # PFW side while a genuine re-evaluation of the same scenario a
        # few minutes later (this agent replays a fixed scenario set) is
        # recorded as its own row. This used to be `scenario_hash` — fine
        # while PFW ignored the key, and a real bug the moment PFW
        # started deduping on it: three MSFT evaluations became one row.
        # The scenario identity still travels as `hash` below.
        "idempotency_key": uuid.uuid4().hex,
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
    return await _deliver(payload, get_webhook_settings().metrics_url, "metrics")
