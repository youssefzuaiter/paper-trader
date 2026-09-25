"""Environment-backed configuration, split by trust boundary.

Two independent settings models, deliberately not one:

* :class:`BrokerSettings` — Alpaca credentials. Needed only to *place* an
  order.
* :class:`WebhookSettings` — the receipt-signing secret, the dashboard
  URL, and the simulated FX rate. Needed only to *deliver* a receipt.

They validate separately, and that separation is the whole point. A single
all-or-nothing ``Settings`` object meant a missing ``ALPACA_API_KEY_ID``
raised while merely building a receipt — blocking the one code path that
has no business touching the broker at all. Signing and delivering a
receipt for a fill that already happened must not depend on credentials
for placing a new one.

Tier-0 risk limits do NOT live here. They are hardcoded ``Final``
constants in ``execution.py`` — a limit that a ``.env`` file can widen is
not a limit.

Note on ``pydantic-settings``: not used, deliberately. It is the idiomatic
tool for env-backed settings, but it is a new dependency for what the
``from_env`` classmethods below already do in a few lines, and this
project's requirements list is deliberately tight. Swapping to
``BaseSettings`` later touches only this file.
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent

# ``override=False`` so a real exported environment variable (CI, systemd,
# Docker) always beats the on-disk .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

#: The only Alpaca host this project is permitted to reach.
PAPER_BASE_URL: Final[str] = "https://paper-api.alpaca.markets"

#: Named only so the validator below can reject it by name if it ever appears.
LIVE_BASE_URL: Final[str] = "https://api.alpaca.markets"

DEFAULT_WEBHOOK_URL: Final[str] = "http://localhost:3000/api/webhooks/trades"
DEFAULT_METRICS_WEBHOOK_URL: Final[str] = "http://localhost:3000/api/webhooks/metrics"
DEFAULT_USD_ILS_RATE: Final[str] = "3.700000"


class ConfigError(RuntimeError):
    """Raised when a settings model is missing or malformed.

    Wraps Pydantic's ``ValidationError`` so callers keep one stable error
    type to catch, and so the message names the offending *environment
    variable* rather than the model field it maps to.
    """


class BrokerSettings(BaseModel):
    """Alpaca credentials. Required only on the order-placing path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_id: str = Field(..., min_length=1)
    api_secret_key: str = Field(..., min_length=1)

    # Present so the broker's target is stated in configuration rather than
    # buried in a client constructor — but NOT read from the environment,
    # and pinned by the validator below. Phase 1 of this project made
    # sandbox execution a structural property, not a setting; routing this
    # through `.env` would hand it back to whoever can edit that file.
    base_url: str = PAPER_BASE_URL

    @field_validator("base_url")
    @classmethod
    def _must_be_sandbox(cls, value: str) -> str:
        if value != PAPER_BASE_URL:
            raise ValueError(
                f"base_url is pinned to the Alpaca paper sandbox ({PAPER_BASE_URL}); "
                f"refusing {value!r}"
                + (" — that is the LIVE trading host" if value == LIVE_BASE_URL else "")
            )
        return value

    @classmethod
    def from_env(cls) -> BrokerSettings:
        """Build from the environment. ``base_url`` is never sourced here."""
        return cls(
            api_key_id=os.getenv("ALPACA_API_KEY_ID", "").strip(),
            api_secret_key=os.getenv("ALPACA_API_SECRET_KEY", "").strip(),
        )

    def redacted(self) -> dict[str, str]:
        """Safe-to-log view. Never log the raw secret."""
        return {
            "api_key_id": f"{self.api_key_id[:4]}…" if self.api_key_id else "",
            "api_secret_key": "***",
            "base_url": self.base_url,
        }


class WebhookSettings(BaseModel):
    """Receipt signing and delivery. Required only on the webhook path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 32-character floor, matching the identical floor PFW's own
    # `src/server/env.ts` enforces on the same shared value — the two
    # services should not disagree about what counts as an acceptable key.
    secret: str = Field(..., min_length=32)
    url: str = Field(default=DEFAULT_WEBHOOK_URL, min_length=1)
    #: Structured AI-strategy telemetry (ad hoc, Phase 4) — a SEPARATE
    #: endpoint from `url` (the trade-receipt one), not derived from it
    #: by string manipulation: an explicit, independently-configurable
    #: setting is more robust than assuming `url` always ends in exactly
    #: "/api/webhooks/trades", and lets metrics delivery point elsewhere
    #: entirely if that's ever wanted without touching trade receipts.
    metrics_url: str = Field(default=DEFAULT_METRICS_WEBHOOK_URL, min_length=1)
    usd_ils_rate: Decimal = Field(default=Decimal(DEFAULT_USD_ILS_RATE), gt=0)

    @classmethod
    def from_env(cls) -> WebhookSettings:
        return cls(
            secret=os.getenv("WEBHOOK_SECRET", "").strip(),
            url=os.getenv("WEBHOOK_URL", "").strip() or DEFAULT_WEBHOOK_URL,
            metrics_url=os.getenv("METRICS_WEBHOOK_URL", "").strip() or DEFAULT_METRICS_WEBHOOK_URL,
            usd_ils_rate=Decimal(os.getenv("USD_ILS_RATE", "").strip() or DEFAULT_USD_ILS_RATE),
        )

    def redacted(self) -> dict[str, str]:
        return {
            "secret": "***",
            "url": self.url,
            "metrics_url": self.metrics_url,
            "usd_ils_rate": str(self.usd_ils_rate),
        }


#: Maps each model field back to the env var it came from, so a validation
#: failure names what the operator actually has to go and set.
_ENV_VAR_BY_FIELD: Final[dict[str, str]] = {
    "api_key_id": "ALPACA_API_KEY_ID",
    "api_secret_key": "ALPACA_API_SECRET_KEY",
    "secret": "WEBHOOK_SECRET",
    "url": "WEBHOOK_URL",
    "metrics_url": "METRICS_WEBHOOK_URL",
    "usd_ils_rate": "USD_ILS_RATE",
}


def _raise_config_error(label: str, error: ValidationError) -> None:
    problems = []
    for issue in error.errors():
        field = str(issue["loc"][0]) if issue["loc"] else "?"
        problems.append(f"{_ENV_VAR_BY_FIELD.get(field, field)}: {issue['msg']}")
    raise ConfigError(
        f"{label} is not configured — {'; '.join(problems)}. "
        "Copy .env.example to .env and fill it in."
    ) from error


@lru_cache(maxsize=1)
def get_broker_settings() -> BrokerSettings:
    """Resolve broker credentials. Raises only when an order is actually placed."""
    try:
        return BrokerSettings.from_env()
    except ValidationError as error:
        _raise_config_error("Alpaca broker", error)
        raise  # unreachable; satisfies the type checker


@lru_cache(maxsize=1)
def get_webhook_settings() -> WebhookSettings:
    """Resolve webhook settings. Independent of broker credentials by design."""
    try:
        return WebhookSettings.from_env()
    except (ValidationError, ArithmeticError) as error:
        if isinstance(error, ValidationError):
            _raise_config_error("Webhook delivery", error)
        raise ConfigError(f"USD_ILS_RATE is not a valid decimal: {error}") from error


def broker_is_configured() -> bool:
    """True when Alpaca credentials validate — for boot-time reporting only.

    Never a gate on the webhook path: that path does not need the broker,
    which is the entire reason these two models are separate.
    """
    try:
        get_broker_settings()
    except ConfigError:
        return False
    return True


def is_autonomous_mode_enabled() -> bool:
    """True when ``AUTONOMOUS_MODE`` is set to a truthy value.

    Deliberately plain ``os.getenv`` parsing, not a Pydantic settings
    model — this is a feature toggle, not a trust-boundary credential the
    way :class:`BrokerSettings`/:class:`WebhookSettings` are, so it needs
    no validate-or-raise treatment. Defaults to ``False`` when unset, so a
    fresh checkout with no ``.env`` change never starts submitting
    autonomous paper trades on its own.
    """
    return os.getenv("AUTONOMOUS_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def orders_via_risk_router() -> bool:
    """True when ``ORDERS_VIA_RISK_ROUTER`` is truthy: order placement has
    moved to the Risk & Routing agent (``risk_router/``).

    This process then places no orders at all — ``/signals/execute``
    answers 410, and the autonomous loop does not start even with
    ``AUTONOMOUS_MODE`` on — while it keeps doing everything else:
    settlement stream, outbox, reconciliation, quotes, control, anomaly
    scoring. Those settle fills account-wide, the router's included.

    Off by default so an existing deployment keeps trading until the
    router is actually running somewhere. While it is off, this process
    can still buy past the router's portfolio limits.
    """
    return os.getenv("ORDERS_VIA_RISK_ROUTER", "").strip().lower() in {"1", "true", "yes", "on"}
