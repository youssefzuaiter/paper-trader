"""Wire types between the Inference Agents and the Risk & Routing agent."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class TradeSignal(BaseModel):
    """One Inference Agent's recommendation to open a long position.

    Carries the model's view only. Price, size, stop and exposure are the
    router's to decide: it fetches its own quote rather than trusting one
    in the signal, and the only market input it accepts is ``atr``, which
    can only ever make an order smaller (``tier0.plan_buy`` takes the
    minimum of the notional- and volatility-based quantities).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    #: Unique per signal. The order's client_order_id is derived from it,
    #: so a redelivered signal can never become a second order.
    signal_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    symbol: str = Field(min_length=1, max_length=10)
    source: str = Field(min_length=1, max_length=64, description="Emitting agent, e.g. inference-7f9c")
    model_name: str = Field(min_length=1, max_length=128)
    #: Calibrated probability that the price rises over the model's
    #: horizon — the scikit-learn pipeline's output, and the router's gate.
    prob_up: float = Field(ge=0.0, le=1.0)
    #: Expected move over the horizon, percent. Must be positive to open.
    predicted_move_pct: float
    confidence: float = Field(ge=0.0, le=1.0)
    headline: str | None = Field(default=None, max_length=500)
    #: From the Quantitative Agents. Optional; can only shrink the order.
    atr: Decimal | None = Field(default=None, gt=0)
    created_at: AwareDatetime

    @field_validator("symbol")
    @classmethod
    def _normalise_symbol(cls, value: str) -> str:
        value = value.strip().upper()
        if not value.replace(".", "").isalpha():
            raise ValueError("symbol must be letters (and '.') only")
        return value


class Decision(BaseModel):
    decision: Literal["accepted", "rejected"]
    code: str
    detail: str
    symbol: str
    intent: Literal["open", "reduce"]
    order: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    decided_at: datetime
