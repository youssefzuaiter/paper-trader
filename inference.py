"""Phase 2 — simulated ingestion + FinBERT-shaped sentiment inference.

The ingestion side is simulated on purpose: this agent must be runnable with a
free Alpaca paper key and no market-data subscription.

The inference side is a *placeholder with real plumbing*. It is not a trained
model, but it has the exact shape of one:

* a ``torch.nn.Module`` producing 3-class logits in ProsusAI/finbert's label
  order — ``0=positive, 1=negative, 2=neutral``;
* softmax -> ``(p_pos, p_neg, p_neu)``;
* the blocking forward pass dispatched via ``asyncio.to_thread`` so a CPU-bound
  transformer never stalls the FastAPI event loop.

The real model is one environment variable away: ``SENTIMENT_MODEL=finbert``
loads ProsusAI/finbert as an int8 ONNX graph (see ``_load_finbert``) —
nothing downstream of ``predict_move`` changes, and the placeholder stays
the default so the test suite and a credential-free checkout keep working
with no download and no extra memory.

Every output is **deterministic** in ``(ticker, headline)``. A Tier-0 engine
whose upstream signal is random is untestable: the same headline must always
produce the same order.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, Field

import broker
from config import ConfigError

logger = logging.getLogger(__name__)

PLACEHOLDER_MODEL_NAME: Final[str] = "placeholder-finbert-v0"

#: ``SENTIMENT_MODEL=finbert`` selects the real checkpoint; anything else
#: (including unset) is the placeholder. Read once at import, like every
#: other setting in config.py — a model is not something to hot-swap.
FINBERT_ENABLED: Final[bool] = os.getenv("SENTIMENT_MODEL", "").strip().lower() == "finbert"

#: Xenova/finbert is ProsusAI/finbert exported to ONNX (its config.json
#: names ProsusAI/finbert as ``_name_or_path`` and keeps the identical
#: label order, verified against both repos' config.json on 2026-09-18).
#: The int8 graph is 110 MB on disk and never materialises fp32 weights —
#: the fp32 checkpoint alone is 438 MB, which a 512 MB Render instance
#: cannot even LOAD, let alone run beside torch. Overridable for a private
#: mirror; the label order must stay 0=positive, 1=negative, 2=neutral.
FINBERT_REPO: Final[str] = os.getenv("SENTIMENT_MODEL_REPO", "Xenova/finbert").strip() or "Xenova/finbert"
FINBERT_FILE: Final[str] = os.getenv("SENTIMENT_MODEL_FILE", "onnx/model_int8.onnx").strip() or "onnx/model_int8.onnx"
#: BERT's positional limit is 512; a headline is a sentence. 128 bounds a
#: pathological input's inference cost without ever truncating a real one.
_FINBERT_MAX_TOKENS: Final[int] = 128

MODEL_NAME: Final[str] = f"finbert-onnx-int8 ({FINBERT_REPO})" if FINBERT_ENABLED else PLACEHOLDER_MODEL_NAME
_EMBEDDING_DIM: Final[int] = 64
_NUM_LABELS: Final[int] = 3  # positive, negative, neutral
_WEIGHT_SEED: Final[int] = 20260905

#: Ceiling on the magnitude the placeholder will ever predict, in percent.
#: A real FinBERT head would be calibrated against realised returns instead.
_MAX_ABS_MOVE_PCT: Final[float] = 15.0

#: Placeholder lexical prior. Delete this entirely when the real checkpoint
#: lands — it exists so the stub behaves intelligibly during development
#: rather than emitting hash noise.
_BULLISH: Final[frozenset[str]] = frozenset(
    {"beats", "surges", "record", "upgrade", "breakthrough", "soars",
     "approval", "wins", "raises", "acquisition", "blowout"}
)
_BEARISH: Final[frozenset[str]] = frozenset(
    {"misses", "plunges", "recall", "downgrade", "probe", "lawsuit",
     "slashes", "halts", "resigns", "fraud", "bankruptcy"}
)


class InferenceSignal(BaseModel):
    """Model output consumed by the Tier-0 execution engine."""

    ticker: str = Field(..., min_length=1, max_length=8)
    headline: str = Field(..., min_length=1)
    predicted_move_pct: float = Field(
        ...,
        description="Signed predicted move over the horizon, in percent. "
                    "12.4 means +12.4%.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    prob_positive: float = Field(..., ge=0.0, le=1.0)
    prob_negative: float = Field(..., ge=0.0, le=1.0)
    prob_neutral: float = Field(..., ge=0.0, le=1.0)
    model_name: str = MODEL_NAME
    inferred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Quote:
    """A two-sided quote (real when available, else simulated — see
    ``fetch_quote``). Prices are exact — never floats.

    ``atr`` (ad hoc, Phase 4 — Volatility Parity Sizing) is the real
    14-day Average True Range from ``broker.get_atr``, or ``None`` on the
    synthetic-quote fallback path (no real market data to compute one
    from) — ``execution.py``'s volatility-parity sizing degrades to its
    notional-only figure in that case, the same graceful-degradation
    posture every other real-data dependency in this app already has.
    """

    ticker: str
    bid: Decimal
    ask: Decimal
    as_of: datetime
    atr: Decimal | None = None

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)


class _PlaceholderFinBERT(torch.nn.Module):
    """Fixed-weight linear head over a hashed pseudo-embedding.

    Stands in for ``AutoModelForSequenceClassification`` with the same
    ``(batch, num_labels)`` logit contract.
    """

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(_WEIGHT_SEED)
        self.projection = torch.nn.Linear(_EMBEDDING_DIM, _NUM_LABELS)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.empty(_NUM_LABELS, _EMBEDDING_DIM).uniform_(
                    -0.5, 0.5, generator=generator
                )
            )
            self.projection.bias.zero_()
        self.eval()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(embedding)


@lru_cache(maxsize=1)
def _load_model() -> _PlaceholderFinBERT:
    """Load the placeholder network once per process.

    Still the primary model unless ``SENTIMENT_MODEL=finbert``, and ALWAYS
    the shadow model's network (``_shadow_score_sync``): the shadow is a
    mock A/B comparator whose whole point is a cheap, deterministic second
    opinion, not a second 110 MB graph.
    """
    logger.info("Loading sentiment model %s", PLACEHOLDER_MODEL_NAME)
    return _PlaceholderFinBERT()


@dataclass(frozen=True, slots=True)
class _FinBERT:
    """The real checkpoint: a fast tokenizer plus an ONNX Runtime session.

    ``input_names`` is read off the graph rather than assumed — optimum's
    BERT exports take ``input_ids``/``attention_mask``/``token_type_ids``,
    but an alternative export may drop the third, and feeding an input the
    graph does not declare is an error, not a no-op.
    """

    tokenizer: object
    session: object
    input_names: frozenset[str]


@lru_cache(maxsize=1)
def _load_finbert() -> _FinBERT:
    """Download (once, into the Hugging Face cache) and open the int8 graph.

    Imported lazily so a placeholder deployment never pays for onnxruntime,
    tokenizers or huggingface_hub at import time. ``main.py``'s startup
    warmup is what triggers this, so a missing download fails the boot —
    loudly, before the first real cycle — instead of the first order.
    Memory: measured on Linux, the session adds ~+138 MB RSS on top of
    whatever this process already carries (numbers in the README).
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    logger.info("Loading sentiment model %s (%s)", MODEL_NAME, FINBERT_FILE)
    model_path = hf_hub_download(FINBERT_REPO, FINBERT_FILE)
    tokenizer_path = hf_hub_download(FINBERT_REPO, "tokenizer.json")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(_FINBERT_MAX_TOKENS)

    options = ort.SessionOptions()
    # One headline at a time on a fractional-CPU instance: thread pools
    # only add contention and per-thread arenas here.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # ORT "prepacks" every MatMul weight into a second, kernel-optimised
    # copy at session creation — a speed trade that costs memory this
    # deployment cannot spare: measured on Linux, the 110 MB graph
    # resident at +192 MB with prepacking and +138 MB without, for 11 ms
    # vs 17 ms per headline. At one headline per cycle the 54 MB is the
    # margin that keeps a 512 MB instance out of the OOM killer.
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    return _FinBERT(
        tokenizer=tokenizer,
        session=session,
        input_names=frozenset(i.name for i in session.get_inputs()),
    )


def _finbert_logits(headline: str) -> np.ndarray:
    """Tokenise one headline and run the graph; returns the 3 raw logits."""
    fb = _load_finbert()
    encoding = fb.tokenizer.encode(headline)  # type: ignore[attr-defined]
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.asarray([encoding.ids], dtype=np.int64),
        "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.asarray([encoding.type_ids], dtype=np.int64),
    }
    feeds = {name: value for name, value in feeds.items() if name in fb.input_names}
    (logits,) = fb.session.run(None, feeds)  # type: ignore[attr-defined]
    return np.asarray(logits, dtype=np.float32)[0]


def _embed(ticker: str, headline: str) -> torch.Tensor:
    """Deterministic pseudo-embedding: SHA-256 -> unit-norm float vector.

    Stands in for tokenisation + encoder. Identical input always yields an
    identical tensor, on any machine, in any process.
    """
    digest = hashlib.sha256(f"{ticker.upper()}|{headline}".encode()).digest()
    # 64 bytes needed, SHA-256 gives 32 — extend with a second round.
    digest += hashlib.sha256(digest).digest()
    raw = torch.tensor([b / 255.0 for b in digest], dtype=torch.float32)
    centred = raw - raw.mean()
    return centred / (centred.norm() + 1e-8)


def _lexical_prior(headline: str) -> torch.Tensor:
    """Placeholder-only nudge so obvious headlines score sensibly."""
    tokens = {t.strip(".,!?:;'\"").lower() for t in headline.split()}
    bull = len(tokens & _BULLISH)
    bear = len(tokens & _BEARISH)
    return torch.tensor([bull * 1.2, bear * 1.2, 0.0], dtype=torch.float32)


def _score_sync(ticker: str, headline: str) -> tuple[float, float, float]:
    """Blocking forward pass. Returns ``(p_positive, p_negative, p_neutral)``.

    The real model sees the headline alone — the ticker is already in the
    text of every headline this agent ingests, and FinBERT was trained on
    sentences, not ``TICKER|sentence`` pairs. The lexical prior is a
    placeholder-only nudge and is deliberately NOT applied on top of a
    trained classifier's own probabilities.
    """
    if FINBERT_ENABLED:
        logits = torch.from_numpy(_finbert_logits(headline))
    else:
        model = _load_model()
        with torch.no_grad():
            logits = model(_embed(ticker, headline)) + _lexical_prior(headline)
    probs = torch.softmax(logits, dim=-1)
    return tuple(round(float(p), 6) for p in probs)  # type: ignore[return-value]


async def predict_move(ticker: str, headline: str) -> InferenceSignal:
    """Score a headline for a ticker.

    The forward pass runs in a worker thread: a real FinBERT pass is tens to
    hundreds of milliseconds of CPU-bound work, and running it inline would
    block every other request on the event loop.

    Args:
        ticker: Equity symbol, e.g. ``"AAPL"``.
        headline: Raw news headline text.

    Returns:
        An :class:`InferenceSignal`. Deterministic in ``(ticker, headline)``.

    Raises:
        ValueError: If ``ticker`` or ``headline`` is blank.
    """
    ticker = ticker.strip().upper()
    headline = headline.strip()
    if not ticker or not headline:
        raise ValueError("ticker and headline must both be non-empty")

    p_pos, p_neg, p_neu = await asyncio.to_thread(_score_sync, ticker, headline)

    # Signed expected move: net sentiment scaled to the magnitude ceiling.
    predicted_move_pct = round((p_pos - p_neg) * _MAX_ABS_MOVE_PCT, 4)

    return InferenceSignal(
        ticker=ticker,
        headline=headline,
        predicted_move_pct=predicted_move_pct,
        confidence=max(p_pos, p_neg, p_neu),
        prob_positive=p_pos,
        prob_negative=p_neg,
        prob_neutral=p_neu,
    )


@dataclass(frozen=True, slots=True)
class ShadowSignal:
    """A shadow (non-live) model's theoretical verdict on a scenario.

    Persisted alongside the primary model's own signal (via
    ``webhook.send_scenario_metrics``) for offline A/B backtesting only —
    this type never reaches ``execution.py``, and nothing in this file or
    ``scheduler.py`` ever gates a real order on it.
    """

    predicted_move_pct: float
    decision: str


#: The shadow model's own lexical weight — deliberately different from
#: ``_lexical_prior``'s ``1.2`` baseline, so its predicted_move_pct
#: genuinely diverges from the primary model instead of just adding
#: noise on top of an identical score.
_SHADOW_LEXICAL_WEIGHT: Final[float] = 2.5

#: The shadow model's own "would this have executed" threshold —
#: deliberately different from ``execution.MIN_PREDICTED_GAIN_PCT``'s
#: 10%, a more permissive variant so the two models' verdicts can
#: genuinely diverge on the same scenario, which is the entire point of
#: an A/B comparison (two models that always agree have nothing to
#: backtest). This constant is intentionally NOT imported from
#: execution.py or compared against it anywhere live — the shadow
#: model's threshold is its own, separate, non-authoritative opinion.
_SHADOW_MIN_GAIN_PCT: Final[float] = 7.0


def _shadow_lexical_prior(headline: str) -> torch.Tensor:
    """Same keyword sets as ``_lexical_prior``, weighted differently —
    the "different lexical scoring threshold" variant this shadow model
    is built around.
    """
    tokens = {t.strip(".,!?:;'\"").lower() for t in headline.split()}
    bull = len(tokens & _BULLISH)
    bear = len(tokens & _BEARISH)
    return torch.tensor([bull * _SHADOW_LEXICAL_WEIGHT, bear * _SHADOW_LEXICAL_WEIGHT, 0.0], dtype=torch.float32)


def _shadow_score_sync(ticker: str, headline: str) -> tuple[float, float, float]:
    """Blocking forward pass for the shadow model. Reuses the SAME
    placeholder network weights as the primary model (``_load_model()``)
    — the divergence is deliberately scoped to the lexical weighting
    alone, matching the task's own "different lexical scoring threshold"
    framing, not a wholly separate model architecture, which would be a
    bigger and unrequested scope than a mock A/B comparator needs.
    """
    model = _load_model()
    with torch.no_grad():
        logits = model(_embed(ticker, headline)) + _shadow_lexical_prior(headline)
        probs = torch.softmax(logits, dim=-1)
    return tuple(round(float(p), 6) for p in probs)  # type: ignore[return-value]


async def _shadow_evaluate(ticker: str, headline: str) -> ShadowSignal:
    """Mock secondary model, for offline A/B backtesting ONLY.

    Never reaches ``execution.py``, never submits an order, never gates
    anything live — ``scheduler._run_one_cycle`` runs this concurrently
    with the PRIMARY pipeline purely to persist a second opinion into
    ``ScenarioMetrics`` for later comparison; only ``predict_move``'s own
    result ever decides what actually happens at the broker.

    Deterministic in ``(ticker, headline)``, matching the primary model's
    own stated design goal ("the same headline must always produce the
    same order," this module's top docstring) — a shadow verdict that
    changed between runs would make offline A/B comparison meaningless,
    since re-running the same historical scenario should reproduce the
    same shadow verdict every time. This is why the task's own suggested
    "randomized sentiment heuristic" alternative was NOT used: genuine
    randomness here would be a real, avoidable regression against a
    value this whole codebase holds consistently elsewhere (seeded RNGs,
    hash-derived pseudo-embeddings, the primary model's own determinism).
    """
    ticker = ticker.strip().upper()
    headline = headline.strip()
    if not ticker or not headline:
        raise ValueError("ticker and headline must both be non-empty")

    p_pos, p_neg, p_neu = await asyncio.to_thread(_shadow_score_sync, ticker, headline)
    predicted_move_pct = round((p_pos - p_neg) * _MAX_ABS_MOVE_PCT, 4)
    decision = "would_execute" if predicted_move_pct >= _SHADOW_MIN_GAIN_PCT else "would_reject"

    return ShadowSignal(predicted_move_pct=predicted_move_pct, decision=decision)


# --------------------------------------------------------------------------
# Simulated ingestion
# --------------------------------------------------------------------------

_SIMULATED_BASE_PRICES: Final[dict[str, str]] = {
    "AAPL": "228.50", "MSFT": "441.20", "NVDA": "121.75", "TSLA": "248.90",
    "AMZN": "186.30", "GOOGL": "165.40", "META": "563.10", "AMD": "142.85",
}
_DEFAULT_BASE_PRICE: Final[str] = "100.00"
_SPREAD_BPS: Final[Decimal] = Decimal("5")  # 5 basis points half-spread

_HEADLINE_TEMPLATES: Final[tuple[str, ...]] = (
    "{t} beats quarterly estimates and raises full-year guidance",
    "{t} shares plunge after regulator opens a probe",
    "{t} announces record datacenter demand and a major acquisition",
    "{t} misses on revenue as management slashes outlook",
    "{t} holds annual shareholder meeting; no guidance change",
)

#: Diverse (ticker, headline) scenarios for stress-testing the Tier-0
#: gates end to end, instead of repeatedly hitting the same hardcoded
#: NVDA headline. Each entry's predicted_move_pct under BOTH models
#: (measured, not keyword-counted — 2026-09-18, int8 ONNX graph):
#:            placeholder      real FinBERT
#:   TSLA     ~+14.8% clears   +11.9% clears   (pos 0.84)
#:   AAPL      ~+6.1% rejected  +7.0% rejected (pos 0.54, neu 0.40)
#:   MSFT     ~-14.5% rejected -11.6% rejected (neg 0.84)
#:   AMZN      ~+0.4% rejected  +0.2% rejected (neu 0.93)
#:   GOOGL    ~+14.8% clears    +2.3% rejected (neu 0.77)
#: The real model reads "wins antitrust approval and announces AI
#: breakthrough acquisition" as mostly neutral corporate news, so under
#: FinBERT only TSLA clears the 10% gate — a genuine difference of
#: opinion, left as-is rather than retuning the headline to flatter the
#: gate. Re-verify if the weights, MIN_PREDICTED_GAIN_PCT, or the
#: _BULLISH/_BEARISH lexical-prior sets change.
MARKET_SCENARIOS: Final[tuple[tuple[str, str], ...]] = (
    ("TSLA", "TSLA soars on blowout deliveries as it wins major approval and raises full-year guidance"),
    ("AAPL", "AAPL beats modest expectations but issues cautious commentary for next quarter"),
    ("MSFT", "MSFT plunges after fraud allegations spark investor lawsuit and rating downgrade"),
    ("AMZN", "AMZN holds annual shareholder meeting with no guidance change"),
    ("GOOGL", "GOOGL wins antitrust approval and announces AI breakthrough acquisition"),
)


def pick_random_scenario() -> tuple[str, str]:
    """Return a random ``(ticker, headline)`` pair from ``MARKET_SCENARIOS``.

    Selected fresh via ``random.choice`` on every call — each invocation
    of the Tier-0 pipeline that uses this can land on a different ticker
    and sentiment direction, exercising both the accept and reject paths
    of ``execution.validate_and_plan`` across repeated runs instead of
    always replaying the same fixed signal.
    """
    return random.choice(MARKET_SCENARIOS)


def _synthetic_quote(ticker: str) -> Quote:
    """Fully synthetic two-sided quote, deterministic per (ticker, UTC minute).

    Used ONLY when a real Alpaca quote can't be fetched (broker not
    configured, or a transient data-provider error) — see ``fetch_quote``.
    Keeps the service runnable/testable with zero Alpaca credentials at
    all, which is the one thing this synthetic path still needs to do;
    it must NEVER be what an actual submitted order gets priced against
    when a real quote is available (see the incident below).
    """
    now = datetime.now(UTC)
    base = Decimal(_SIMULATED_BASE_PRICES.get(ticker, _DEFAULT_BASE_PRICE))

    # +/- 1% deterministic intraday drift keyed on the current minute.
    seed = hashlib.sha256(f"{ticker}|{now:%Y%m%d%H%M}".encode()).digest()
    drift = (Decimal(seed[0]) / Decimal(255) - Decimal("0.5")) / Decimal(50)
    mid = (base * (Decimal(1) + drift)).quantize(Decimal("0.01"))

    half_spread = (mid * _SPREAD_BPS / Decimal(10_000)).quantize(Decimal("0.01"))
    half_spread = max(half_spread, Decimal("0.01"))

    return Quote(ticker=ticker, bid=mid - half_spread, ask=mid + half_spread, as_of=now)


async def fetch_quote(ticker: str) -> Quote:
    """Return a two-sided quote to price a Tier-0 order against.

    Real Alpaca market data (free IEX feed) is tried FIRST. A limit order
    priced off a fake number can sit far from the real market and simply
    never fill — verified live: every order this agent submitted before
    this fix priced GOOGL/TSLA/NVDA off a synthetic ~$120-250 base price
    while Alpaca's real ask was $359+ for GOOGL alone, and all 8 orders
    sat at ACCEPTED/filled_qty=0 indefinitely, while PFW's ledger had
    already recorded every one of them as a completed, executed trade.
    The synthetic ``_synthetic_quote`` is now a fallback ONLY — broker not
    configured, or Alpaca's data API unavailable/no quote for this symbol
    — so the service stays runnable/testable with zero credentials, but
    never silently substitutes a fake price for a real submitted order.

    The real 14-day ATR (``broker.get_atr``, ad hoc Phase 4) is fetched
    CONCURRENTLY with the quote itself, via ``asyncio.gather`` — genuinely
    at the same time, not sequentially awaited one after the other — and
    fails independently: a quote-fetch failure still falls back to the
    fully synthetic quote as before, while an ATR-fetch failure only
    leaves ``Quote.atr`` as ``None`` (a real quote with no ATR is still
    usable; ``execution.py``'s volatility-parity sizing degrades to its
    notional-only figure in that case). ATR is deliberately handled more
    leniently than the quote itself — it's a sizing REFINEMENT, not
    something this agent's core pricing correctness depends on the way a
    real bid/ask is.
    """
    ticker = ticker.strip().upper()

    # alpaca-py's REST calls are blocking I/O; dispatch both off the
    # event loop the same way _score_sync's CPU-bound work already is.
    # return_exceptions=True so one call's failure doesn't cancel the
    # other — each is handled independently below.
    quote_result, atr_result = await asyncio.gather(
        asyncio.to_thread(broker.get_real_quote, ticker),
        asyncio.to_thread(broker.get_atr, ticker),
        return_exceptions=True,
    )

    if isinstance(quote_result, (ConfigError, broker.RealQuoteUnavailable)):
        logger.warning(
            "Real quote unavailable for %s (%s) — falling back to the synthetic quote",
            ticker, quote_result,
        )
        return _synthetic_quote(ticker)
    if isinstance(quote_result, BaseException):
        raise quote_result

    bid, ask, as_of = quote_result

    atr: Decimal | None = None
    if isinstance(atr_result, BaseException):
        logger.warning(
            "ATR unavailable for %s (%s) — volatility-parity sizing falls back to notional-only",
            ticker, atr_result,
        )
    else:
        atr = atr_result

    return Quote(ticker=ticker, bid=bid, ask=ask, as_of=as_of, atr=atr)


async def fetch_latest_headline(ticker: str) -> str:
    """Return a simulated latest headline for ``ticker``.

    Rotates deterministically on the UTC hour so repeated calls within an hour
    are stable but the feed is not frozen.
    """
    ticker = ticker.strip().upper()
    seed = hashlib.sha256(f"news|{ticker}|{datetime.now(UTC):%Y%m%d%H}".encode())
    index = seed.digest()[0] % len(_HEADLINE_TEMPLATES)
    return _HEADLINE_TEMPLATES[index].format(t=ticker)
