"""inference.py — the real-FinBERT path behind ``SENTIMENT_MODEL=finbert``.

A fake tokenizer and a fake ONNX session stand in for the 110 MB graph:
what's under test is the plumbing — the feeds are int64, only the inputs
the graph declares are fed, the label order is honoured, the lexical prior
is NOT applied on top of a trained model, and the shadow model still runs
the placeholder network. The one test that needs the real graph is opt-in
(``SENTIMENT_MODEL=finbert``, a one-time download) and pins the behaviour
the README's numbers are quoted from.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest

import inference


class FakeTokenizer:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.seen: list[str] = []

    def encode(self, text: str) -> SimpleNamespace:
        self.seen.append(text)
        return SimpleNamespace(ids=self.ids, attention_mask=[1] * len(self.ids), type_ids=[0] * len(self.ids))


class FakeSession:
    """Answers fixed logits and records exactly what it was fed."""

    def __init__(self, logits: list[float], declared_inputs: tuple[str, ...]) -> None:
        self.logits = logits
        self.declared = declared_inputs
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, _outputs: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        return [np.asarray([self.logits], dtype=np.float32)]


def _install_fake(monkeypatch: pytest.MonkeyPatch, logits: list[float], declared: tuple[str, ...]) -> tuple[FakeTokenizer, FakeSession]:
    tokenizer = FakeTokenizer(ids=[101, 2023, 102])
    session = FakeSession(logits, declared)
    fake = inference._FinBERT(tokenizer=tokenizer, session=session, input_names=frozenset(declared))
    monkeypatch.setattr(inference, "_load_finbert", lambda: fake)
    monkeypatch.setattr(inference, "FINBERT_ENABLED", True)
    return tokenizer, session


def test_placeholder_is_the_default_when_the_env_var_is_unset() -> None:
    if os.getenv("SENTIMENT_MODEL", "").strip().lower() == "finbert":
        pytest.skip("suite is running against the real model")
    assert inference.FINBERT_ENABLED is False
    assert inference.MODEL_NAME == inference.PLACEHOLDER_MODEL_NAME


def test_feeds_are_int64_and_only_the_declared_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer, session = _install_fake(monkeypatch, [0.0, 0.0, 0.0], declared=("input_ids", "attention_mask"))
    inference._finbert_logits("AAPL beats estimates")

    assert tokenizer.seen == ["AAPL beats estimates"]
    (feeds,) = session.feeds
    assert set(feeds) == {"input_ids", "attention_mask"}, "token_type_ids must not be fed to a graph that does not declare it"
    assert all(v.dtype == np.int64 and v.shape == (1, 3) for v in feeds.values())


def test_label_order_is_positive_negative_neutral_and_no_lexical_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strongly negative logits on a headline full of *bullish* keywords: if
    # the placeholder's lexical prior leaked onto the real model, the
    # positive probability would climb. It must not.
    _install_fake(monkeypatch, [-4.0, 4.0, 0.0], declared=("input_ids", "attention_mask", "token_type_ids"))
    signal = asyncio.run(inference.predict_move("TSLA", "TSLA soars beats record upgrade breakthrough wins"))

    assert signal.prob_negative > 0.95
    assert signal.prob_positive < 0.01
    assert signal.predicted_move_pct < -14.0
    assert abs(signal.prob_positive + signal.prob_negative + signal.prob_neutral - 1.0) < 1e-5


def test_shadow_model_keeps_the_placeholder_network(monkeypatch: pytest.MonkeyPatch) -> None:
    _tokenizer, session = _install_fake(monkeypatch, [-4.0, 4.0, 0.0], declared=("input_ids", "attention_mask", "token_type_ids"))
    shadow = asyncio.run(inference._shadow_evaluate("TSLA", "TSLA soars on blowout deliveries and record approval"))

    assert session.feeds == [], "the shadow A/B comparator must never touch the real graph"
    assert shadow.predicted_move_pct > 0, "placeholder lexical scoring still drives the shadow verdict"


@pytest.mark.skipif(os.getenv("SENTIMENT_MODEL", "").strip().lower() != "finbert", reason="needs the 110 MB ONNX download; run with SENTIMENT_MODEL=finbert")
def test_real_graph_scores_the_fixed_scenarios_sensibly_and_deterministically() -> None:
    async def score(ticker: str, headline: str) -> inference.InferenceSignal:
        return await inference.predict_move(ticker, headline)

    tsla = asyncio.run(score(*inference.MARKET_SCENARIOS[0]))
    msft = asyncio.run(score(*inference.MARKET_SCENARIOS[2]))
    amzn = asyncio.run(score(*inference.MARKET_SCENARIOS[3]))
    assert tsla.prob_positive > 0.8 and tsla.predicted_move_pct > 10
    assert msft.prob_negative > 0.8 and msft.predicted_move_pct < -10
    assert amzn.prob_neutral > 0.9 and abs(amzn.predicted_move_pct) < 1
    assert asyncio.run(score(*inference.MARKET_SCENARIOS[0])) .prob_positive == tsla.prob_positive
    assert inference.MODEL_NAME.startswith("finbert-onnx-int8")
