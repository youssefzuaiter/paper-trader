"""ProsusAI/FinBERT as an int8 ONNX graph — no torch, no transformers.

Shared by the monolith (``inference.py``, behind ``SENTIMENT_MODEL=finbert``)
and the swarm's Inference Agent. Split out so the agent's image carries
ONNX Runtime and a tokenizer (~60 MB) instead of torch (~700 MB).

Xenova/finbert is ProsusAI/finbert exported to ONNX (its config.json names
ProsusAI/finbert as ``_name_or_path`` and keeps the identical label order,
verified against both repos on 2026-09-18). The int8 graph is 110 MB and
never materialises fp32 weights. Label order: 0=positive, 1=negative,
2=neutral.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REPO: Final[str] = "Xenova/finbert"
DEFAULT_FILE: Final[str] = "onnx/model_int8.onnx"
LABELS: Final[tuple[str, str, str]] = ("positive", "negative", "neutral")
#: BERT's positional limit is 512; a headline is a sentence. 128 bounds a
#: pathological input's inference cost without ever truncating a real one.
MAX_TOKENS: Final[int] = 128


@dataclass(frozen=True, slots=True)
class FinBERT:
    """A fast tokenizer plus an ONNX Runtime session.

    ``input_names`` is read off the graph rather than assumed — optimum's
    BERT exports take ``input_ids``/``attention_mask``/``token_type_ids``,
    but an alternative export may drop the third, and feeding an input the
    graph does not declare is an error, not a no-op.
    """

    tokenizer: object
    session: object
    input_names: frozenset[str]


@lru_cache(maxsize=4)
def load(repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE, *, threads: int = 1) -> FinBERT:
    """Download (once, into the Hugging Face cache) and open the int8 graph.

    Imports lazily so importing this module costs nothing until a model is
    actually needed. ``threads=1`` for serving one headline at a time on a
    fractional CPU; batch scoring (training) passes more.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    logger.info("Loading FinBERT %s (%s)", repo, filename)
    model_path = hf_hub_download(repo, filename)
    tokenizer_path = hf_hub_download(repo, "tokenizer.json")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(MAX_TOKENS)

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # ORT "prepacks" every MatMul weight into a second, kernel-optimised
    # copy at session creation. Measured on Linux: +192 MB resident with
    # prepacking, +138 MB without, for 11 ms vs 17 ms per headline — the
    # 54 MB is what keeps a 512 MB instance out of the OOM killer.
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    return FinBERT(tokenizer=tokenizer, session=session, input_names=frozenset(i.name for i in session.get_inputs()))


def _feeds(model: FinBERT, encodings: Sequence[object]) -> dict[str, np.ndarray]:
    feeds = {
        "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.asarray([e.type_ids for e in encodings], dtype=np.int64),
    }
    return {name: value for name, value in feeds.items() if name in model.input_names}


def logits(model: FinBERT, headline: str) -> np.ndarray:
    """The 3 raw logits for one headline."""
    encoding = model.tokenizer.encode(headline)  # type: ignore[attr-defined]
    (out,) = model.session.run(None, _feeds(model, [encoding]))  # type: ignore[attr-defined]
    return np.asarray(out, dtype=np.float32)[0]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def probabilities(model: FinBERT, headlines: Sequence[str], *, workers: int = 1) -> np.ndarray:
    """``(n, 3)`` probabilities in label order, one headline per graph call.

    Deliberately NOT padded batches: the int8 graph quantises activations
    with a scale computed over the whole input tensor, so a headline scored
    inside a batch comes out measurably different (up to 0.025 in a class
    probability, measured) from the same headline scored alone — and the
    Inference Agent scores alone. Training features must be computed the
    way serving computes them. ``workers`` > 1 runs calls on threads (ONNX
    Runtime releases the GIL) for bulk scoring.
    """
    if not headlines:
        return np.zeros((0, 3), dtype=np.float32)

    def score(headline: str) -> np.ndarray:
        return softmax(logits(model, headline))

    if workers <= 1:
        return np.stack([score(h) for h in headlines])
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(score, headlines, chunksize=64)))
