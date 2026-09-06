"""Lightweight PyTorch autoencoder for personal cash-flow anomaly detection.

Phase 3 (ad hoc): flags an individual transaction (amount, category,
time-of-day) as anomalous by how poorly it reconstructs through an
autoencoder trained ONLY on synthetic "normal" transactions. A genuinely
anomalous transaction reconstructs badly (unfamiliar combination of
amount/category/hour), which is the standard autoencoder-anomaly-
detection paradigm — the same one PFW's own client-side spending-anomaly
feature already uses (an LSTM autoencoder over 30-day sequences; this is
the same *idea*, applied to a single transaction's tabular features
instead of a sequence, hence the "1D tabular" framing this phase asked
for).

This module is the RUNTIME half only: it loads a checkpoint
(``autoencoder_checkpoint.pt``) and metadata (``autoencoder_meta.json``)
produced by ``train_autoencoder.py`` and scores one transaction at a
time. It does not train anything itself — same split this repo's other
inference code keeps (``inference.py``'s own placeholder model is loaded,
not trained, at runtime).

Feature vector, in order (11 dims total):
  - ``amount_z``: log1p(amount) z-scored against the TRAINING set's own
    log1p(amount) mean/std (stored in the metadata, not recomputed here)
    — log1p because amounts span a wide range (a $3 coffee to a $2,000
    rent payment) and a linear z-score would let one large-amount
    category dominate the reconstruction error for every OTHER category.
  - ``hour_sin``, ``hour_cos``: cyclical encoding of hour-of-day (0-23) —
    a raw ``hour`` feature would make 23:00 and 00:00 look maximally
    FAR apart numerically despite being one hour apart on a real clock;
    the same reasoning PFW's own cash-flow forecaster already applies to
    day-of-week.
  - one-hot over ``CATEGORY_VOCAB`` (8 dims).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

MODELS_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = MODELS_DIR / "autoencoder_checkpoint.pt"
METADATA_PATH = MODELS_DIR / "autoencoder_meta.json"

#: Fixed, ordered vocabulary — one-hot position is this list's index.
#: Shared by `train_autoencoder.py` (which imports it from here) so the
#: two scripts can never silently disagree about feature order.
CATEGORY_VOCAB: list[str] = [
    "groceries",
    "dining",
    "transport",
    "entertainment",
    "subscriptions",
    "shopping",
    "utilities",
    "other",
]

INPUT_DIM = 1 + 2 + len(CATEGORY_VOCAB)  # amount_z + hour_sin/cos + one-hot category
HIDDEN_DIM = 8
BOTTLENECK_DIM = 4


class TransactionAutoencoder(nn.Module):
    """Encoder ``INPUT_DIM -> HIDDEN_DIM -> BOTTLENECK_DIM``, decoder mirrors it.

    Deliberately tiny — this is 11 input features, not a 384-dim
    embedding or a 30-day sequence; a bigger network would just memorize
    the synthetic training set instead of learning the general shape of
    a "normal" transaction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass(frozen=True)
class AutoencoderMetadata:
    amount_log_mean: float
    amount_log_std: float
    mean_normal_mse: float
    std_normal_mse: float
    z_threshold: float
    trained_on: str


def _load_metadata() -> AutoencoderMetadata:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} not found — run `python -m models.train_autoencoder` first "
            "to produce a real trained checkpoint before calling score_transaction()."
        )
    raw = json.loads(METADATA_PATH.read_text())
    return AutoencoderMetadata(
        amount_log_mean=raw["amount_log_mean"],
        amount_log_std=raw["amount_log_std"],
        mean_normal_mse=raw["mean_normal_mse"],
        std_normal_mse=raw["std_normal_mse"],
        z_threshold=raw["z_threshold"],
        trained_on=raw["trained_on"],
    )


@lru_cache(maxsize=1)
def _load_model() -> tuple[TransactionAutoencoder, AutoencoderMetadata]:
    """Load the trained checkpoint + its normalization/threshold metadata once per process."""
    metadata = _load_metadata()
    model = TransactionAutoencoder()
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def build_feature_vector(amount: float, category: str, hour: int, metadata: AutoencoderMetadata) -> torch.Tensor:
    """Builds the 11-dim feature vector for one transaction — the exact same construction `train_autoencoder.py` uses, so training and inference can never silently drift apart."""
    if hour < 0 or hour > 23:
        raise ValueError(f"hour must be 0-23, received {hour}")

    amount_log = math.log1p(abs(amount))
    amount_z = (amount_log - metadata.amount_log_mean) / metadata.amount_log_std

    hour_radians = 2 * math.pi * (hour / 24)
    hour_sin = math.sin(hour_radians)
    hour_cos = math.cos(hour_radians)

    one_hot = [0.0] * len(CATEGORY_VOCAB)
    normalized_category = category.strip().lower()
    if normalized_category in CATEGORY_VOCAB:
        one_hot[CATEGORY_VOCAB.index(normalized_category)] = 1.0
    else:
        # An unrecognized category is real, expected input (a user's own
        # custom category name) — folded into "other" rather than
        # rejected, same "unrecognized falls back to a catch-all bucket"
        # convention PFW's own subscription radar/anomaly detector use
        # for the identical shape of gap.
        one_hot[CATEGORY_VOCAB.index("other")] = 1.0

    return torch.tensor([amount_z, hour_sin, hour_cos, *one_hot], dtype=torch.float32)


@dataclass(frozen=True)
class AnomalyResult:
    is_anomaly: bool
    reconstruction_error: float
    z_score: float
    threshold: float


def score_transaction(amount: float, category: str, hour: int) -> AnomalyResult:
    """Scores one transaction. Never throws for a malformed `category` (falls back to "other" — see `build_feature_vector`); does throw for an out-of-range `hour`, which is a genuine caller bug, not untrusted free text."""
    model, metadata = _load_model()
    features = build_feature_vector(amount, category, hour, metadata)

    with torch.no_grad():
        reconstruction = model(features.unsqueeze(0)).squeeze(0)
        mse = torch.mean((reconstruction - features) ** 2).item()

    z_score = (mse - metadata.mean_normal_mse) / metadata.std_normal_mse
    return AnomalyResult(
        is_anomaly=z_score > metadata.z_threshold,
        reconstruction_error=mse,
        z_score=z_score,
        threshold=metadata.z_threshold,
    )
