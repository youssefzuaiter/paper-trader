"""Trains `autoencoder.py`'s `TransactionAutoencoder` on synthetic "normal"
cash-flow transactions.

Run via ``python -m models.train_autoencoder`` from the repo root (inside
the venv with ``torch`` installed — see ``requirements.txt``). Produces
``autoencoder_checkpoint.pt`` (state_dict) and ``autoencoder_meta.json``
(normalization stats + the reconstruction-error distribution's mean/std,
which is what turns a raw MSE into a Z-score) in this same directory —
both are what ``autoencoder.score_transaction()`` loads at request time.

This app has no real transaction history to train against (paper-trader
has no database connection to PFW's ledger at all) — same honest
constraint PFW's own `ml-pipeline/synthesize_ledger.py` and
`scripts/train-forecaster.py` are already built around. The synthetic
generator below is a plausible, per-category-shaped distribution over
amount/hour, not real data, and this file says so rather than implying
otherwise.
"""

from __future__ import annotations

import json
import math
import random

import torch
from torch import nn, optim

from models.autoencoder import (
    CATEGORY_VOCAB,
    CHECKPOINT_PATH,
    METADATA_PATH,
    AutoencoderMetadata,
    TransactionAutoencoder,
    build_feature_vector,
)

SEED = 20260907
NUM_TRAIN = 6000
NUM_VALIDATION = 1500
EPOCHS = 60
LEARNING_RATE = 0.01
Z_THRESHOLD = 2.5

# Per-category (mu, sigma) of log(amount_usd) — a log-normal draw, since
# real spending amounts are right-skewed and never negative. Plausible,
# not real: exp(mu) is roughly the category's "typical" amount.
CATEGORY_AMOUNT_LOG_PARAMS: dict[str, tuple[float, float]] = {
    "groceries": (3.6, 0.4),       # ~$37 typical
    "dining": (3.1, 0.5),          # ~$22
    "transport": (2.6, 0.6),       # ~$13
    "entertainment": (3.3, 0.6),   # ~$27
    "subscriptions": (2.4, 0.35),  # ~$11, narrow — subscriptions are fixed-price
    "shopping": (3.9, 0.7),        # ~$49, wide spread
    "utilities": (4.3, 0.3),       # ~$74, narrow — bills are fairly fixed
    "other": (3.0, 0.8),           # ~$20, widest — a genuine catch-all
}

#: Center hour(s) for a Gaussian time-of-day draw. `None` means billed
#: automatically at any hour (subscriptions/utilities) -> uniform, not
#: Gaussian. `dining` gets a real bimodal draw (lunch or dinner), not a
#: single averaged center, since a "16:00 average" of two real peaks
#: would put mass exactly where dining transactions almost never occur.
CATEGORY_HOUR_CENTERS: dict[str, list[float] | None] = {
    "groceries": [15.0],
    "dining": [13.0, 19.0],
    "transport": [9.0],
    "entertainment": [20.0],
    "subscriptions": None,
    "shopping": [14.0],
    "utilities": None,
    "other": [12.0],
}
HOUR_STDDEV = 3.0


def sample_amount(category: str, rng: random.Random) -> float:
    mu, sigma = CATEGORY_AMOUNT_LOG_PARAMS[category]
    return math.exp(rng.gauss(mu, sigma))


def sample_hour(category: str, rng: random.Random) -> int:
    centers = CATEGORY_HOUR_CENTERS[category]
    if centers is None:
        return rng.randint(0, 23)
    center = rng.choice(centers)
    return round(rng.gauss(center, HOUR_STDDEV)) % 24


def generate_normal_transactions(n: int, rng: random.Random) -> list[tuple[float, str, int]]:
    transactions = []
    for _ in range(n):
        category = rng.choice(CATEGORY_VOCAB)
        transactions.append((sample_amount(category, rng), category, sample_hour(category, rng)))
    return transactions


def build_anomaly_probes() -> list[tuple[str, float, str, int]]:
    """A handful of deliberately-unrealistic (amount, category, hour) combinations, for the post-training sanity check below — not a real precision/recall study, just confirming the trained model actually separates these from normal traffic before calling this done."""
    return [
        ("$4,800 'dining' at 3am", 4800.0, "dining", 3),
        ("$0.40 'utilities'", 0.40, "utilities", 12),
        ("$9,500 'entertainment'", 9500.0, "entertainment", 21),
        ("$2,200 'groceries' at 4am", 2200.0, "groceries", 4),
    ]


def main() -> None:
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    train_raw = generate_normal_transactions(NUM_TRAIN, rng)
    amount_logs = [math.log1p(amount) for amount, _category, _hour in train_raw]
    amount_log_mean = sum(amount_logs) / len(amount_logs)
    amount_log_variance = sum((value - amount_log_mean) ** 2 for value in amount_logs) / len(amount_logs)
    amount_log_std = math.sqrt(amount_log_variance)

    # Bootstrap metadata just for `build_feature_vector`'s normalization
    # step — the reconstruction-error fields are unused until the model
    # is actually trained, filled with placeholders here.
    bootstrap_metadata = AutoencoderMetadata(
        amount_log_mean=amount_log_mean,
        amount_log_std=amount_log_std,
        mean_normal_mse=0.0,
        std_normal_mse=1.0,
        z_threshold=Z_THRESHOLD,
        trained_on="synthetic",
    )

    X_train = torch.stack(
        [build_feature_vector(amount, category, hour, bootstrap_metadata) for amount, category, hour in train_raw]
    )

    model = TransactionAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        reconstruction = model(X_train)
        loss = loss_fn(reconstruction, X_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:3d}/{EPOCHS}  train MSE: {loss.item():.5f}")

    model.eval()

    # A FRESH, held-out normal set (different draws, same distribution) —
    # its reconstruction-error mean/std is what actually becomes the
    # Z-score baseline, not the training set's own (which would be
    # optimistically low, since the model was fit directly against it).
    validation_raw = generate_normal_transactions(NUM_VALIDATION, rng)
    X_validation = torch.stack(
        [build_feature_vector(amount, category, hour, bootstrap_metadata) for amount, category, hour in validation_raw]
    )
    with torch.no_grad():
        validation_reconstruction = model(X_validation)
        per_sample_mse = torch.mean((validation_reconstruction - X_validation) ** 2, dim=1)

    mean_normal_mse = per_sample_mse.mean().item()
    std_normal_mse = per_sample_mse.std().item()
    print(f"\nvalidation (held-out normal) reconstruction MSE: mean={mean_normal_mse:.5f} std={std_normal_mse:.5f}")

    final_metadata = AutoencoderMetadata(
        amount_log_mean=amount_log_mean,
        amount_log_std=amount_log_std,
        mean_normal_mse=mean_normal_mse,
        std_normal_mse=std_normal_mse,
        z_threshold=Z_THRESHOLD,
        trained_on="synthetic",
    )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "amount_log_mean": final_metadata.amount_log_mean,
                "amount_log_std": final_metadata.amount_log_std,
                "mean_normal_mse": final_metadata.mean_normal_mse,
                "std_normal_mse": final_metadata.std_normal_mse,
                "z_threshold": final_metadata.z_threshold,
                "trained_on": final_metadata.trained_on,
            },
            indent=2,
        )
    )
    print(f"\nsaved {CHECKPOINT_PATH.name} and {METADATA_PATH.name}")

    # Sanity check: how many of the HELD-OUT NORMAL transactions would
    # false-positive at this threshold, and do the deliberately-planted
    # anomalies actually clear it? Printed, not asserted — this is a
    # visibility check for a human reading the training run, same spirit
    # as PFW's own train-forecaster.py's numerical-equivalence check.
    false_positive_rate = (per_sample_mse > mean_normal_mse + final_metadata.z_threshold * std_normal_mse).float().mean().item()
    print(f"\nfalse-positive rate on held-out normal traffic at z={final_metadata.z_threshold}: {false_positive_rate:.2%}")

    print("\nanomaly probes:")
    for label, amount, category, hour in build_anomaly_probes():
        features = build_feature_vector(amount, category, hour, final_metadata)
        with torch.no_grad():
            reconstruction = model(features.unsqueeze(0)).squeeze(0)
            mse = torch.mean((reconstruction - features) ** 2).item()
        z_score = (mse - mean_normal_mse) / std_normal_mse
        flagged = "ANOMALY" if z_score > final_metadata.z_threshold else "normal"
        print(f"  {label:35s} z={z_score:7.2f}  [{flagged}]")


if __name__ == "__main__":
    main()
