"""Honest statistics (design §10): intervals over days, never over trades.

News arrives in bursts and a symbol-day's samples share a label, so neither
trades nor articles are independent; days are closer. Every interval here is
a **stationary block bootstrap over days** (Politis & Romano: blocks of
geometric length, mean 5 days, wrapping around), with a recorded seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

TRADING_DAYS: Final[int] = 252
MEAN_BLOCK: Final[int] = 5
RESAMPLES: Final[int] = 10_000


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    resamples: int
    seed: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> dict[str, float | int]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "resamples": self.resamples,
                "seed": self.seed}


def stationary_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """One resample of ``range(n)``: a new block starts with probability 1/mean_block."""
    starts = rng.random(n) < 1.0 / mean_block
    starts[0] = True
    begin = rng.integers(0, n, size=n)
    out = np.empty(n, dtype=np.int64)
    current = 0
    for k in range(n):
        current = begin[k] if starts[k] else (current + 1) % n
        out[k] = current
    return out


def block_bootstrap(daily: Sequence[float], stat: Callable[[np.ndarray], float], *, mean_block: int = MEAN_BLOCK,
                    n: int = RESAMPLES, seed: int = 0, level: float = 0.95) -> Interval:
    x = np.asarray(daily, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([stat(x[stationary_indices(len(x), mean_block, rng)]) for _ in range(n)])
    draws = draws[np.isfinite(draws)]
    tail = (1 - level) / 2
    return Interval(float(stat(x)), float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail)), n, seed)


def bootstrap_auc(y: np.ndarray, prob: np.ndarray, day: np.ndarray, *, mean_block: int = MEAN_BLOCK,
                  n: int = 1000, seed: int = 0, level: float = 0.95) -> Interval:
    """AUC with its interval from resampling whole days (in time order)."""
    days, first = np.unique(day, return_index=True)
    ordered = days[np.argsort(first)]
    members = [np.flatnonzero(day == d) for d in ordered]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in stationary_indices(len(ordered), mean_block, rng)])
        if 0 < y[idx].sum() < len(idx):
            draws.append(roc_auc_score(y[idx], prob[idx]))
    tail = (1 - level) / 2
    return Interval(float(roc_auc_score(y, prob)), float(np.quantile(draws, tail)),
                    float(np.quantile(draws, 1 - tail)), n, seed)


def sharpe(daily: np.ndarray) -> float:
    sd = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    return float(np.mean(daily) / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def max_drawdown(daily: np.ndarray) -> tuple[float, int]:
    """Deepest fall of cumulative (additive) return from a peak, and the
    longest stretch in days spent below a previous peak."""
    wealth = np.concatenate([[0.0], np.cumsum(daily)])
    peak = np.maximum.accumulate(wealth)
    depth = float((wealth - peak).min())
    under, longest = 0, 0
    for w, p in zip(wealth, peak, strict=True):
        under = under + 1 if w < p else 0
        longest = max(longest, under)
    return depth, longest


def deflated_sharpe(observed: float, trials_sharpes: Sequence[float], n_days: int, skew: float,
                    kurtosis: float) -> float:
    """Probability that the true Sharpe exceeds the best of ``len(trials_sharpes)``
    configurations under the null (Bailey & López de Prado, 2014), all per day."""
    n_trials = max(len(trials_sharpes), 1)
    variance = float(np.var(trials_sharpes, ddof=1)) if n_trials > 1 else 0.0
    gamma = 0.5772156649
    expected_max = math.sqrt(variance) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                                          + gamma * norm.ppf(1 - 1 / (n_trials * math.e))) if n_trials > 1 else 0.0
    denominator = math.sqrt(max(1 - skew * observed + (kurtosis - 1) / 4 * observed ** 2, 1e-12))
    return float(norm.cdf((observed - expected_max) * math.sqrt(max(n_days - 1, 1)) / denominator))
