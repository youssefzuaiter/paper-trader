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


# --- the long-term core: compound (wealth-path) metrics, core design §5.1 -----------------------------------
#
# Phase 1's ``max_drawdown`` and its report's totals add daily returns: harmless over 1.7 years, wrong over
# 10.7. They stay as they are (phase 1 reproduces through them); the core uses these.

def index_matrix(n: int, *, mean_block: int = MEAN_BLOCK, resamples: int = RESAMPLES, seed: int = 0) -> np.ndarray:
    """``resamples`` stationary-bootstrap index rows of ``range(n)``, drawn with exactly the random calls,
    in exactly the order, that ``block_bootstrap`` makes: row *j* equals its *j*-th resample. Vectorised;
    every core series in a window is resampled with the same matrix, so paired differences are coherent."""
    rng = np.random.default_rng(seed)
    out = np.empty((resamples, n), dtype=np.int32)
    positions = np.arange(n)
    for j in range(resamples):
        starts = rng.random(n) < 1.0 / mean_block
        starts[0] = True
        begin = rng.integers(0, n, size=n)
        last_start = np.maximum.accumulate(np.where(starts, positions, 0))
        out[j] = (begin[last_start] + positions - last_start) % n
    return out


def wealth(returns: np.ndarray) -> np.ndarray:
    """W₀ = 1, Wₜ = Π(1 + r): the compound path, along the last axis."""
    w = np.cumprod(1.0 + returns, axis=-1)
    ones = np.ones(w.shape[:-1] + (1,))
    return np.concatenate([ones, w], axis=-1)


def cagr(returns: np.ndarray) -> np.ndarray | float:
    """Compound annual growth, annualised by 252 sessions a year (along the last axis)."""
    n = returns.shape[-1]
    growth = np.exp(np.sum(np.log1p(returns), axis=-1))
    return growth ** (TRADING_DAYS / n) - 1.0


def volatility(returns: np.ndarray) -> np.ndarray | float:
    return np.std(returns, axis=-1, ddof=1) * math.sqrt(TRADING_DAYS)


def sharpe_excess(returns: np.ndarray, riskfree: np.ndarray) -> np.ndarray | float:
    x = returns - riskfree
    sd = np.std(x, axis=-1, ddof=1)
    return np.where(sd > 0, np.mean(x, axis=-1) / np.where(sd > 0, sd, 1.0) * math.sqrt(TRADING_DAYS), 0.0)


def sortino(returns: np.ndarray, riskfree: np.ndarray) -> np.ndarray | float:
    x = returns - riskfree
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2, axis=-1))
    return np.where(downside > 0, np.mean(x, axis=-1) / np.where(downside > 0, downside, 1.0)
                    * math.sqrt(TRADING_DAYS), 0.0)


def compound_drawdown(returns: np.ndarray) -> np.ndarray | float:
    """min over t of Wₜ / max₍ₛ≤ₜ₎ Wₛ − 1 (negative), along the last axis."""
    w = wealth(returns)
    return np.min(w / np.maximum.accumulate(w, axis=-1) - 1.0, axis=-1)


def calmar(returns: np.ndarray) -> np.ndarray | float:
    dd = compound_drawdown(returns)
    return np.where(dd < 0, cagr(returns) / np.where(dd < 0, -dd, 1.0), np.inf)


@dataclass(frozen=True)
class Drawdown:
    depth: float                 # negative, compound
    peak: int                    # index into the wealth path (0 = before the first return)
    trough: int
    recovery: int | None         # first index back at the peak, None if never
    longest_under: int           # longest spell below a previous peak, in sessions
    longest_under_from: int
    longest_under_to: int | None  # None: still under at the end
    open_at_end: bool


def drawdown_detail(returns: np.ndarray) -> Drawdown:
    w = wealth(np.asarray(returns, dtype=float))
    peak_path = np.maximum.accumulate(w)
    depth_path = w / peak_path - 1.0
    trough = int(np.argmin(depth_path))
    peak = int(np.argmax(w[: trough + 1]))
    after = np.flatnonzero(w[trough:] >= w[peak])
    recovery = trough + int(after[0]) if len(after) else None
    longest, start, best_from, best_to = 0, None, 0, None
    for k in range(len(w)):
        if w[k] < peak_path[k]:
            start = k - 1 if start is None else start  # the spell starts at its peak
            if k - start > longest:
                longest, best_from, best_to = k - start, start, None
        else:
            if start is not None and start == best_from:
                best_to = k  # the longest spell recovered here
            start = None
    return Drawdown(float(depth_path[trough]), peak, trough, recovery, longest, best_from, best_to,
                    start is not None and start == best_from)


def year_returns(returns: Sequence[float], years: Sequence[int]) -> dict[int, float]:
    """Compound return per calendar year (the first and last may be partial)."""
    out: dict[int, float] = {}
    for r, y in zip(returns, years, strict=True):
        out[y] = (1.0 + out.get(y, 0.0)) * (1.0 + r) - 1.0
    return out


def interval_from(estimate: float, draws: np.ndarray, *, resamples: int, seed: int, level: float = 0.95) -> Interval:
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    tail = (1 - level) / 2
    return Interval(float(estimate), float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail)),
                    resamples, seed)


def resampled(stat: Callable[..., np.ndarray], indices: np.ndarray, *series: np.ndarray,
              chunk: int = 1000) -> np.ndarray:
    """``stat`` on every resample row, in chunks (memory: chunk × n floats per series)."""
    out = np.empty(len(indices))
    for lo in range(0, len(indices), chunk):
        rows = indices[lo:lo + chunk]
        out[lo:lo + chunk] = stat(*(s[rows] for s in series))
    return out
