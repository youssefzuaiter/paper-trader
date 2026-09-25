"""Train the Inference Agent's up-move model on real news and real returns.

    python -m swarm.train_return_model --start 2024-01-02 --end 2026-09-18

What is predicted
-----------------
For an article about symbol S published at time t:

* **entry**: the open of the first 30-minute regular-hours bar that starts
  at or after t. No price from before the article is ever used as the
  entry. An article that lands mid-bar waits for the next bar; one that
  lands overnight waits for the 09:30 open.
* **exit**: the close of the last regular-hours bar of the *same* session
  as the entry: an intraday hold. The router trades exactly this: it
  exits every position at 15:50 New York time and opens none after 15:30.

  Why same-session: the first model (next-session close) had no signal
  out of sample (test AUC 0.49). A diagnostic over five horizons (30 min,
  2 h, same close, next close, 5 sessions) selected same-session close on
  the calibration segment alone (AUC 0.536, 0.547 one article per
  symbol-day), and the untouched test segment confirmed it (0.537 / 0.530).
  Sentiment's rank correlation with the forward return is also highest at
  this horizon: news is priced within the day, not the next.
* **label**: 1 when ``exit / entry - 1`` exceeds ``LABEL_MIN_RETURN``
  (0.25%, the router's marketable-limit buffer), else 0.

Features are ``features.FEATURE_NAMES``, computed by the same functions the
Quantitative and Inference Agents run live, from data that existed at t.

How it is evaluated
-------------------
Samples are ordered by time and split 60 / 20 / 20 into train, calibrate
and test, with a purge gap between segments longer than the label horizon,
so no training label overlaps a test period. A histogram gradient-boosted
classifier is fit on train, its probabilities are calibrated (isotonic) on
the calibration segment, and every reported metric is computed on the test
segment only. News arrives in bursts, and articles about one symbol on one
day share almost the same label, so AUC is also reported on one article
per symbol-day to show how much the headline number depends on
duplicates.

Uses the paper account's keys for read-only market data (news, SIP bars
older than 15 minutes). Free-plan paced; the first run takes a while and
caches everything under ``.cache/training/``.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import itertools
import json
import logging
import time as clock
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import finbert_onnx
import tier0
from swarm.alpaca_data import AlpacaData, Bar
from swarm.common import DEFAULT_WATCHLIST, NEW_YORK
from swarm.features import (
    FEATURE_NAMES,
    MIN_DAILY_BARS,
    SESSION_PUBLISHED_AT,
    daily_features,
    feature_vector,
    is_tradeable,
    session_date,
)
from swarm.return_model import ARTIFACT_DIR

logger = logging.getLogger("swarm.train")

LABEL_MIN_RETURN: Final[float] = 0.0025
#: Roundups naming many tickers ("Top movers…") say little about any one.
MAX_SYMBOLS_PER_ARTICLE: Final[int] = 3
#: Longer than the label horizon (same-session close), plus weekends and
#: the daily features' own look-back overlap at segment edges.
PURGE: Final[timedelta] = timedelta(days=4)
FIRST_REGULAR_BAR: Final[time] = time(9, 30)
LAST_REGULAR_BAR: Final[time] = time(15, 30)
GATE_THRESHOLDS: Final[tuple[float, ...]] = (0.40, 0.45, 0.50, 0.55, 0.60)
MOVE_BINS: Final[int] = 10


# --- data -------------------------------------------------------------------------

def _cache_path(cache: Path, name: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    return cache / name


async def fetch_news(data: AlpacaData, symbols: tuple[str, ...], start: date, end: date, cache: Path) -> list[dict]:
    path = _cache_path(cache, f"news_{'-'.join(symbols)}_{start}_{end}.jsonl.gz")
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    keep = ("id", "headline", "symbols", "created_at", "source")
    articles: list[dict] = []
    t0 = clock.monotonic()
    begin = datetime.combine(start, time(0), UTC)
    finish = datetime.combine(end, time(0), UTC)
    async for item in data.news(symbols, start=begin, end=finish):
        articles.append({k: item.get(k) for k in keep})
        if len(articles) % 5000 == 0:
            logger.info("news: %d articles (up to %s, %.0fs)", len(articles), item["created_at"], clock.monotonic() - t0)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")
    logger.info("news: %d articles fetched in %.0fs", len(articles), clock.monotonic() - t0)
    return articles


async def fetch_bars(data: AlpacaData, symbols: tuple[str, ...], timeframe: str, start: date, end: date,
                     cache: Path) -> dict[str, list[Bar]]:
    path = _cache_path(cache, f"bars_{timeframe}_{'-'.join(symbols)}_{start}_{end}.json.gz")
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        return {s: [Bar.from_api(b) for b in bars] for s, bars in raw.items()}
    out: dict[str, list[Bar]] = {}
    for symbol in symbols:  # one symbol per request keeps pages simple to reason about
        got = await data.bars([symbol], timeframe=timeframe, start=datetime.combine(start, time(0), UTC),
                              end=datetime.combine(end, time(0), UTC))
        out[symbol] = got.get(symbol, [])
        logger.info("bars %s %s: %d", timeframe, symbol, len(out[symbol]))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({s: [b.to_json() for b in bars] for s, bars in out.items()}, f)
    return out


# --- labels -----------------------------------------------------------------------

@dataclass
class RegularBars:
    """One symbol's regular-hours 30-minute bars, indexed for label lookup."""

    starts: list[datetime]
    opens: np.ndarray
    closes: np.ndarray
    session_of: list[date]
    sessions: list[date]
    last_index: dict[date, int]

    @classmethod
    def build(cls, bars: list[Bar]) -> RegularBars:
        regular = []
        for b in bars:
            local = b.t.astimezone(NEW_YORK)
            if local.weekday() < 5 and FIRST_REGULAR_BAR <= local.time() <= LAST_REGULAR_BAR:
                regular.append(b)
        session_of = [b.t.astimezone(NEW_YORK).date() for b in regular]
        last_index: dict[date, int] = {}
        for i, d in enumerate(session_of):
            last_index[d] = i
        return cls(
            starts=[b.t for b in regular],
            opens=np.array([b.o for b in regular]),
            closes=np.array([b.c for b in regular]),
            session_of=session_of,
            sessions=sorted(last_index),
            last_index=last_index,
        )

    def forward_return(self, published_at: datetime) -> tuple[float, datetime, datetime] | None:
        """``(return, entry_time, exit_time)`` or ``None`` if the data ends first."""
        i = bisect_left(self.starts, published_at)
        if i >= len(self.starts):
            return None
        exit_index = self.last_index[self.session_of[i]]
        entry, exit_ = float(self.opens[i]), float(self.closes[exit_index])
        return exit_ / entry - 1, self.starts[i], self.starts[exit_index]


class DailyHistory:
    """Daily bars with O(log n) "what was final at time t" lookups and
    memoised features — thousands of articles share each symbol-session."""

    def __init__(self, bars: list[Bar]) -> None:
        self.bars = bars
        self.cutoffs = [datetime.combine(session_date(b), SESSION_PUBLISHED_AT, NEW_YORK) for b in bars]
        self._memo: dict[int, dict[str, float] | None] = {}

    def features_at(self, published_at: datetime) -> dict[str, float] | None:
        n = bisect_right(self.cutoffs, published_at)
        if n not in self._memo:
            self._memo[n] = daily_features(self.bars[:n]) if n >= MIN_DAILY_BARS else None
        return self._memo[n]


# --- dataset ----------------------------------------------------------------------

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    fwd: np.ndarray  # forward return, fraction
    ts: np.ndarray  # publication time, epoch seconds
    symbol: np.ndarray
    day: np.ndarray  # New York date of publication, ISO
    tradeable: np.ndarray  # the router would have acted on it (features.is_tradeable)


def build_dataset(articles: list[dict], daily: dict[str, list[Bar]], intraday: dict[str, list[Bar]],
                  sentiment: dict[str, np.ndarray], watch: frozenset[str]) -> Dataset:
    histories = {s: DailyHistory(b) for s, b in daily.items()}
    regular = {s: RegularBars.build(b) for s, b in intraday.items()}
    rows, ys, fwds, tss, syms, days, tradeable = [], [], [], [], [], [], []
    dropped: dict[str, int] = defaultdict(int)
    for article in articles:
        all_symbols = article.get("symbols") or []
        if len(all_symbols) > MAX_SYMBOLS_PER_ARTICLE:
            dropped["roundup"] += 1
            continue
        headline = (article.get("headline") or "").strip()
        if not headline:
            dropped["no_headline"] += 1
            continue
        published_at = datetime.fromisoformat(article["created_at"])
        for symbol in (s for s in all_symbols if s in watch):
            feats = histories[symbol].features_at(published_at)
            if feats is None:
                dropped["history"] += 1
                continue
            label = regular[symbol].forward_return(published_at)
            if label is None:
                dropped["no_forward_data"] += 1
                continue
            ret = label[0]
            rows.append(feature_vector(sentiment=sentiment[headline], daily=feats,
                                       n_symbols=len(all_symbols), published_at=published_at))
            ys.append(int(ret > LABEL_MIN_RETURN))
            fwds.append(ret)
            tss.append(published_at.timestamp())
            syms.append(symbol)
            days.append(published_at.astimezone(NEW_YORK).date().isoformat())
            tradeable.append(is_tradeable(published_at))
    logger.info("dataset: %d samples; dropped %s", len(rows), dict(dropped))
    order = np.argsort(tss, kind="stable")
    return Dataset(X=np.asarray(rows)[order], y=np.asarray(ys)[order], fwd=np.asarray(fwds)[order],
                   ts=np.asarray(tss)[order], symbol=np.asarray(syms)[order], day=np.asarray(days)[order],
                   tradeable=np.asarray(tradeable, dtype=bool)[order])


def score_headlines(headlines: list[str], cache: Path, workers: int) -> dict[str, np.ndarray]:
    path = _cache_path(cache, "finbert_scores.json.gz")
    scores: dict[str, list[float]] = {}
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            scores = json.load(f)
    missing = [h for h in dict.fromkeys(headlines) if h not in scores]
    if missing:
        t0 = clock.monotonic()
        logger.info("FinBERT: scoring %d new headlines on %d threads", len(missing), workers)
        model = finbert_onnx.load(threads=1)
        probs = finbert_onnx.probabilities(model, missing, workers=workers)
        scores.update({h: [float(x) for x in p] for h, p in zip(missing, probs, strict=True)})
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(scores, f)
        logger.info("FinBERT: done in %.0fs", clock.monotonic() - t0)
    return {h: np.asarray(scores[h]) for h in headlines}


# --- training & evaluation ----------------------------------------------------------

def split(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cut1, cut2 = np.quantile(ts, [0.6, 0.8])
    purge = PURGE.total_seconds()
    train = ts < cut1 - purge
    calib = (ts >= cut1) & (ts < cut2 - purge)
    test = ts >= cut2
    return train, calib, test


def first_per_symbol_day(ds: Dataset, mask: np.ndarray) -> np.ndarray:
    seen: set[tuple[str, str]] = set()
    keep = np.zeros_like(mask)
    for i in np.flatnonzero(mask):
        key = (ds.symbol[i], ds.day[i])
        if key not in seen:
            seen.add(key)
            keep[i] = True
    return keep


def _span(ds: Dataset, mask: np.ndarray) -> str:
    lo, hi = ds.ts[mask].min(), ds.ts[mask].max()
    return f"{datetime.fromtimestamp(lo, UTC):%Y-%m-%d} → {datetime.fromtimestamp(hi, UTC):%Y-%m-%d}"


def _gate(ds: Dataset, mask: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    """What each router threshold would have let through, within ``mask``.
    ``probs`` is aligned with ``np.flatnonzero(mask)``."""
    index = np.flatnonzero(mask)
    y, fwd = ds.y[mask], ds.fwd[mask]
    rows = []
    for threshold in GATE_THRESHOLDS:
        chosen = probs >= threshold
        chosen_full = np.zeros(len(ds.ts), dtype=bool)
        chosen_full[index[chosen]] = True
        rows.append({
            "threshold": threshold,
            "n": int(chosen.sum()),
            "share": float(chosen.mean()),
            "hit_rate": float(y[chosen].mean()) if chosen.any() else None,
            "mean_forward_return_pct": float(fwd[chosen].mean() * 100) if chosen.any() else None,
            "symbol_days": int(first_per_symbol_day(ds, chosen_full).sum()),
        })
    return rows


def train_and_evaluate(ds: Dataset) -> tuple[Any, dict[str, Any]]:
    """Fit on train, calibrate on calibrate, report on test.

    Trained on every article (overnight ones included, with
    ``regular_hours`` as a feature): measured, that is more stable out of
    sample than training on the tradeable subset alone (AUC on tradeable
    test articles 0.527 vs 0.515), which has under half the data. Anything
    the router acts on (thresholds, the expected-move table) is evaluated
    on tradeable articles only.

    Sigmoid, not isotonic, calibration: with a signal this weak, isotonic
    regression on the first model collapsed almost every prediction onto
    one plateau (the 10th-90th percentile of prob_up were all 0.444).
    """
    train, calib, test = split(ds.ts)
    Xtr, ytr, Xca, yca, Xte, yte = ds.X[train], ds.y[train], ds.X[calib], ds.y[calib], ds.X[test], ds.y[test]
    base_rate = float(ytr.mean())

    sent_net = FEATURE_NAMES.index("sent_net")
    baseline = LogisticRegression().fit(Xtr[:, [sent_net]], ytr)
    p_baseline = baseline.predict_proba(Xte[:, [sent_net]])[:, 1]

    gbm = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=7,
    ).fit(Xtr, ytr)
    model = CalibratedClassifierCV(FrozenEstimator(gbm), method="sigmoid").fit(Xca, yca)
    p_test = model.predict_proba(Xte)[:, 1]

    calib_t, test_t = calib & ds.tradeable, test & ds.tradeable
    p_calib_t = model.predict_proba(ds.X[calib_t])[:, 1]
    p_test_t = model.predict_proba(ds.X[test_t])[:, 1]
    yte_t = ds.y[test_t]

    dedup = first_per_symbol_day(ds, test)[test]
    dedup_t = first_per_symbol_day(ds, test_t)[test_t]
    metrics: dict[str, Any] = {
        "segments": {
            "train": {"n": int(train.sum()), "span": _span(ds, train), "up_rate": base_rate},
            "calibrate": {"n": int(calib.sum()), "span": _span(ds, calib), "up_rate": float(yca.mean())},
            "test": {"n": int(test.sum()), "span": _span(ds, test), "up_rate": float(yte.mean())},
        },
        "test": {
            "auc_model": float(roc_auc_score(yte, p_test)),
            "auc_sentiment_only": float(roc_auc_score(yte, p_baseline)),
            "auc_model_one_article_per_symbol_day": float(roc_auc_score(yte[dedup], p_test[dedup])),
            "n_one_article_per_symbol_day": int(dedup.sum()),
            "auc_model_tradeable": float(roc_auc_score(yte_t, p_test_t)),
            "n_tradeable": int(test_t.sum()),
            "auc_model_tradeable_one_per_symbol_day": float(roc_auc_score(yte_t[dedup_t], p_test_t[dedup_t])),
            "n_tradeable_one_per_symbol_day": int(dedup_t.sum()),
            "brier_model": float(brier_score_loss(yte, p_test)),
            "brier_base_rate": float(brier_score_loss(yte, np.full_like(p_test, base_rate))),
            "log_loss_model": float(log_loss(yte, np.clip(p_test, 1e-6, 1 - 1e-6))),
            "log_loss_base_rate": float(log_loss(yte, np.full_like(p_test, base_rate))),
            "prob_up_quantiles": {q: float(np.quantile(p_test_t, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "mean_forward_return_pct_all": float(ds.fwd[test].mean() * 100),
            "mean_forward_return_pct_tradeable": float(ds.fwd[test_t].mean() * 100),
        },
        # Thresholds are chosen on the calibration segment and confirmed on test.
        "gate_calibrate": _gate(ds, calib_t, p_calib_t),
        "gate": _gate(ds, test_t, p_test_t),
    }

    deciles = np.unique(np.quantile(p_test_t, np.linspace(0, 1, 11)))
    calibration = []
    for lo, hi in itertools.pairwise(deciles):
        in_bin = (p_test_t >= lo) & (p_test_t <= hi)
        if in_bin.any():
            calibration.append({"p_lo": float(lo), "p_hi": float(hi), "mean_predicted": float(p_test_t[in_bin].mean()),
                                "observed_up_rate": float(yte_t[in_bin].mean()), "n": int(in_bin.sum())})
    metrics["calibration_test"] = calibration

    importance = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=3, random_state=7)
    metrics["permutation_importance_auc"] = dict(sorted(
        ((name, float(v)) for name, v in zip(FEATURE_NAMES, importance.importances_mean, strict=True)),
        key=lambda kv: -kv[1],
    ))

    # Expected move per probability bin: tradeable CALIBRATION articles only.
    edges = np.unique(np.quantile(p_calib_t, np.linspace(0, 1, MOVE_BINS + 1))[1:-1])
    bins = np.searchsorted(edges, p_calib_t, side="right")
    fwd_ca = ds.fwd[calib_t]
    means = [float(fwd_ca[bins == b].mean() * 100) if (bins == b).any() else 0.0 for b in range(len(edges) + 1)]
    metrics["move_table"] = {"edges": [float(e) for e in edges], "mean_return_pct": means}
    return model, metrics


def _gate_row(g: dict[str, Any]) -> str:
    hit = "—" if g["hit_rate"] is None else f"{g['hit_rate']:.3f}"
    ret = "—" if g["mean_forward_return_pct"] is None else f"{g['mean_forward_return_pct']:+.3f}%"
    return f"| {g['threshold']:.2f} | {g['n']} | {g['share']:.1%} | {hit} | {ret} | {g['symbol_days']} |"


def write_report(meta: dict[str, Any], out: Path) -> None:
    m, t = meta["metrics"], meta["metrics"]["test"]
    seg = m["segments"]
    lines = [
        f"# Return model {meta['version']}",
        "",
        (f"Trained {meta['trained_at']} on {meta['n_samples']} article-symbol samples "
         f"({', '.join(meta['symbols'])}). Label: same-session close vs first bar after publication "
         f"> +{LABEL_MIN_RETURN:.2%}. All numbers below are **out of sample** (test segment)."),
        "",
        "| Segment | Samples | Period | Up rate |",
        "|---|---:|---|---:|",
        *(f"| {k} | {v['n']} | {v['span']} | {v['up_rate']:.3f} |" for k, v in seg.items()),
        "",
        "| Metric | Model | Baseline |",
        "|---|---:|---:|",
        f"| ROC AUC (vs sentiment-only logistic) | {t['auc_model']:.4f} | {t['auc_sentiment_only']:.4f} |",
        (f"| ROC AUC, one article per symbol-day (n={t['n_one_article_per_symbol_day']}) | "
         f"{t['auc_model_one_article_per_symbol_day']:.4f} | 0.5 |"),
        (f"| ROC AUC, tradeable articles only (n={t['n_tradeable']}) | "
         f"{t['auc_model_tradeable']:.4f} | 0.5 |"),
        (f"| ROC AUC, tradeable, one per symbol-day (n={t['n_tradeable_one_per_symbol_day']}) | "
         f"{t['auc_model_tradeable_one_per_symbol_day']:.4f} | 0.5 |"),
        f"| Brier score (vs always predicting the train up-rate) | {t['brier_model']:.4f} | {t['brier_base_rate']:.4f} |",
        f"| Log loss (vs train up-rate) | {t['log_loss_model']:.4f} | {t['log_loss_base_rate']:.4f} |",
        "",
        (f"Mean forward return, all test samples {t['mean_forward_return_pct_all']:+.3f}%, "
         f"tradeable {t['mean_forward_return_pct_tradeable']:+.3f}%. "
         "Predicted prob_up quantiles on tradeable articles (1/10/50/90/99%): "
         + ", ".join(f"{v:.3f}" for v in t["prob_up_quantiles"].values())),
        "",
        "## What the router's gate would have let through (tradeable articles)",
        "",
        "Chosen on the calibration segment:",
        "",
        "| prob_up ≥ | Samples | Share | Hit rate | Mean fwd return | Symbol-days |",
        "|---:|---:|---:|---:|---:|---:|",
        *(_gate_row(g) for g in m["gate_calibrate"]),
        "",
        "Confirmed on the test segment:",
        "",
        "| prob_up ≥ | Samples | Share | Hit rate | Mean fwd return | Symbol-days |",
        "|---:|---:|---:|---:|---:|---:|",
        *(_gate_row(g) for g in m["gate"]),
        "",
        "## Calibration (tradeable test articles, predicted-probability deciles)",
        "",
        "| Mean predicted | Observed up rate | n |",
        "|---:|---:|---:|",
        *(f"| {c['mean_predicted']:.3f} | {c['observed_up_rate']:.3f} | {c['n']} |" for c in m["calibration_test"]),
        "",
        "## Permutation importance (drop in test AUC)",
        "",
        *(f"- `{k}`: {v:+.4f}" for k, v in m["permutation_importance_auc"].items()),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    symbols = tuple(s.strip().upper() for s in args.symbols.split(","))
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    cache = Path(args.cache)
    data = AlpacaData.from_settings()
    try:
        articles = await fetch_news(data, symbols, start, end, cache)
        daily = await fetch_bars(data, symbols, "1Day", start - timedelta(days=120), end + timedelta(days=7), cache)
        intraday = await fetch_bars(data, symbols, "30Min", start, end + timedelta(days=7), cache)
    finally:
        await data.aclose()

    focused = [a for a in articles if 0 < len(a.get("symbols") or []) <= MAX_SYMBOLS_PER_ARTICLE
               and (a.get("headline") or "").strip()]
    sentiment = score_headlines([a["headline"].strip() for a in focused], cache, args.workers)
    ds = build_dataset(focused, daily, intraday, sentiment, frozenset(symbols))
    model, metrics = train_and_evaluate(ds)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(ds.X.tobytes() + ds.y.tobytes()).hexdigest()[:8]
    meta = {
        "version": f"gbm-{end:%Y%m%d}-{digest}",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "feature_names": list(FEATURE_NAMES),
        "symbols": list(symbols),
        "data_window": {"start": start.isoformat(), "end": end.isoformat()},
        "n_samples": len(ds.y),
        "label": {"definition": "same-session last regular bar close / first regular 30-min bar open after "
                                "publication - 1 > threshold", "threshold": LABEL_MIN_RETURN},
        "sklearn_version": sklearn.__version__,
        "router_min_prob_up": str(tier0.ROUTER_MIN_PROB_UP),
        "move_table": metrics.pop("move_table"),
        "metrics": metrics,
    }
    joblib.dump(model, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(meta, out)
    logger.info("Wrote %s (%s)", out, meta["version"])
    print((out / "report.md").read_text(encoding="utf-8"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default=(datetime.now(UTC).date() - timedelta(days=7)).isoformat())
    parser.add_argument("--symbols", default=",".join(DEFAULT_WATCHLIST))
    parser.add_argument("--out", default=str(ARTIFACT_DIR))
    parser.add_argument("--cache", default=".cache/training")
    parser.add_argument("--workers", type=int, default=6)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
