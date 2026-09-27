"""Walk-forward training and scoring (design §6, rules P5 and P6).

One fold per month, first scored month 2025-01, last 2026-09 (21 folds):

* **Window**: every sample whose label *resolved* before the fold's
  embargo cut-off, the open of the session five sessions before the
  month's first session. Expanding by default; rolling 12 months as a variant.
* **Calibration**: the window's last 25% by ``made_at``; the classifier
  trains on what precedes it by more than ``PURGE`` and resolved before it
  (the 60 : 20 of ``train_return_model.split``).
* **Per fold, from its window only**: the classifier and its calibration
  (``train_return_model.fit_calibrated``), and for S2 and S3 each: a move
  table and a probability threshold.
* **Scoring**: a prediction whose ``made_at`` falls in the fold's month.

**Threshold rule** (fixed in advance): on the calibration segment, the
lowest of 0.40, 0.42, ..., 0.60 with at least 30 qualifying symbol-days and
a positive mean return net of the central round-trip cost; none → the
fold does not trade. The cost charges fees rounded per order at the
configured order size whatever the level's fee rounding: a threshold must
not depend on how many other trades share a day's rounding. The whole grid
is kept and reported, marked not selectable.

S2's unit is the symbol-night (score = mean ``prob_up`` of the night's
events, D7); S3's is the first qualifying event of a symbol-day.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

import numpy as np
from sklearn.metrics import roc_auc_score

from backtest import costs
from backtest.calendar import Calendar
from backtest.dataset import Samples
from swarm.common import NEW_YORK
from swarm.features import FEATURE_NAMES
from swarm.return_model import ReturnModel
from swarm.train_return_model import GBM_PARAMS, PURGE, fit_calibrated, move_table

logger = logging.getLogger("backtest.walkforward")

THRESHOLD_GRID: Final[tuple[float, ...]] = tuple(round(0.40 + 0.02 * k, 2) for k in range(11))
MIN_SYMBOL_DAYS: Final[int] = 30
STRATEGIES: Final[tuple[str, ...]] = ("S2", "S2_collapsed", "S3", "S3_collapsed")


@dataclass(frozen=True)
class Schedule:
    first_month: date = date(2025, 1, 1)
    last_month: date = date(2026, 9, 1)
    window: str = "expanding"            # expanding | rolling12
    embargo_sessions: int = 5
    calib_fraction: float = 0.25
    purge: timedelta = PURGE
    refit_months: int = 1                # 1 monthly, 3 quarterly

    def months(self) -> list[date]:
        out, m = [], self.first_month
        while m <= self.last_month:
            out.append(m)
            m = _add_months(m, self.refit_months)
        return out

    def as_params(self) -> dict[str, Any]:
        return {"first_month": self.first_month.isoformat(), "last_month": self.last_month.isoformat(),
                "window": self.window, "embargo_sessions": self.embargo_sessions,
                "calib_fraction": self.calib_fraction, "purge_s": self.purge.total_seconds(),
                "refit_months": self.refit_months, "gbm": GBM_PARAMS}


def _add_months(m: date, n: int) -> date:
    y, k = divmod(m.month - 1 + n, 12)
    return date(m.year + y, k + 1, 1)


@dataclass(frozen=True)
class FoldBounds:
    month: date
    start: datetime           # the month's first instant, New York
    end: datetime             # the next refit's first instant
    embargo_cutoff: datetime  # labels resolved strictly before this may train


def fold_bounds(month: date, schedule: Schedule, calendar: Calendar) -> FoldBounds:
    start = datetime(month.year, month.month, 1, tzinfo=NEW_YORK)
    end_month = _add_months(month, schedule.refit_months)
    end = datetime(end_month.year, end_month.month, 1, tzinfo=NEW_YORK)
    first = calendar.next_session(start)
    anchor = calendar.shift(first, -schedule.embargo_sessions)
    if anchor is None:
        raise ValueError(f"calendar too short for a {schedule.embargo_sessions}-session embargo before {month}")
    return FoldBounds(month, start, end, anchor.open_at)


@dataclass
class FoldResult:
    bounds: FoldBounds
    train: np.ndarray
    calib: np.ndarray
    scored: np.ndarray
    model: Any
    model_version: str
    artifact_sha256: str
    prob_scored: np.ndarray
    prob_calib: np.ndarray
    thresholds: dict[str, float | None]
    grids: dict[str, list[dict[str, Any]]]
    move_tables: dict[str, dict[str, list[float]]]
    calib_auc: float | None
    windows: dict[str, Any] = field(default_factory=dict)


# --- the per-strategy units -----------------------------------------------------------------

def _first_per_story(samples: Samples, idx: np.ndarray) -> np.ndarray:
    """Keep each story's first event (by ``made_at``); events without a story stay."""
    seen: set[tuple[str, str]] = set()
    keep = []
    for i in idx[np.argsort(samples.made_at[idx], kind="stable")]:
        story = str(samples.story_id[i])
        key = (str(samples.symbol[i]), story)
        if story and key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return np.asarray(sorted(keep), dtype=int)


def s2_nights(samples: Samples, idx: np.ndarray, prob: np.ndarray, *, collapsed: bool) -> dict[str, np.ndarray]:
    """Symbol-nights among ``idx`` (night samples): score = mean prob, and the night's return."""
    pos = {int(i): k for k, i in enumerate(idx)}
    pick = idx[samples.night[idx]]
    if collapsed:
        pick = _first_per_story(samples, pick)
    groups: dict[tuple[str, int], list[int]] = {}
    for i in pick:
        groups.setdefault((str(samples.symbol[i]), int(samples.session[i])), []).append(int(i))
    keys = sorted(groups)
    return {
        "symbol": np.asarray([k[0] for k in keys]),
        "session": np.asarray([k[1] for k in keys], dtype=int),
        "score": np.asarray([np.mean([prob[pos[i]] for i in groups[k]]) for k in keys]),
        "ret": np.asarray([samples.fwd[groups[k][0]] for k in keys]),
        "first": np.asarray([groups[k][0] for k in keys], dtype=int),
    }


def s3_events(samples: Samples, idx: np.ndarray, prob: np.ndarray, *, collapsed: bool) -> dict[str, np.ndarray]:
    pos = {int(i): k for k, i in enumerate(idx)}
    pick = idx[samples.tradeable[idx]]
    if collapsed:
        pick = _first_per_story(samples, pick)
    return {"symbol": samples.symbol[pick], "session": samples.session[pick],
            "score": np.asarray([prob[pos[int(i)]] for i in pick]), "ret": samples.fwd[pick],
            "first": pick, "made_at": samples.made_at[pick]}


def threshold_rule(units: Mapping[str, np.ndarray], cost: np.ndarray, *, per_symbol_day_first: bool
                   ) -> tuple[float | None, list[dict[str, Any]]]:
    """The lowest grid threshold with >= 30 qualifying symbol-days and a
    positive mean net return; and the whole grid, for the record."""
    grid, chosen = [], None
    for thr in THRESHOLD_GRID:
        mask = units["score"] >= thr
        idx = np.flatnonzero(mask)
        if per_symbol_day_first and len(idx):  # S3: the router trades a symbol-day's first event only
            order = idx[np.argsort(units["made_at"][idx], kind="stable")]
            seen: set[tuple[str, int]] = set()
            firsts = []
            for i in order:
                key = (str(units["symbol"][i]), int(units["session"][i]))
                if key not in seen:
                    seen.add(key)
                    firsts.append(i)
            idx = np.asarray(sorted(firsts), dtype=int)
        n = len(idx)
        mean = float(units["ret"][idx].mean()) if n else None
        net = float((units["ret"][idx] - cost[idx]).mean()) if n else None
        ok = n >= MIN_SYMBOL_DAYS and net is not None and net > 0
        grid.append({"threshold": thr, "symbol_days": n, "mean_return": mean, "mean_net_return": net,
                     "qualifies": ok, "selectable": False})
        if ok and chosen is None:
            chosen = thr
    return chosen, grid


# --- the fold ------------------------------------------------------------------------------

def _hash_arrays(*arrays: np.ndarray, extra: str = "") -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(extra.encode())
    return h.hexdigest()


CostFn = Callable[[Samples, np.ndarray], np.ndarray]


def central_cost(fees: costs.FeeTable, notional: Decimal, calendar: Calendar) -> CostFn:
    """Per-sample round-trip cost at the central level, fees rounded per order (see the module doc)."""
    level = replace(costs.LEVELS["central"], fee_rounding="per_order")
    cache: dict[tuple[str, bool, date, float], float] = {}

    def cost(samples: Samples, idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(idx))
        for k, i in enumerate(idx):
            session = calendar.sessions[int(samples.session[i])]
            opening = samples.entry_at[i] < (session.open_at + costs.OPENING_WINDOW).timestamp()
            key = (str(samples.symbol[i]), bool(opening), session.date, round(float(samples.entry_price[i]), 2))
            if key not in cache:
                cache[key] = costs.round_trip_fraction(
                    key[0], level, fees, day=key[2], price=Decimal(str(key[3])), notional=notional,
                    entry_in_opening_window=key[1])
            out[k] = cache[key]
        return out

    return cost


def fold_thresholds(samples: Samples, calib: np.ndarray, prob_calib: np.ndarray, cost_fn: CostFn) -> tuple[
        dict[str, float | None], dict[str, list[dict[str, Any]]], dict[str, dict[str, list[float]]]]:
    """Per strategy, from the calibration segment only (P6): the threshold rule's
    choice, its whole grid, and the move table. Only the threshold depends on cost."""
    thresholds: dict[str, float | None] = {}
    grids: dict[str, list[dict[str, Any]]] = {}
    tables: dict[str, dict[str, list[float]]] = {}
    for name in STRATEGIES:
        collapsed = name.endswith("_collapsed")
        if name.startswith("S2"):
            units = s2_nights(samples, calib, prob_calib, collapsed=collapsed)
        else:
            units = s3_events(samples, calib, prob_calib, collapsed=collapsed)
        if len(units["score"]) == 0:
            thresholds[name], grids[name] = None, []
            tables[name] = {"edges": [], "mean_return_pct": [0.0]}
            continue
        unit_cost = cost_fn(samples, units["first"])
        thresholds[name], grids[name] = threshold_rule(units, unit_cost,
                                                       per_symbol_day_first=name.startswith("S3"))
        tables[name] = move_table(units["score"], units["ret"])
    return thresholds, grids, tables


def at_cost(samples: Samples, folds: Sequence[FoldResult], cost_fn: CostFn) -> list[FoldResult]:
    """The same fitted folds with thresholds re-chosen under another cost (another
    order size, D3): models, calibration and move tables are unchanged."""
    out = []
    for f in folds:
        thresholds, grids, tables = fold_thresholds(samples, f.calib, f.prob_calib, cost_fn)
        if tables != f.move_tables:
            raise AssertionError("move tables must not depend on cost")
        out.append(replace(f, thresholds=thresholds, grids=grids))
    return out


def fit_fold(samples: Samples, bounds: FoldBounds, schedule: Schedule, cost_fn: CostFn) -> FoldResult:
    made, resolved = samples.made_at, samples.resolved_at
    window = resolved < bounds.embargo_cutoff.timestamp()
    if schedule.window == "rolling12":
        window &= made >= _add_rolling_start(bounds.start).timestamp()
    elif schedule.window != "expanding":
        raise ValueError(schedule.window)
    if window.sum() < 1000:
        raise ValueError(f"{bounds.month}: only {int(window.sum())} samples in the training window")
    cut = float(np.quantile(made[window], 1 - schedule.calib_fraction))
    calib = np.flatnonzero(window & (made >= cut))
    train = np.flatnonzero(window & (made < cut - schedule.purge.total_seconds()) & (resolved < cut))
    scored = np.flatnonzero((made >= bounds.start.timestamp()) & (made < bounds.end.timestamp()))

    model = fit_calibrated(samples.X[train], samples.y[train], samples.X[calib], samples.y[calib])
    prob_calib = model.predict_proba(samples.X[calib])[:, 1]
    prob_scored = model.predict_proba(samples.X[scored])[:, 1] if len(scored) else np.empty(0)

    thresholds, grids, tables = fold_thresholds(samples, calib, prob_calib, cost_fn)
    digest = _hash_arrays(samples.X[train], samples.y[train], samples.X[calib], samples.y[calib],
                          extra=repr(sorted(GBM_PARAMS.items())))
    # The exact pickled artifact. It embeds the OpenMP thread count (in the model's _BinMapper), so two
    # artifacts compare only at the same OMP_NUM_THREADS; every experiment records it. Predictions do not
    # depend on it (verified bit for bit at 1, 2 and 8 threads).
    artifact = hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()
    y_cal = samples.y[calib]
    auc = float(roc_auc_score(y_cal, prob_calib)) if 0 < y_cal.sum() < len(y_cal) else None
    return FoldResult(
        bounds=bounds, train=train, calib=calib, scored=scored, model=model,
        model_version=f"wf-{bounds.month:%Y-%m}-{digest[:10]}", artifact_sha256=artifact,
        prob_scored=prob_scored, prob_calib=prob_calib, thresholds=thresholds, grids=grids, move_tables=tables,
        calib_auc=auc, windows={
            "train": _span(samples, train), "calib": _span(samples, calib),
            "latest_resolved_training_label": float(resolved[np.concatenate([train, calib])].max()),
        })


def _add_rolling_start(start: datetime) -> datetime:
    return start.replace(year=start.year - 1)


def _span(samples: Samples, idx: np.ndarray) -> dict[str, Any]:
    if not len(idx):
        return {"n": 0}
    return {"n": len(idx), "from": float(samples.made_at[idx].min()), "to": float(samples.made_at[idx].max()),
            "up_rate": float(samples.y[idx].mean())}


def run(samples: Samples, schedule: Schedule, calendar: Calendar, cost_fn: CostFn) -> list[FoldResult]:
    folds = []
    for month in schedule.months():
        bounds = fold_bounds(month, schedule, calendar)
        fold = fit_fold(samples, bounds, schedule, cost_fn)
        logger.info("fold %s: train %d, calib %d, scored %d, calib AUC %s, thresholds %s", month,
                    len(fold.train), len(fold.calib), len(fold.scored),
                    None if fold.calib_auc is None else f"{fold.calib_auc:.3f}", fold.thresholds)
        folds.append(fold)
    return folds


def fold_for(folds: list[FoldResult], ts: datetime) -> FoldResult | None:
    """The fold valid at ``ts``: the one whose month contains it."""
    for fold in folds:
        if fold.bounds.start <= ts < fold.bounds.end:
            return fold
    return None


def serving_models(fold: FoldResult) -> dict[str, ReturnModel]:
    """The fold as the Inference Agent would serve it, one per strategy's move
    table: ``ReturnModel.expected_move`` is the production lookup, not a copy."""
    return {name: ReturnModel(version=fold.model_version, classifier=fold.model,
                              move_bin_edges=tuple(table["edges"]), move_bin_means=tuple(table["mean_return_pct"]),
                              meta={"strategy": name})
            for name, table in fold.move_tables.items()}


# --- to the store ----------------------------------------------------------------------------

def model_rows(folds: list[FoldResult], *, experiment_id: str, git_commit: str, data_hash: str,
               label: Mapping[str, Any], schedule: Schedule) -> list[dict[str, Any]]:
    return [{
        "model_version": f.model_version, "experiment_id": experiment_id, "fold_month": f.bounds.month,
        "fold_start": f.bounds.start, "embargo_cutoff": f.bounds.embargo_cutoff,
        "feature_names": list(FEATURE_NAMES), "windows": f.windows, "label": dict(label),
        "params": schedule.as_params(), "seed": int(GBM_PARAMS["random_state"]), "git_commit": git_commit,
        "data_hash": data_hash, "artifact_sha256": f.artifact_sha256, "thresholds": f.thresholds,
        "move_tables": f.move_tables,
    } for f in folds]


def prediction_rows(samples: Samples, folds: list[FoldResult], *, experiment_id: str,
                    mode: str = "backtest") -> list[dict[str, Any]]:
    rows = []
    for f in folds:
        serving = serving_models(f)
        for k, i in enumerate(f.scored):
            prob = float(f.prob_scored[k])
            expected = serving["S2" if samples.night[i] else "S3"].expected_move(prob)
            made_at = datetime.fromtimestamp(float(samples.made_at[i]), NEW_YORK)
            key = f"{samples.event_id[i]}|{samples.symbol[i]}|{f.model_version}|{made_at.isoformat()}"
            rows.append({
                "prediction_id": hashlib.sha256(key.encode()).hexdigest()[:32],
                "event_id": str(samples.event_id[i]), "symbol": str(samples.symbol[i]),
                "model_version": f.model_version, "made_at": made_at, "mode": mode, "experiment_id": experiment_id,
                "inputs": dict(zip(FEATURE_NAMES, map(float, samples.X[i]), strict=True)), "prob_up": prob,
                "expected_move_pct": expected, "horizon": "close",
                "reference_at": datetime.fromtimestamp(float(samples.entry_at[i]), NEW_YORK),
                "reference_price": costs.to_decimal(float(samples.entry_price[i])).quantize(Decimal("0.0001")),
            })
    return rows
