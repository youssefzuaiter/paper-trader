"""The trained up-move model, as the Inference Agent serves it.

Artifact layout (written by ``train_return_model.py``)::

    models/return_model/model.joblib   calibrated scikit-learn classifier
    models/return_model/meta.json      version, feature contract, move table, metrics
    models/return_model/report.md      the out-of-sample evaluation, human-readable

``load`` refuses an artifact whose feature list differs from
``features.FEATURE_NAMES``: a model fed columns in a different order than
it was trained on produces confident nonsense rather than an error.
"""

from __future__ import annotations

import json
import logging
import warnings
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import sklearn

from swarm.features import FEATURE_NAMES

logger = logging.getLogger("swarm.return_model")

ARTIFACT_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "models" / "return_model"


class ArtifactMismatch(RuntimeError):
    """The artifact was trained against a different feature contract."""


@dataclass(frozen=True)
class ReturnModel:
    version: str
    classifier: Any
    #: Interior edges of the predicted-probability bins, ascending.
    move_bin_edges: tuple[float, ...]
    #: Mean realised forward return (%) of the calibration samples in each bin.
    move_bin_means: tuple[float, ...]
    meta: dict[str, Any]

    @classmethod
    def load(cls, directory: Path = ARTIFACT_DIR) -> ReturnModel:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if tuple(meta["feature_names"]) != FEATURE_NAMES:
            raise ArtifactMismatch(
                f"artifact features {meta['feature_names']} != serving features {list(FEATURE_NAMES)}"
            )
        if meta.get("sklearn_version") != sklearn.__version__:
            logger.warning("Model trained with scikit-learn %s, serving with %s",
                           meta.get("sklearn_version"), sklearn.__version__)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            classifier = joblib.load(directory / "model.joblib")
        return cls(
            version=meta["version"],
            classifier=classifier,
            move_bin_edges=tuple(meta["move_table"]["edges"]),
            move_bin_means=tuple(meta["move_table"]["mean_return_pct"]),
            meta=meta,
        )

    def predict(self, row: Sequence[float]) -> tuple[float, float]:
        """``(prob_up, expected_move_pct)`` for one feature row.

        ``expected_move_pct`` is not a second model: it is the average
        realised forward return of calibration-period samples whose
        predicted probability fell in the same bin. An honest,
        data-derived number, not a rescaled sentiment score.
        """
        x = np.asarray([row], dtype=float)
        prob_up = float(self.classifier.predict_proba(x)[0, 1])
        expected = self.move_bin_means[bisect_right(self.move_bin_edges, prob_up)]
        return prob_up, float(expected)
