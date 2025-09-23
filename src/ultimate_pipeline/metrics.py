"""Metric helpers and calibration curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


@dataclass
class MetricResult:
    auc: float
    brier: float
    ece: float

    def as_dict(self) -> Dict[str, float]:  # pragma: no cover - simple accessor
        return {"auc": self.auc, "brier": self.brier, "ece": self.ece}


def calibration_curve_fixed_bins(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate calibration statistics over a fixed set of probability bins.

    The helper mirrors the logic from the conversational write-up: predictions are
    assigned to ``n_bins`` uniformly spaced buckets across ``[0, 1]`` and the
    function returns the per-bin average prediction, empirical accuracy, sample
    counts, and normalised weights alongside the underlying bin edges.
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # ``np.digitize`` maps each prediction into a bucket index. ``right=False``
    # ensures that the rightmost edge (1.0) is included in the final bin.
    bin_ids = np.digitize(y_pred, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    total = counts.sum()

    sum_pred = np.bincount(bin_ids, weights=y_pred, minlength=n_bins)
    sum_true = np.bincount(bin_ids, weights=y_true, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.divide(sum_pred, counts, out=np.full(n_bins, np.nan), where=counts > 0)
        frac_pos = np.divide(sum_true, counts, out=np.full(n_bins, np.nan), where=counts > 0)

    if total > 0:
        weights = counts / total
    else:  # pragma: no cover - defensive branch for empty inputs
        weights = np.zeros_like(counts)

    return mean_pred, frac_pos, counts, weights, edges


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> float:
    mean_pred, frac_pos, counts, weights, _ = calibration_curve_fixed_bins(y_true, y_pred, n_bins)
    non_empty = counts > 0
    ece = np.sum(weights[non_empty] * np.abs(mean_pred[non_empty] - frac_pos[non_empty]))
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> MetricResult:
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_pred, n_bins)
    return MetricResult(auc=auc, brier=brier, ece=ece)
