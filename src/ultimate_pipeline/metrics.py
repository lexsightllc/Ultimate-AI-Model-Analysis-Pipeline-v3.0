"""Metric helpers and calibration curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


@dataclass
class MetricResult:
    auc: float
    brier: float
    ece: float

    def as_dict(self) -> Dict[str, float]:  # pragma: no cover - simple accessor
        return {"auc": self.auc, "brier": self.brier, "ece": self.ece}


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= start) & (y_pred < end)
        if np.any(mask):
            bin_conf = y_pred[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += abs(bin_conf - bin_acc) * mask.mean()
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> MetricResult:
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_pred, n_bins)
    return MetricResult(auc=auc, brier=brier, ece=ece)
