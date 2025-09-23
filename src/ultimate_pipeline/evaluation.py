"""Utilities for evaluating prediction files outside the training pipeline."""
from __future__ import annotations

"""Utilities for evaluating saved prediction files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .metrics import MetricResult, calibration_curve_fixed_bins, compute_metrics


@dataclass
class CalibrationBin:
    """Container holding aggregated calibration statistics for a probability bin."""

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float
    weight: float

    def as_dict(self) -> dict[str, float | int]:  # pragma: no cover - convenience helper
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "accuracy": self.accuracy,
            "confidence": self.confidence,
            "weight": self.weight,
        }


@dataclass
class EvaluationSummary:
    """Summary object returned by :func:`evaluate_prediction_file`."""

    metrics: MetricResult
    n_samples: int
    positive_rate: float
    prediction_mean: float
    calibration_table: List[CalibrationBin]
    label_column: str
    prediction_column: str

    def as_dict(self) -> dict:
        """Serialise the evaluation summary into a JSON-friendly dictionary."""

        payload = {
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "prediction_mean": self.prediction_mean,
            "label_column": self.label_column,
            "prediction_column": self.prediction_column,
            "metrics": self.metrics.as_dict(),
            "calibration_table": [bin_.as_dict() for bin_ in self.calibration_table],
        }
        return payload


def _build_calibration_table(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_bins: int
) -> List[CalibrationBin]:
    mean_pred, frac_pos, counts, weights, edges = calibration_curve_fixed_bins(y_true, y_pred, n_bins)
    bins: List[CalibrationBin] = []
    for idx in range(len(edges) - 1):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        count = int(counts[idx])
        if count == 0:
            accuracy = float("nan")
            confidence = float("nan")
        else:
            accuracy = float(frac_pos[idx])
            confidence = float(mean_pred[idx])
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=count,
                accuracy=accuracy,
                confidence=confidence,
                weight=float(weights[idx]),
            )
        )
    if bins:
        bins[-1] = CalibrationBin(
            lower=bins[-1].lower,
            upper=1.0,
            count=bins[-1].count,
            accuracy=bins[-1].accuracy,
            confidence=bins[-1].confidence,
            weight=bins[-1].weight,
        )
    return bins


def evaluate_prediction_file(
    path: Path | str,
    *,
    label_column: str = "rule_violation",
    prediction_column: str = "prediction",
    n_bins: int = 10,
    clip: bool = True,
    epsilon: float = 1e-6,
) -> EvaluationSummary:
    """Evaluate a CSV containing true labels and model predictions.

    Parameters
    ----------
    path:
        Location of the CSV file.
    label_column:
        Name of the ground-truth column containing binary labels.
    prediction_column:
        Name of the column holding predicted probabilities.
    n_bins:
        Number of bins used when computing the Expected Calibration Error (ECE).
    clip:
        Whether to clip predictions to the open interval (0, 1) using ``epsilon``.
    epsilon:
        Minimum/maximum value applied when ``clip`` is True.
    """

    df = pd.read_csv(path)
    missing = [col for col in [label_column, prediction_column] if col not in df.columns]
    if missing:
        raise ValueError(f"File {path} is missing required columns: {missing}")

    y_true = df[label_column].astype(float).to_numpy()
    y_pred = df[prediction_column].astype(float).to_numpy()

    if not np.isfinite(y_true).all():
        raise ValueError("Label column contains non-finite values.")
    if not np.isfinite(y_pred).all():
        raise ValueError("Prediction column contains non-finite values.")

    uniques = np.unique(y_true)
    if not np.all(np.isin(uniques, (0.0, 1.0))):
        raise ValueError("Labels must be binary (0 or 1).")

    if clip:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    metrics = compute_metrics(y_true, y_pred, n_bins)
    calibration_table = _build_calibration_table(y_true, y_pred, n_bins=n_bins)
    summary = EvaluationSummary(
        metrics=metrics,
        n_samples=int(len(df)),
        positive_rate=float(y_true.mean()) if len(y_true) else float("nan"),
        prediction_mean=float(y_pred.mean()) if len(y_pred) else float("nan"),
        calibration_table=calibration_table,
        label_column=label_column,
        prediction_column=prediction_column,
    )
    return summary
