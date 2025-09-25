"""Metric helpers and calibration curves supporting multi-class regimes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import stable_cumsum


Array = np.ndarray


@dataclass
class CalibrationCurveResult:
    """Container describing aggregated calibration statistics for one regime."""

    mean_prediction: Array
    accuracy: Array
    counts: Array
    weights: Array
    edges: Array
    lower_ci: Array
    upper_ci: Array
    bin_strategy: str
    min_count: float

    def as_dict(self) -> Dict[str, Any]:  # pragma: no cover - convenience helper
        return {
            "mean_prediction": self.mean_prediction.tolist(),
            "accuracy": self.accuracy.tolist(),
            "counts": self.counts.tolist(),
            "weights": self.weights.tolist(),
            "edges": self.edges.tolist(),
            "lower_ci": self.lower_ci.tolist(),
            "upper_ci": self.upper_ci.tolist(),
            "bin_strategy": self.bin_strategy,
            "min_count": float(self.min_count),
        }


@dataclass
class MetricResult:
    """Structured bundle of metrics for binary or multi-class predictions."""

    auc: float
    brier: float
    ece: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:  # pragma: no cover - simple accessor
        payload: Dict[str, Any] = {"auc": float(self.auc), "brier": float(self.brier), "ece": float(self.ece)}
        if self.extra:
            payload["extra"] = self.extra
        return payload


def _wilson_interval(successes: Array, counts: Array, z: float = 1.96) -> Tuple[Array, Array]:
    successes = np.asarray(successes, dtype=float)
    counts = np.asarray(counts, dtype=float)
    lower = np.full_like(successes, np.nan, dtype=float)
    upper = np.full_like(successes, np.nan, dtype=float)
    mask = counts > 0
    if not np.any(mask):
        return lower, upper
    p_hat = np.divide(successes[mask], counts[mask])
    denom = 1.0 + (z**2) / counts[mask]
    center = (p_hat + (z**2) / (2.0 * counts[mask])) / denom
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / counts[mask] + (z**2) / (4.0 * counts[mask] ** 2)) / denom
    lower[mask] = center - margin
    upper[mask] = center + margin
    return lower, upper


def _uniform_edges(n_bins: int) -> Array:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _ensure_1d(array: ArrayLike) -> Array:
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array of probabilities for binary calibration.")
    return arr


def _validate_n_bins(n_bins: int) -> int:
    if int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer")
    return int(n_bins)


def _equal_mass_bins(
    y_prob: Array,
    sample_weight: Array | None,
    n_bins: int,
) -> Tuple[List[Tuple[int, int]], Array]:
    """Return index ranges defining approximately equal-mass bins.

    The helper sorts predictions and then slices them into ``n_bins`` contiguous
    segments with equal total weight (or count when ``sample_weight`` is ``None``).
    """

    order = np.argsort(y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    if sample_weight is None:
        weights = np.ones_like(sorted_prob, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)[order]
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return [(0, 0)] * n_bins, np.concatenate([np.zeros(1), np.ones(1)])
    cumulative = stable_cumsum(weights)
    target = np.linspace(0.0, total_weight, n_bins + 1)
    bins: List[Tuple[int, int]] = []
    edges = np.zeros(n_bins + 1, dtype=float)
    edges[0] = 0.0
    start = 0
    n_samples = len(sorted_prob)
    for idx in range(1, n_bins):
        threshold = target[idx]
        end = int(np.searchsorted(cumulative, threshold, side="right"))
        end = min(max(end, start + 1), n_samples)
        bins.append((start, end))
        edges[idx] = float(sorted_prob[end - 1]) if end > 0 else 0.0
        start = end
    bins.append((start, n_samples))
    edges[-1] = 1.0
    return bins, edges


def calibration_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int,
    *,
    sample_weight: ArrayLike | None = None,
    strategy: str = "equal_mass",
    min_count: float = 10.0,
    base_rate: float | None = None,
) -> CalibrationCurveResult:
    """Compute calibration curve statistics for binary targets.

    Parameters
    ----------
    y_true:
        Array of binary labels (0/1).
    y_prob:
        Array of predicted probabilities for the positive class.
    n_bins:
        Number of calibration bins to construct.
    sample_weight:
        Optional per-sample weights.
    strategy:
        Either ``"equal_mass"`` (default) or ``"uniform"`` for fixed-width bins.
    min_count:
        Minimum effective sample size per bin before shrinkage is applied.
    base_rate:
        Optional base rate used when shrinking low-count bins. When ``None`` the
        weighted prevalence in ``y_true`` is used.
    """

    n_bins = _validate_n_bins(n_bins)
    y_prob = _ensure_1d(np.asarray(y_prob, dtype=float))
    y_true = _ensure_1d(np.asarray(y_true, dtype=float))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != y_prob.shape:
            raise ValueError("sample_weight must match the shape of y_prob")

    if base_rate is None:
        if sample_weight is None:
            base_rate = float(np.mean(y_true)) if y_true.size else 0.0
        else:
            total_weight = float(np.sum(sample_weight))
            base_rate = float(np.sum(sample_weight * y_true) / total_weight) if total_weight > 0 else 0.0

    if strategy not in {"equal_mass", "uniform"}:
        raise ValueError("strategy must be one of {'equal_mass','uniform'}")

    if strategy == "uniform":
        edges = _uniform_edges(n_bins)
        bin_ids = np.digitize(y_prob, edges, right=False) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
        if sample_weight is None:
            weights = counts.copy()
            successes = np.bincount(bin_ids, weights=y_true, minlength=n_bins)
            mean_pred = np.bincount(bin_ids, weights=y_prob, minlength=n_bins)
        else:
            weights = np.bincount(bin_ids, weights=sample_weight, minlength=n_bins)
            successes = np.bincount(bin_ids, weights=sample_weight * y_true, minlength=n_bins)
            mean_pred = np.bincount(bin_ids, weights=sample_weight * y_prob, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            confidence = np.divide(mean_pred, weights, out=np.full(n_bins, np.nan), where=weights > 0)
            accuracy = np.divide(successes, weights, out=np.full(n_bins, np.nan), where=weights > 0)
        lower_ci, upper_ci = _wilson_interval(successes, counts)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return CalibrationCurveResult(confidence, accuracy, counts, weights, edges, lower_ci, upper_ci, strategy, min_count)

    # Equal-mass binning
    if sample_weight is None:
        sample_weight = np.ones_like(y_prob, dtype=float)
    bins, edges = _equal_mass_bins(y_prob, sample_weight, n_bins)
    mean_prediction = np.zeros(n_bins, dtype=float)
    accuracy = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    effective_weights = np.zeros(n_bins, dtype=float)
    successes = np.zeros(n_bins, dtype=float)

    sorted_indices = np.argsort(y_prob, kind="mergesort")
    sorted_prob = y_prob[sorted_indices]
    sorted_true = y_true[sorted_indices]
    sorted_weight = np.asarray(sample_weight, dtype=float)[sorted_indices]

    total_weight = float(np.sum(sorted_weight))
    for idx, (start, end) in enumerate(bins):
        if end <= start:
            counts[idx] = 0.0
            effective_weights[idx] = 0.0
            mean_prediction[idx] = np.nan
            accuracy[idx] = np.nan
            continue
        slice_weight = sorted_weight[start:end]
        slice_true = sorted_true[start:end]
        slice_prob = sorted_prob[start:end]
        weight_sum = float(np.sum(slice_weight))
        counts[idx] = float(np.sum(slice_weight))
        effective_weights[idx] = weight_sum / total_weight if total_weight > 0 else 0.0
        successes[idx] = float(np.sum(slice_weight * slice_true))
        mean_prediction[idx] = float(np.average(slice_prob, weights=slice_weight))
        raw_accuracy = successes[idx] / weight_sum if weight_sum > 0 else np.nan
        if weight_sum > 0:
            shrink = max(0.0, min_count - weight_sum)
            adjusted_accuracy = (successes[idx] + base_rate * shrink) / (weight_sum + shrink)
        else:
            adjusted_accuracy = base_rate
        accuracy[idx] = float(adjusted_accuracy)

    lower_ci, upper_ci = _wilson_interval(successes, counts)
    if np.sum(effective_weights) > 0:
        effective_weights = effective_weights / np.sum(effective_weights)
    return CalibrationCurveResult(
        mean_prediction,
        accuracy,
        counts,
        effective_weights,
        edges,
        lower_ci,
        upper_ci,
        strategy,
        float(min_count),
    )


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int,
    *,
    sample_weight: ArrayLike | None = None,
    strategy: str = "equal_mass",
    min_count: float = 10.0,
    base_rate: float | None = None,
) -> Tuple[float, CalibrationCurveResult]:
    """Compute the Expected Calibration Error and return the supporting curve."""

    curve = calibration_curve(
        y_true,
        y_prob,
        n_bins,
        sample_weight=sample_weight,
        strategy=strategy,
        min_count=min_count,
        base_rate=base_rate,
    )
    mask = np.isfinite(curve.mean_prediction) & np.isfinite(curve.accuracy)
    ece = float(np.sum(curve.weights[mask] * np.abs(curve.mean_prediction[mask] - curve.accuracy[mask])))
    return ece, curve


def _brier_score(y_true: Array, y_prob: Array, sample_weight: Array | None = None) -> float:
    diff = y_prob - y_true
    squared = diff**2
    if sample_weight is None:
        return float(np.mean(np.sum(squared, axis=1)))
    weight = np.asarray(sample_weight, dtype=float)
    return float(np.sum(np.sum(squared, axis=1) * weight) / np.sum(weight))


def _prepare_probabilities(y_prob: ArrayLike) -> Array:
    probs = np.asarray(y_prob, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.ndim != 2:
        raise ValueError("y_prob must be a one- or two-dimensional array of probabilities")
    return probs


def _labels_to_indices(labels: ArrayLike, class_labels: Array | None, n_classes: int) -> Array:
    label_array = np.asarray(labels)
    if label_array.ndim == 2:
        if label_array.shape[1] != n_classes:
            raise ValueError("Provided y_true has unexpected shape for one-hot encoding")
        return np.argmax(label_array, axis=1)
    if class_labels is None:
        unique = np.unique(label_array)
        if unique.size != n_classes:
            unique = unique[:n_classes]
        mapping = {label: idx for idx, label in enumerate(unique)}
    else:
        mapping = {label: idx for idx, label in enumerate(class_labels)}
    indices = np.empty(label_array.shape[0], dtype=int)
    for i, value in enumerate(label_array):
        if value not in mapping:
            raise ValueError(f"Label {value!r} not found in class labels {list(mapping.keys())}.")
        indices[i] = mapping[value]
    return indices


def _to_one_hot(y_true: ArrayLike, class_labels: Array | None, n_classes: int) -> Array:
    indices = _labels_to_indices(y_true, class_labels, n_classes)
    one_hot = np.zeros((indices.shape[0], n_classes), dtype=float)
    one_hot[np.arange(indices.shape[0]), indices] = 1.0
    return one_hot


def _top_label_ece(
    y_true: Array,
    y_prob: Array,
    n_bins: int,
    *,
    sample_weight: Array | None = None,
    strategy: str = "equal_mass",
    min_count: float = 10.0,
) -> Tuple[float, CalibrationCurveResult]:
    top_labels = np.argmax(y_prob, axis=1)
    true_indices = np.argmax(y_true, axis=1)
    top_probs = y_prob[np.arange(len(y_prob)), top_labels]
    top_correct = (top_labels == true_indices).astype(float)
    return expected_calibration_error(
        top_correct,
        top_probs,
        n_bins,
        sample_weight=sample_weight,
        strategy=strategy,
        min_count=min_count,
    )


def compute_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int,
    *,
    sample_weight: ArrayLike | None = None,
    strategy: str = "equal_mass",
    min_count: float = 10.0,
    class_labels: ArrayLike | None = None,
) -> MetricResult:
    """Compute AUC, Brier score, and ECE with rich calibration diagnostics."""

    probs = _prepare_probabilities(y_prob)
    n_samples, n_columns = probs.shape
    if n_samples == 0:
        return MetricResult(auc=float("nan"), brier=float("nan"), ece=float("nan"))

    if n_columns == 1:
        labels = np.asarray(y_true, dtype=float).reshape(-1)
        auc = (
            roc_auc_score(labels, probs[:, 0], sample_weight=sample_weight)
            if len(np.unique(labels)) > 1
            else float("nan")
        )
        diff = (labels - probs[:, 0]) ** 2
        if sample_weight is None:
            brier = float(np.mean(diff))
        else:
            weights = np.asarray(sample_weight, dtype=float)
            brier = float(np.sum(diff * weights) / np.sum(weights))
        ece, curve = expected_calibration_error(
            labels,
            probs[:, 0],
            n_bins,
            sample_weight=sample_weight,
            strategy=strategy,
            min_count=min_count,
        )
        extra = {"calibration_curve": curve.as_dict()}
        return MetricResult(auc=auc, brier=brier, ece=ece, extra=extra)

    # Multi-class handling
    n_classes = n_columns
    class_labels_array = np.asarray(class_labels) if class_labels is not None else None
    y_true_hot = _to_one_hot(y_true, class_labels_array, n_classes)
    true_labels = np.argmax(y_true_hot, axis=1)
    auc = roc_auc_score(y_true_hot, probs, multi_class="ovr") if n_classes > 1 else float("nan")
    brier = _brier_score(y_true_hot, probs, sample_weight)

    per_class_ece: Dict[str, float] = {}
    per_class_curves: Dict[str, Dict[str, Any]] = {}
    max_error = 0.0
    for class_idx in range(n_classes):
        label_key = str(class_labels_array[class_idx]) if class_labels_array is not None else str(class_idx)
        class_ece, class_curve = expected_calibration_error(
            y_true_hot[:, class_idx],
            probs[:, class_idx],
            n_bins,
            sample_weight=sample_weight,
            strategy=strategy,
            min_count=min_count,
        )
        per_class_ece[label_key] = class_ece
        per_class_curves[label_key] = class_curve.as_dict()
        max_error = max(max_error, class_ece)

    mean_ece = float(np.mean(list(per_class_ece.values())))
    top_ece, top_curve = _top_label_ece(
        y_true_hot,
        probs,
        n_bins,
        sample_weight=sample_weight,
        strategy=strategy,
        min_count=min_count,
    )
    extra = {
        "calibration_curve": per_class_curves,
        "per_class_ece": per_class_ece,
        "top_label_ece": top_ece,
        "top_label_curve": top_curve.as_dict(),
        "max_calibration_error": max_error,
    }
    return MetricResult(auc=auc, brier=brier, ece=mean_ece, extra=extra)
