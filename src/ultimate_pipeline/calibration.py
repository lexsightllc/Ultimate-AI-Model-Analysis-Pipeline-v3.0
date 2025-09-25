"""Calibration strategies with configurable backends and multi-class support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


Array = np.ndarray


def _softmax(logits: Array) -> Array:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs


def _clip_probs(proba: Array, clip: float) -> Array:
    clipped = np.clip(proba, clip, 1 - clip)
    row_sum = clipped.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return clipped / row_sum


def _predict_proba(model, X, clip: float) -> Array:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision, dtype=float)
        if decision.ndim == 1:
            proba = _softmax(np.column_stack([np.zeros_like(decision), decision]))
        else:
            proba = _softmax(decision)
    else:
        raise ValueError(
            f"Model {type(model).__name__} does not support predict_proba or decision_function."
        )
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if proba.shape[1] == 1:
        proba = np.column_stack([1.0 - proba[:, 0], proba[:, 0]])
    return _clip_probs(proba, clip)


def _encode_labels(y: ArrayLike, classes: Array) -> Array:
    y_array = np.asarray(y)
    if y_array.ndim == 2:
        return np.argmax(y_array, axis=1)
    mapping: Dict[Any, int] = {cls: idx for idx, cls in enumerate(classes)}
    return np.array([mapping.get(label, int(label)) for label in y_array], dtype=int)


def _negative_log_likelihood(
    probabilities: Array,
    targets: Array,
    *,
    sample_weight: Optional[Array] = None,
) -> float:
    eps = 1e-12
    chosen = probabilities[np.arange(len(probabilities)), targets]
    chosen = np.clip(chosen, eps, 1.0)
    log_likelihood = np.log(chosen)
    if sample_weight is None:
        return float(-np.mean(log_likelihood))
    weights = np.asarray(sample_weight, dtype=float)
    weights = weights / weights.sum()
    return float(-np.sum(weights * log_likelihood))


@dataclass
class CalibratedModel:
    """Wrapper exposing calibrated probability predictions."""

    model: Any
    calibrator: Any
    clip: float
    classes_: Array

    def predict_proba(self, X) -> Array:
        proba = self.calibrator.predict_proba(X)
        return _clip_probs(proba, self.clip)

    def predict(self, X) -> Array:
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return np.argmax(proba, axis=1)


@dataclass
class Calibrator:
    method: str
    clip: float = 1e-6

    def __post_init__(self) -> None:
        self.clip = float(self.clip)

    def needs_calibration(self) -> bool:
        return (self.method or "none").lower() not in {"none", "off", "identity", "uncalibrated"}

    def calibrate(
        self,
        model,
        X_val=None,
        y_val=None,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> CalibratedModel:
        method = (self.method or "none").lower()
        classes = getattr(model, "classes_", None)
        if classes is None and X_val is not None:
            proba = _predict_proba(model, X_val, self.clip)
            classes = np.arange(proba.shape[1])
        elif classes is None:
            classes = np.array([0, 1])

        if method in {"none", "off", "identity", None}:
            return CalibratedModel(model, _IdentityBackend(model, self.clip), self.clip, np.asarray(classes))

        if X_val is None or y_val is None:
            raise ValueError(f"Calibration method '{method}' requires validation data.")

        calibrator: _BaseBackend
        if method == "isotonic":
            calibrator = _IsotonicBackend(model, self.clip)
        elif method in {"sigmoid", "platt"}:
            calibrator = _SigmoidBackend(model, self.clip)
        elif method in {"temperature", "temperature_scaling"}:
            calibrator = _TemperatureBackend(model, self.clip)
        elif method in {"vector", "vector_scaling"}:
            calibrator = _VectorScalingBackend(model, self.clip)
        elif method in {"matrix", "matrix_scaling"}:
            calibrator = _MatrixScalingBackend(model, self.clip)
        elif method in {"dirichlet", "dirichlet_calibration"}:
            calibrator = _DirichletBackend(model, self.clip)
        else:
            raise ValueError(f"Unknown calibration method '{self.method}'.")

        calibrator.fit(X_val, y_val, sample_weight=sample_weight)
        return CalibratedModel(model, calibrator, self.clip, calibrator.classes_)


def make_identity_calibrator(model, clip: float, X=None, y=None) -> CalibratedModel:
    """Return an identity calibration wrapper for convenience."""

    backend = _IdentityBackend(model, clip)
    backend.fit(X, y)
    classes = backend.classes_ if backend.classes_.size else np.asarray(getattr(model, "classes_", np.array([0, 1])))
    return CalibratedModel(model, backend, clip, classes)


class _BaseBackend:
    def __init__(self, model, clip: float) -> None:
        self.model = model
        self.clip = float(clip)
        self.classes_ = np.asarray(getattr(model, "classes_", None))

    def fit(self, X, y, *, sample_weight=None) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict_proba(self, X) -> Array:
        return _predict_proba(self.model, X, self.clip)


class _IdentityBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:  # pragma: no cover - nothing to fit
        if self.classes_.size == 0 and X is not None:
            proba = _predict_proba(self.model, X, self.clip)
            self.classes_ = np.arange(proba.shape[1])
        elif self.classes_.size == 0:
            self.classes_ = np.array([0, 1])


class _IsotonicBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:
        proba = _predict_proba(self.model, X, self.clip)
        if proba.shape[1] != 2:
            raise ValueError("Isotonic calibration currently supports binary problems only.")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba[:, 1], y, sample_weight=sample_weight)
        self._iso = iso
        self.classes_ = np.asarray(getattr(self.model, "classes_", np.array([0, 1])))

    def predict_proba(self, X) -> Array:
        base = _predict_proba(self.model, X, self.clip)
        calibrated_pos = self._iso.transform(base[:, 1])
        calibrated_pos = np.clip(calibrated_pos, self.clip, 1 - self.clip)
        calibrated = np.column_stack([1.0 - calibrated_pos, calibrated_pos])
        return _clip_probs(calibrated, self.clip)


class _SigmoidBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:
        calibrator = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid")
        calibrator.fit(X, y, sample_weight=sample_weight)
        self._calibrator = calibrator
        self.classes_ = np.asarray(calibrator.classes_)

    def predict_proba(self, X) -> Array:
        calibrated = self._calibrator.predict_proba(X)
        return _clip_probs(calibrated, self.clip)


class _TemperatureBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:
        proba = _predict_proba(self.model, X, self.clip)
        logits = np.log(proba)
        classes = np.asarray(getattr(self.model, "classes_", np.arange(proba.shape[1])))
        targets = _encode_labels(y, classes)

        def loss(log_temperature: Array) -> float:
            temperature = np.exp(log_temperature[0])
            scaled = logits / temperature
            calibrated = _softmax(scaled)
            return _negative_log_likelihood(calibrated, targets, sample_weight=sample_weight)

        result = minimize(loss, x0=np.array([0.0]), method="L-BFGS-B")
        if not result.success:  # pragma: no cover - numerical corner case
            raise RuntimeError(f"Temperature scaling failed to converge: {result.message}")
        self._temperature = float(np.exp(result.x[0]))
        self.classes_ = classes

    def predict_proba(self, X) -> Array:
        proba = _predict_proba(self.model, X, self.clip)
        logits = np.log(proba)
        scaled = logits / self._temperature
        calibrated = _softmax(scaled)
        return _clip_probs(calibrated, self.clip)


class _VectorScalingBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:
        proba = _predict_proba(self.model, X, self.clip)
        logits = np.log(proba)
        n_classes = logits.shape[1]
        classes = np.asarray(getattr(self.model, "classes_", np.arange(n_classes)))
        targets = _encode_labels(y, classes)

        def unpack(params: Array) -> tuple[Array, Array]:
            scales = params[:n_classes]
            bias = params[n_classes:]
            return scales, bias

        def loss(params: Array) -> float:
            scales, bias = unpack(params)
            scaled = logits * scales + bias
            calibrated = _softmax(scaled)
            return _negative_log_likelihood(calibrated, targets, sample_weight=sample_weight)

        x0 = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
        result = minimize(loss, x0=x0, method="L-BFGS-B")
        if not result.success:  # pragma: no cover - numerical corner case
            raise RuntimeError(f"Vector scaling failed to converge: {result.message}")
        self._scales, self._bias = unpack(result.x)
        self.classes_ = classes

    def predict_proba(self, X) -> Array:
        proba = _predict_proba(self.model, X, self.clip)
        logits = np.log(proba)
        scaled = logits * self._scales + self._bias
        return _clip_probs(_softmax(scaled), self.clip)


class _MatrixScalingBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:
        proba = _predict_proba(self.model, X, self.clip)
        logits = np.log(proba)
        n_classes = logits.shape[1]
        classes = np.asarray(getattr(self.model, "classes_", np.arange(n_classes)))
        targets = _encode_labels(y, classes)

        def unpack(params: Array) -> tuple[Array, Array]:
            matrix = params[: n_classes * n_classes].reshape(n_classes, n_classes)
            bias = params[n_classes * n_classes :]
            return matrix, bias

        def loss(params: Array) -> float:
            matrix, bias = unpack(params)
            scaled = logits @ matrix + bias
            calibrated = _softmax(scaled)
            return _negative_log_likelihood(calibrated, targets, sample_weight=sample_weight)

        x0 = np.concatenate([np.eye(n_classes).ravel(), np.zeros(n_classes)])
        result = minimize(loss, x0=x0, method="L-BFGS-B")
        if not result.success:  # pragma: no cover - numerical corner case
            raise RuntimeError(f"Matrix scaling failed to converge: {result.message}")
        self._matrix, self._bias = unpack(result.x)
        self.classes_ = classes

    def predict_proba(self, X) -> Array:
        proba = _predict_proba(self.model, X, self.clip)
        logits = np.log(proba)
        scaled = logits @ self._matrix + self._bias
        return _clip_probs(_softmax(scaled), self.clip)


class _DirichletBackend(_BaseBackend):
    def fit(self, X, y, *, sample_weight=None) -> None:
        proba = _predict_proba(self.model, X, self.clip)
        features = np.concatenate([np.log(proba), np.log(1.0 - proba)], axis=1)
        classes = np.asarray(getattr(self.model, "classes_", np.arange(proba.shape[1])))
        targets = _encode_labels(y, classes)
        clf = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            solver="lbfgs",
            random_state=getattr(self.model, "random_state", None),
        )
        clf.fit(features, targets, sample_weight=sample_weight)
        self._clf = clf
        self.classes_ = classes

    def predict_proba(self, X) -> Array:
        proba = _predict_proba(self.model, X, self.clip)
        features = np.concatenate([np.log(proba), np.log(1.0 - proba)], axis=1)
        calibrated = self._clf.predict_proba(features)
        return _clip_probs(calibrated, self.clip)
