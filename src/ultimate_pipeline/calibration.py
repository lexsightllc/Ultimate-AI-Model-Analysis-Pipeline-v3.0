"""Calibration strategies with configurable backends."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


@dataclass
class Calibrator:
    method: str
    clip: float = 1e-6

    def __post_init__(self) -> None:
        self.clip = float(self.clip)

    def needs_calibration(self) -> bool:
        return self.method not in {"none", "off", None}

    def calibrate(self, model, X_val=None, y_val=None):
        method = (self.method or "none").lower()
        if method == "isotonic":
            if X_val is None or y_val is None:
                raise ValueError("Isotonic calibration requires validation features and labels.")
            return _IsotonicCalibrator(self.clip).fit(model, X_val, y_val)
        if method in {"sigmoid", "platt"}:
            if X_val is None or y_val is None:
                raise ValueError("Sigmoid calibration requires validation features and labels.")
            return _SigmoidCalibrator(self.clip).fit(model, X_val, y_val)
        if method in {"temperature", "temperature_scaling"}:
            if X_val is None or y_val is None:
                raise ValueError("Temperature scaling requires validation features and labels.")
            return _TemperatureCalibrator(self.clip).fit(model, X_val, y_val)
        return _IdentityCalibrator(self.clip).fit(model)


@dataclass
class _IdentityCalibrator:
    clip: float

    def fit(self, model):
        self._model = model
        return self

    def predict(self, X) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            if proba.ndim == 1 or proba.shape[1] == 1:
                raw = proba.ravel()
            else:
                raw = proba[:, 1]
        elif hasattr(self._model, "decision_function"):
            decision = self._model.decision_function(X)
            if decision.ndim == 1:
                raw = 1.0 / (1.0 + np.exp(-decision))
            else:
                stabilized = decision - decision.max(axis=1, keepdims=True)
                exp_scores = np.exp(stabilized)
                probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                raw = probs[:, 1] if probs.shape[1] > 1 else probs.ravel()
        else:
            raise ValueError(
                f"Model {type(self._model).__name__} does not support predict_proba or decision_function."
            )
        return np.clip(raw, self.clip, 1 - self.clip)


@dataclass
class _IsotonicCalibrator:
    clip: float

    def fit(self, model, X_val, y_val):
        raw = model.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, y_val)
        self._iso = iso
        self._model = model
        return self

    def predict(self, X) -> np.ndarray:
        raw = self._model.predict_proba(X)[:, 1]
        calibrated = self._iso.transform(raw)
        return np.clip(calibrated, self.clip, 1 - self.clip)


@dataclass
class _SigmoidCalibrator:
    clip: float

    def fit(self, model, X_val, y_val):
        calibrator = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
        calibrator.fit(X_val, y_val)
        self._calibrator = calibrator
        return self

    def predict(self, X) -> np.ndarray:
        calibrated = self._calibrator.predict_proba(X)[:, 1]
        return np.clip(calibrated, self.clip, 1 - self.clip)


@dataclass
class _TemperatureCalibrator:
    clip: float

    def fit(self, model, X_val, y_val):
        proba = _extract_positive_proba(model, X_val, self.clip)
        logits = _prob_to_logit(proba, self.clip)
        y_val = np.asarray(y_val, dtype=float)

        def loss(log_temperature: np.ndarray) -> float:
            temperature = np.exp(log_temperature[0])
            scaled = logits / temperature
            calibrated = expit(scaled)
            calibrated = np.clip(calibrated, self.clip, 1 - self.clip)
            eps = 1e-12
            calibrated = np.clip(calibrated, eps, 1 - eps)
            log_likelihood = y_val * np.log(calibrated) + (1.0 - y_val) * np.log(1.0 - calibrated)
            return float(-np.mean(log_likelihood))

        result = minimize(loss, x0=np.array([0.0]), method="L-BFGS-B")
        if not result.success:  # pragma: no cover - numerical corner case
            raise RuntimeError(f"Temperature scaling failed to converge: {result.message}")
        self._temperature = float(np.exp(result.x[0]))
        self._model = model
        return self

    def predict(self, X) -> np.ndarray:
        proba = _extract_positive_proba(self._model, X, self.clip)
        logits = _prob_to_logit(proba, self.clip)
        scaled = logits / self._temperature
        calibrated = expit(scaled)
        return np.clip(calibrated, self.clip, 1 - self.clip)


def _extract_positive_proba(model, X, clip: float) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model {type(model).__name__} must expose predict_proba for temperature scaling."
        )
    proba = model.predict_proba(X)
    if proba.ndim == 1 or proba.shape[1] == 1:
        positive = proba.ravel()
    else:
        positive = proba[:, 1]
    return np.clip(positive, clip, 1 - clip)


def _prob_to_logit(probabilities: np.ndarray, clip: float) -> np.ndarray:
    probabilities = np.clip(probabilities, clip, 1 - clip)
    odds = probabilities / (1.0 - probabilities)
    return np.log(odds)
