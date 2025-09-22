"""Calibration strategies with configurable backends."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
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
