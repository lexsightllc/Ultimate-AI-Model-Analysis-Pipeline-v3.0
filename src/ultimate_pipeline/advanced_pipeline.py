"""Advanced production implementation of the Ultimate AI Model Analysis Pipeline.

This module provides a self-contained pipeline that focuses on probabilistic
calibration diagnostics, manifold-aware dimensionality reduction and
competition-style submission helpers.  It intentionally mirrors the structure
outlined in the user requirements while adopting the code-quality and
robustness expectations of the existing project.

The implementation favours explicit error handling and detailed logging so that
experiments run by downstream users remain debuggable even in constrained
environments (for instance, Kaggle notebooks or CI pipelines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial.distance import pdist
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional dependency
    import umap.umap_ as umap

    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install umap-learn for full functionality.")


LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for the advanced analysis pipeline."""

    performance_mode: str = "balanced"
    calibration_enabled: bool = True
    calibration_method: str = "auto"
    calibration_cv_folds: int = 5
    epsilon_prob_clip: float = 1e-15
    ece_bins: int = 15
    auto_calibration_threshold: float = 0.05
    dimensionality_reduction: bool = True
    dr_method: str = "auto"
    n_components: int = 2
    explained_variance_threshold: float = 0.95
    min_samples_for_manifold: int = 10
    manifold_regularization: float = 0.01
    geodesic_aware: bool = True
    constraint_tolerance: float = 1e-6
    random_state: int = 42
    n_jobs: int = 1
    verbose: bool = False

    def apply_performance_mode(self) -> None:
        """Apply lightweight heuristics for preset performance profiles."""

        mode = (self.performance_mode or "balanced").lower()
        if mode == "max_speed":
            self.calibration_cv_folds = 3
            self.ece_bins = 10
            self.dr_method = "pca"
            self.n_jobs = 1
        elif mode == "best_accuracy":
            self.calibration_cv_folds = 10
            self.ece_bins = 20
            self.dr_method = "umap" if UMAP_AVAILABLE else "tsne"
            self.manifold_regularization = 0.001


class CalibrationMathematics:
    """Utility collection implementing calibration diagnostics and fitting."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.calibrator: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.calibration_metrics_: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.CalibrationMathematics")

    def expected_calibration_error(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: Optional[int] = None
    ) -> Dict[str, Any]:
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if np.any(y_prob < 0) or np.any(y_prob > 1):
            raise ValueError("y_prob must contain probabilities in [0, 1]")

        n_bins = int(n_bins or self.config.ece_bins)

        if np.allclose(y_prob, y_prob[0]):
            ece = float(abs(y_true.mean() - y_prob[0]))
            return {
                "ece": ece,
                "bin_info": [
                    {
                        "bin_lower": 0.0,
                        "bin_upper": 1.0,
                        "accuracy": float(y_true.mean()),
                        "confidence": float(y_prob[0]),
                        "proportion": 1.0,
                        "count": int(len(y_true)),
                    }
                ],
                "reliability_curve": self._compute_reliability_curve(y_true, y_prob, n_bins),
            }

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_info: List[Dict[str, Any]] = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            if bin_lower == 0.0:
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            else:
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

            prop_in_bin = float(in_bin.mean())
            if prop_in_bin > 0:
                accuracy_in_bin = float(y_true[in_bin].mean())
                avg_confidence_in_bin = float(y_prob[in_bin].mean())
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                bin_info.append(
                    {
                        "bin_lower": float(bin_lower),
                        "bin_upper": float(bin_upper),
                        "accuracy": accuracy_in_bin,
                        "confidence": avg_confidence_in_bin,
                        "proportion": prop_in_bin,
                        "count": int(in_bin.sum()),
                    }
                )
            else:
                bin_info.append(
                    {
                        "bin_lower": float(bin_lower),
                        "bin_upper": float(bin_upper),
                        "accuracy": 0.0,
                        "confidence": 0.0,
                        "proportion": 0.0,
                        "count": 0,
                    }
                )

        return {
            "ece": float(ece),
            "bin_info": bin_info,
            "reliability_curve": self._compute_reliability_curve(y_true, y_prob, n_bins),
        }

    def _compute_reliability_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Reliability curve calculation failed: %s", exc)
            return self._manual_reliability_curve(y_true, y_prob, n_bins)

    def _manual_reliability_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0

        fraction_pos: List[float] = []
        mean_pred: List[float] = []

        for idx in range(n_bins):
            mask = (y_prob >= bin_boundaries[idx]) & (y_prob < bin_boundaries[idx + 1])
            if idx == n_bins - 1:
                mask = (y_prob >= bin_boundaries[idx]) & (y_prob <= bin_boundaries[idx + 1])
            if mask.sum() > 0:
                fraction_pos.append(float(y_true[mask].mean()))
                mean_pred.append(float(y_prob[mask].mean()))
            else:
                fraction_pos.append(0.0)
                mean_pred.append(float(bin_centers[idx]))
        return np.array(fraction_pos), np.array(mean_pred)

    def brier_score_decomposition(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Brier score calculation failed: %s", exc)
            brier = float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))

        ece_result = self.expected_calibration_error(y_true, y_prob)
        base_rate = float(np.asarray(y_true).mean())
        uncertainty = base_rate * (1.0 - base_rate)
        resolution = 0.0
        reliability = 0.0

        for bin_info in ece_result["bin_info"]:
            if bin_info["count"] > 0:
                n_k = bin_info["count"] / len(y_true)
                o_k = bin_info["accuracy"]
                p_k = bin_info["confidence"]
                resolution += n_k * (o_k - base_rate) ** 2
                reliability += n_k * (p_k - o_k) ** 2

        return {
            "brier_score": brier,
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "decomposition_check": float(reliability - resolution + uncertainty),
        }

    def temperature_scaling_calibrate(
        self, logits: np.ndarray, y_true: np.ndarray, validation_split: float = 0.2
    ) -> Tuple[float, np.ndarray]:
        logits = np.asarray(logits)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)

        n_val = max(1, int(len(logits) * validation_split))
        val_logits = logits[:n_val]
        val_true = np.asarray(y_true[:n_val])

        def temperature_loss(temp: float) -> float:
            if temp <= 0:
                return 1e10
            try:
                scaled_logits = val_logits / temp
                max_logits = np.max(scaled_logits, axis=1, keepdims=True)
                exp_logits = np.exp(scaled_logits - max_logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                if probs.shape[1] == 2:
                    probs = probs[:, 1]
                elif probs.shape[1] == 1:
                    probs = probs.ravel()
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                return float(log_loss(val_true, probs))
            except Exception:
                return 1e10

        try:
            result = optimize.minimize_scalar(
                temperature_loss, bounds=(0.01, 100.0), method="bounded"
            )
            optimal_temp = float(result.x)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Temperature optimisation failed: %s", exc)
            optimal_temp = 1.0

        try:
            scaled_logits = logits / optimal_temp
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            if calibrated_probs.shape[1] == 2:
                calibrated_probs = calibrated_probs[:, 1]
            elif calibrated_probs.shape[1] == 1:
                calibrated_probs = calibrated_probs.ravel()
            calibrated_probs = np.clip(
                calibrated_probs, self.config.epsilon_prob_clip, 1 - self.config.epsilon_prob_clip
            )
            return optimal_temp, calibrated_probs
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Temperature scaling application failed: %s", exc)
            logits_flat = logits.ravel()
            calibrated = 1.0 / (1.0 + np.exp(-logits_flat))
            calibrated = np.clip(
                calibrated, self.config.epsilon_prob_clip, 1 - self.config.epsilon_prob_clip
            )
            return 1.0, calibrated

    def fit_calibrator(
        self, y_true: np.ndarray, y_prob: np.ndarray, method: Optional[str] = None
    ) -> "CalibrationMathematics":
        method = method or self.config.calibration_method
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")

        if method == "auto":
            try:
                ece_result = self.expected_calibration_error(y_true, y_prob)
                if ece_result["ece"] < self.config.auto_calibration_threshold:
                    method = "identity"
                elif len(y_true) < 1000:
                    method = "platt"
                else:
                    method = "isotonic"
            except Exception as exc:
                self.logger.warning("Auto-selection failed: %s", exc)
                method = "platt"

        try:
            if method == "identity":
                self.calibrator = lambda x: np.asarray(x)
                self.calibration_metrics_["method"] = "identity"
            elif method == "platt":
                lr = LogisticRegression(C=1.0, random_state=self.config.random_state)
                lr.fit(y_prob.reshape(-1, 1), y_true)
                self.calibrator = lambda x: lr.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
                self.calibration_metrics_["method"] = "platt"
            elif method == "isotonic":
                iso_reg = IsotonicRegression(out_of_bounds="clip")
                iso_reg.fit(y_prob, y_true)
                self.calibrator = lambda x: iso_reg.predict(np.asarray(x))
                self.calibration_metrics_["method"] = "isotonic"
            elif method == "temperature":
                eps = self.config.epsilon_prob_clip
                y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
                pseudo_logits = np.log(y_prob_clipped / (1.0 - y_prob_clipped))
                temperature, _ = self.temperature_scaling_calibrate(
                    pseudo_logits.reshape(-1, 1), y_true
                )

                def temp_calibrator(x: np.ndarray) -> np.ndarray:
                    x = np.asarray(x)
                    x_clipped = np.clip(x, eps, 1 - eps)
                    logits = np.log(x_clipped / (1.0 - x_clipped))
                    scaled = logits / temperature
                    calibrated = 1.0 / (1.0 + np.exp(-scaled))
                    return np.asarray(calibrated)

                self.calibrator = temp_calibrator
                self.calibration_metrics_["method"] = "temperature"
                self.calibration_metrics_["temperature"] = temperature
            else:
                raise ValueError(f"Unknown calibration method: {method}")
        except Exception as exc:
            self.logger.warning("Calibration fitting failed for %s: %s", method, exc)
            self.calibrator = lambda x: np.asarray(x)
            self.calibration_metrics_["method"] = "identity"
            self.calibration_metrics_["fallback"] = True

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise ValueError("Calibrator must be fitted first. Call fit_calibrator().")
        try:
            calibrated = np.asarray(self.calibrator(y_prob))
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Calibration transform failed: %s", exc)
            calibrated = np.asarray(y_prob)
        return np.clip(
            calibrated, self.config.epsilon_prob_clip, 1 - self.config.epsilon_prob_clip
        )

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_prob_original: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        try:
            ece_orig = self.expected_calibration_error(y_true, y_prob_original)
            brier_orig = self.brier_score_decomposition(y_true, y_prob_original)
            auc = (
                roc_auc_score(y_true, y_prob_original)
                if len(np.unique(y_true)) > 1
                else 0.5
            )
            results["original"] = {
                "ece": ece_orig["ece"],
                "brier_score": brier_orig["brier_score"],
                "auc": float(auc),
                "reliability_curve": ece_orig["reliability_curve"],
            }

            if y_prob_calibrated is not None:
                ece_cal = self.expected_calibration_error(y_true, y_prob_calibrated)
                brier_cal = self.brier_score_decomposition(y_true, y_prob_calibrated)
                auc_cal = (
                    roc_auc_score(y_true, y_prob_calibrated)
                    if len(np.unique(y_true)) > 1
                    else 0.5
                )
                results["calibrated"] = {
                    "ece": ece_cal["ece"],
                    "brier_score": brier_cal["brier_score"],
                    "auc": float(auc_cal),
                    "reliability_curve": ece_cal["reliability_curve"],
                }
                results["improvement"] = {
                    "ece_reduction": ece_orig["ece"] - ece_cal["ece"],
                    "brier_reduction": brier_orig["brier_score"] - brier_cal["brier_score"],
                }
        except Exception as exc:
            self.logger.error("Calibration evaluation failed: %s", exc)
            results["error"] = str(exc)
        return results

    def brier_score_decomposition(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        try:
            brier = float(brier_score_loss(y_true, y_prob))
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Brier score calculation failed: %s", exc)
            brier = float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))

        ece_result = self.expected_calibration_error(y_true, y_prob)
        base_rate = float(np.asarray(y_true).mean())
        uncertainty = base_rate * (1.0 - base_rate)
        resolution = 0.0
        reliability = 0.0

        for bin_info in ece_result["bin_info"]:
            if bin_info["count"] > 0:
                n_k = bin_info["count"] / len(y_true)
                o_k = bin_info["accuracy"]
                p_k = bin_info["confidence"]
                resolution += n_k * (o_k - base_rate) ** 2
                reliability += n_k * (p_k - o_k) ** 2

        return {
            "brier_score": brier,
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "decomposition_check": float(reliability - resolution + uncertainty),
        }


class ManifoldDimensionalityReducer:
    """Wrapper implementing PCA, t-SNE and (optionally) UMAP."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.reducer: Any = None
        self.scaler = StandardScaler()
        self.reduction_metrics_: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.ManifoldDimensionalityReducer")

    def _compute_manifold_regularization(self, X: np.ndarray, embedding: np.ndarray) -> float:
        if not self.config.geodesic_aware or X.shape[0] < 4:
            return 0.0
        try:
            n_samples = min(500, X.shape[0])
            if n_samples < X.shape[0]:
                rng = np.random.default_rng(self.config.random_state)
                idx = rng.choice(X.shape[0], n_samples, replace=False)
                X_sample = X[idx]
                emb_sample = embedding[idx]
            else:
                X_sample = X
                emb_sample = embedding

            if X_sample.shape[0] <= 1:
                return 0.0

            dist_orig = pdist(X_sample, metric="euclidean")
            dist_emb = pdist(emb_sample, metric="euclidean")
            if len(dist_orig) == 0 or len(dist_emb) == 0:
                return 0.0
            stress = float(np.mean((dist_orig - dist_emb) ** 2))
            return stress * float(self.config.manifold_regularization)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Manifold regularisation computation failed: %s", exc)
            return 0.0

    def fit_pca(self, X: np.ndarray) -> "ManifoldDimensionalityReducer":
        try:
            if self.config.n_components == "auto":  # type: ignore[comparison-overlap]
                raise ValueError("auto component selection not supported in this variant")
            n_components = min(self.config.n_components, X.shape[1], X.shape[0] - 1)
            n_components = max(1, n_components)
            self.reducer = PCA(n_components=n_components, random_state=self.config.random_state)
            X_scaled = self.scaler.fit_transform(X)
            embedding = self.reducer.fit_transform(X_scaled)
            explained = float(self.reducer.explained_variance_ratio_.sum())
            self.reduction_metrics_ = {
                "method": "pca",
                "n_components": int(n_components),
                "explained_variance_ratio": explained,
                "principal_components": self.reducer.components_,
                "manifold_stress": self._compute_manifold_regularization(X_scaled, embedding),
            }
        except Exception as exc:
            self.logger.error("PCA fitting failed: %s", exc)
            raise
        return self

    def fit_tsne(self, X: np.ndarray) -> "ManifoldDimensionalityReducer":
        if X.shape[0] < 4:
            raise ValueError(f"t-SNE requires at least 4 samples, got {X.shape[0]}")
        try:
            n_samples = X.shape[0]
            perplexity = min(30, max(1, min(n_samples - 1, n_samples // 3)))
            self.reducer = TSNE(
                n_components=min(self.config.n_components, 3),
                perplexity=perplexity,
                random_state=self.config.random_state,
                n_jobs=1,
                init="pca",
            )
            X_scaled = self.scaler.fit_transform(X)
            embedding = self.reducer.fit_transform(X_scaled)
            self.reduction_metrics_ = {
                "method": "tsne",
                "perplexity": float(perplexity),
                "kl_divergence": float(getattr(self.reducer, "kl_divergence_", np.nan)),
                "manifold_stress": self._compute_manifold_regularization(X_scaled, embedding),
            }
        except Exception as exc:
            self.logger.error("t-SNE fitting failed: %s", exc)
            raise
        return self

    def fit_umap(self, X: np.ndarray) -> "ManifoldDimensionalityReducer":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install umap-learn.")
        if X.shape[0] < 4:
            raise ValueError(f"UMAP requires at least 4 samples, got {X.shape[0]}")
        try:
            n_samples, n_features = X.shape
            n_neighbors = max(2, min(n_samples - 1, min(15, int(np.sqrt(n_samples)))))
            reducer = umap.UMAP(
                n_components=min(self.config.n_components, n_features),
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="euclidean",
                random_state=self.config.random_state,
                n_jobs=1,
                verbose=False,
            )
            X_scaled = self.scaler.fit_transform(X)
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                raise ValueError("Input contains NaN or infinite values after scaling")
            if np.any(np.var(X_scaled, axis=0) == 0):
                self.logger.warning(
                    "Some features have zero variance, this may affect UMAP performance"
                )
            embedding = reducer.fit_transform(X_scaled)
            self.reducer = reducer
            self.reduction_metrics_ = {
                "method": "umap",
                "n_neighbors": int(n_neighbors),
                "manifold_stress": self._compute_manifold_regularization(X_scaled, embedding),
            }
        except Exception as exc:
            self.logger.error("UMAP fitting failed: %s", exc)
            raise
        return self
    def fit(self, X: np.ndarray, method: Optional[str] = None) -> "ManifoldDimensionalityReducer":
        method = (method or self.config.dr_method or "pca").lower()
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples for dimensionality reduction, got {n_samples}")
        if n_features == 0:
            raise ValueError("Input has no features")

        if method == "auto":
            if n_samples < self.config.min_samples_for_manifold:
                method = "pca"
            elif n_features < 10 and n_samples < 1000:
                method = "pca"
            elif n_samples < 1000:
                method = "tsne" if n_samples >= 10 else "pca"
            elif UMAP_AVAILABLE and n_samples >= 10:
                method = "umap"
            else:
                method = "pca"

        methods_to_try = [method]
        if method != "pca":
            methods_to_try.append("pca")

        last_error: Optional[Exception] = None
        for attempt in methods_to_try:
            try:
                if attempt == "pca":
                    return self.fit_pca(X)
                if attempt == "tsne":
                    return self.fit_tsne(X)
                if attempt == "umap":
                    return self.fit_umap(X)
                raise ValueError(f"Unknown dimensionality reduction method: {attempt}")
            except Exception as exc:
                last_error = exc
                self.logger.warning("Method %s failed: %s", attempt, exc)
                continue

        raise RuntimeError(f"All dimensionality reduction methods failed. Last error: {last_error}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reducer is None:
            raise ValueError("Reducer must be fitted first. Call fit().")
        X_scaled = self.scaler.transform(X)
        if hasattr(self.reducer, "transform"):
            return self.reducer.transform(X_scaled)
        self.logger.warning(
            "Method %s doesn't support transform. Refitting.",
            self.reduction_metrics_.get("method", "unknown"),
        )
        fitted = self.fit(X)
        reducer = fitted.reducer
        if hasattr(reducer, "embedding_"):
            return reducer.embedding_
        raise RuntimeError("Reducer does not expose a usable embedding after refit")

    def fit_transform(self, X: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        self.fit(X, method)
        if hasattr(self.reducer, "embedding_"):
            return self.reducer.embedding_
        return self.transform(X)


class UltimateModelAnalysisPipeline:
    """Main orchestration connecting calibration and dimensionality reduction."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.config.apply_performance_mode()
        self.calibrator = CalibrationMathematics(self.config)
        self.dim_reducer = ManifoldDimensionalityReducer(self.config)
        self.results_: Dict[str, Any] = {}
        self.visualizations_: Dict[str, plt.Figure] = {}
        self.logger = logging.getLogger(f"{__name__}.UltimateModelAnalysisPipeline")
        if self.config.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def analyze_classification_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        X_features: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {"model_name": model_name}
        try:
            y_true = np.asarray(y_true)
            y_prob = np.asarray(y_prob)
            if len(y_true) != len(y_prob):
                raise ValueError("y_true and y_prob must have same length")

            if self.config.calibration_enabled:
                try:
                    self.calibrator.fit_calibrator(y_true, y_prob)
                    calibrated = self.calibrator.transform(y_prob)
                    calibration_results = self.calibrator.evaluate_calibration(
                        y_true, y_prob, calibrated
                    )
                    results["calibration"] = calibration_results
                    results["calibrated_probabilities"] = calibrated
                except Exception as exc:
                    self.logger.error("Calibration analysis failed for %s: %s", model_name, exc)
                    results["calibration_error"] = str(exc)

            if self.config.dimensionality_reduction and X_features is not None:
                try:
                    X_features = np.asarray(X_features)
                    analysis_matrix = np.column_stack(
                        [X_features, y_prob.reshape(-1, 1), y_true.reshape(-1, 1)]
                    )
                    embedding = self.dim_reducer.fit_transform(analysis_matrix)
                    results["dimensionality_reduction"] = {
                        "embedding": embedding,
                        "metrics": self.dim_reducer.reduction_metrics_,
                    }
                except Exception as exc:
                    self.logger.error("Dimensionality reduction failed for %s: %s", model_name, exc)
                    results["dimensionality_reduction_error"] = str(exc)

            self.results_[model_name] = results
        except Exception as exc:
            self.logger.error("Model analysis failed for %s: %s", model_name, exc)
            results["analysis_error"] = str(exc)
        return results

    def compare_models(self, models_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        comparison_results: Dict[str, Any] = {}
        for model_name, data in models_data.items():
            try:
                comparison_results[model_name] = self.analyze_classification_model(
                    data["y_true"], data["y_prob"], data.get("X_features"), model_name
                )
            except Exception as exc:
                self.logger.error("Model %s analysis failed: %s", model_name, exc)
                comparison_results[model_name] = {"error": str(exc)}

        if len(models_data) > 1:
            try:
                comparison_matrix = self._create_comparison_matrix(comparison_results)
                if comparison_matrix is not None and comparison_matrix.shape[0] >= 3:
                    if self.config.dimensionality_reduction:
                        try:
                            meta_reducer = ManifoldDimensionalityReducer(self.config)
                            meta_embedding = meta_reducer.fit_transform(comparison_matrix)
                            comparison_results["meta_analysis"] = {
                                "comparison_matrix": comparison_matrix,
                                "model_embedding": meta_embedding,
                                "reducer_metrics": meta_reducer.reduction_metrics_,
                            }
                        except Exception as exc:
                            self.logger.warning(
                                "Meta-analysis dimensionality reduction failed: %s", exc
                            )
                            comparison_results["meta_analysis"] = {
                                "comparison_matrix": comparison_matrix,
                                "note": f"Dimensionality reduction failed: {exc}",
                            }
                    else:
                        comparison_results["meta_analysis"] = {
                            "comparison_matrix": comparison_matrix,
                            "note": "Dimensionality reduction disabled",
                        }
                else:
                    shape_info = (
                        None if comparison_matrix is None else comparison_matrix.shape
                    )
                    comparison_results["meta_analysis"] = {
                        "note": f"Insufficient data for meta-analysis (shape: {shape_info})"
                    }
            except Exception as exc:
                self.logger.error("Meta-analysis failed: %s", exc)
                comparison_results["meta_analysis"] = {
                    "error": str(exc),
                    "note": "Meta-analysis could not be completed",
                }
        return comparison_results

    def _create_comparison_matrix(self, model_results: Dict[str, Any]) -> Optional[np.ndarray]:
        features: List[List[float]] = []
        for model_name, results in model_results.items():
            if "calibration" not in results or "error" in results:
                continue
            cal_metrics = results["calibration"]
            if "original" not in cal_metrics:
                continue
            feature_vector = [
                float(cal_metrics["original"].get("ece", 0.0)),
                float(cal_metrics["original"].get("brier_score", 0.0)),
                float(cal_metrics["original"].get("auc", 0.5)),
            ]
            if "calibrated" in cal_metrics:
                feature_vector.extend(
                    [
                        float(cal_metrics["calibrated"].get("ece", 0.0)),
                        float(cal_metrics["calibrated"].get("brier_score", 0.0)),
                        float(cal_metrics["improvement"].get("ece_reduction", 0.0)),
                    ]
                )
            else:
                feature_vector.extend([0.0, 0.0, 0.0])
            if all(np.isfinite(feature_vector)):
                features.append(feature_vector)
            else:  # pragma: no cover
                self.logger.warning("Skipping %s due to non-finite metrics", model_name)
        if len(features) >= 2:
            return np.array(features)
        self.logger.warning("Insufficient valid model results for comparison matrix: %d", len(features))
        return None

    def generate_calibration_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
        title: str = "Calibration Plot",
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)

        ax = axes[0, 0]
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=self.config.ece_bins, strategy="uniform"
            )
            ax.plot(mean_pred, fraction_pos, "s-", label="Original Model", markersize=6)
            if y_prob_calibrated is not None:
                fraction_pos_cal, mean_pred_cal = calibration_curve(
                    y_true, y_prob_calibrated, n_bins=self.config.ece_bins, strategy="uniform"
                )
                ax.plot(
                    mean_pred_cal,
                    fraction_pos_cal,
                    "o-",
                    label="Calibrated Model",
                    markersize=6,
                )
            ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Reliability Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Reliability curve failed:\n{exc}", ha="center", va="center", transform=ax.transAxes)

        ax = axes[0, 1]
        try:
            ax.hist(y_prob, bins=20, alpha=0.7, label="Original", density=True, edgecolor="black")
            if y_prob_calibrated is not None:
                ax.hist(
                    y_prob_calibrated,
                    bins=20,
                    alpha=0.7,
                    label="Calibrated",
                    density=True,
                    edgecolor="black",
                )
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Density")
            ax.set_title("Prediction Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Histogram failed:\n{exc}", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 0]
        try:
            ece_result = self.calibrator.expected_calibration_error(y_true, y_prob)
            bin_centers = [
                (bin_info["bin_lower"] + bin_info["bin_upper"]) / 2.0 for bin_info in ece_result["bin_info"]
            ]
            bin_errors = [abs(info["confidence"] - info["accuracy"]) for info in ece_result["bin_info"]]
            ax.bar(bin_centers, bin_errors, width=0.08, alpha=0.7, label="Original", edgecolor="black")
            if y_prob_calibrated is not None:
                ece_result_cal = self.calibrator.expected_calibration_error(y_true, y_prob_calibrated)
                bin_errors_cal = [
                    abs(info["confidence"] - info["accuracy"]) for info in ece_result_cal["bin_info"]
                ]
                ax.bar(
                    [center + 0.04 for center in bin_centers],
                    bin_errors_cal,
                    width=0.08,
                    alpha=0.7,
                    label="Calibrated",
                    edgecolor="black",
                )
            ax.set_xlabel("Prediction Bin")
            ax.set_ylabel("|Confidence - Accuracy|")
            ax.set_title("Expected Calibration Error by Bin")
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"ECE plot failed:\n{exc}", ha="center", va="center", transform=ax.transAxes)

        ax = axes[1, 1]
        try:
            ece_result = self.calibrator.expected_calibration_error(y_true, y_prob)
            for idx, bin_info in enumerate(ece_result["bin_info"]):
                if bin_info["count"] > 0:
                    ax.scatter(
                        bin_info["confidence"],
                        bin_info["accuracy"],
                        s=max(10, bin_info["count"]),
                        alpha=0.6,
                        label="Original" if idx == 0 else "",
                    )
            if y_prob_calibrated is not None:
                ece_result_cal = self.calibrator.expected_calibration_error(y_true, y_prob_calibrated)
                for idx, bin_info in enumerate(ece_result_cal["bin_info"]):
                    if bin_info["count"] > 0:
                        ax.scatter(
                            bin_info["confidence"],
                            bin_info["accuracy"],
                            s=max(10, bin_info["count"]),
                            alpha=0.6,
                            marker="^",
                            label="Calibrated" if idx == 0 else "",
                        )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("Mean Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title("Confidence vs Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Scatter plot failed:\n{exc}", ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        return fig

    def generate_manifold_plot(
        self, embedding: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "Manifold Visualization"
    ) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        try:
            if labels is not None and len(labels) == len(embedding):
                scatter = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=labels,
                    cmap="viridis",
                    alpha=0.6,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    alpha=0.6,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )
            ax.set_title(title)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            self.logger.error("Manifold plot generation failed: %s", exc)
            ax.text(0.5, 0.5, f"Manifold plot failed:\n{exc}", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        return fig

    def generate_submission(
        self,
        test_features: np.ndarray,
        model_predictions: np.ndarray,
        test_ids: Optional[np.ndarray] = None,
        submission_path: str = "submission.csv",
        id_column: str = "id",
        target_column: str = "prediction",
    ) -> pd.DataFrame:
        test_features = np.asarray(test_features)
        model_predictions = np.asarray(model_predictions)
        if len(test_features) != len(model_predictions):
            raise ValueError("test_features and model_predictions must have same length")

        if test_ids is None:
            test_ids = np.arange(len(test_features))
        else:
            test_ids = np.asarray(test_ids)

        if self.calibrator.calibrator is not None:
            if self.config.verbose:
                print("Applying calibration to test predictions...")
            calibrated_predictions = self.calibrator.transform(model_predictions)
        else:
            if self.config.verbose:
                print("Warning: No calibrator fitted, using raw predictions")
            calibrated_predictions = np.asarray(model_predictions)

        submission_df = pd.DataFrame(
            {id_column: test_ids, target_column: calibrated_predictions}, columns=[id_column, target_column]
        )
        submission_df.to_csv(submission_path, index=False)
        if self.config.verbose:
            print(f"Submission saved to {submission_path}")
            print(f"Shape: {submission_df.shape}")
            print(
                f"Prediction range: [{calibrated_predictions.min():.4f}, {calibrated_predictions.max():.4f}]"
            )
        return submission_df

    def fit_and_predict(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        train_predictions: np.ndarray,
        test_features: np.ndarray,
        test_predictions: np.ndarray,
        test_ids: Optional[np.ndarray] = None,
        model_name: str = "competition_model",
    ) -> Dict[str, Any]:
        analysis_results = self.analyze_classification_model(
            train_targets, train_predictions, train_features, model_name
        )
        submission_df = self.generate_submission(
            test_features,
            test_predictions,
            test_ids,
            f"{model_name}_submission.csv",
        )
        return {
            "analysis": analysis_results,
            "submission": submission_df,
            "calibration_applied": self.calibrator.calibrator is not None,
            "calibration_method": self.calibrator.calibration_metrics_.get("method", "none"),
        }

    def export_results(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        export_data = {
            "config": self.config.__dict__,
            "results": self.results_,
            "timestamp": pd.Timestamp.now().isoformat(),
            "pipeline_version": "3.0",
        }
        if filepath:
            import json

            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(export_data, handle, indent=2, default=str)
        return export_data


def optimize_calibration_parameters(
    y_true: np.ndarray, y_prob: np.ndarray, method: str = "grid_search"
) -> Dict[str, Any]:
    try:
        if method != "grid_search":
            raise ValueError("Only grid_search method is implemented")
        methods = ["platt", "isotonic"]
        best_score = float("inf")
        best_params: Dict[str, Any] = {
            "method": "identity",
            "cross_val_ece": float("inf"),
            "cross_val_std": 0.0,
        }
        for cal_method in methods:
            try:
                config = AnalysisConfig(calibration_method=cal_method)
                config.apply_performance_mode()
                calibrator = CalibrationMathematics(config)
                scores: List[float] = []
                unique_labels = np.unique(y_true)
                n_splits = min(5, len(unique_labels))
                if n_splits < 2:
                    continue
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                for train_idx, test_idx in skf.split(y_prob.reshape(-1, 1), y_true):
                    try:
                        calibrator.fit_calibrator(y_true[train_idx], y_prob[train_idx])
                        calibrated = calibrator.transform(y_prob[test_idx])
                        ece_result = calibrator.expected_calibration_error(
                            y_true[test_idx], calibrated
                        )
                        scores.append(float(ece_result["ece"]))
                    except Exception as fold_exc:
                        LOGGER.warning(
                            "Fold evaluation failed for %s: %s", cal_method, fold_exc
                        )
                        continue
                if scores:
                    avg_score = float(np.mean(scores))
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = {
                            "method": cal_method,
                            "cross_val_ece": avg_score,
                            "cross_val_std": float(np.std(scores)),
                        }
            except Exception as method_exc:
                LOGGER.warning("Method %s optimization failed: %s", cal_method, method_exc)
                continue
        return best_params
    except Exception as exc:
        LOGGER.error("Calibration parameter optimization failed: %s", exc)
        return {
            "method": "identity",
            "cross_val_ece": float("inf"),
            "cross_val_std": 0.0,
            "error": str(exc),
        }


def create_synthetic_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    miscalibration_factor: float = 2.0,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    else:  # pragma: no cover
        np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    logits = X[:, 0] + 0.5 * X[:, 1]
    true_probs = 1.0 / (1.0 + np.exp(-logits))
    y_true = np.random.binomial(1, true_probs)
    raw_logits = X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    y_prob_raw = 1.0 / (1.0 + np.exp(-raw_logits))
    if miscalibration_factor != 1.0:
        y_prob = np.power(y_prob_raw, miscalibration_factor)
        y_prob = y_prob / (y_prob + np.power(1.0 - y_prob_raw, miscalibration_factor))
    else:
        y_prob = y_prob_raw
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "X": X,
        "y_true": y_true,
        "y_prob": y_prob,
        "true_probs": true_probs,
    }


def demonstrate_pipeline_comprehensive() -> Tuple[
    Optional[UltimateModelAnalysisPipeline],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    print("Ultimate AI Model Analysis Pipeline v3.0 - Comprehensive Demonstration")
    print("=" * 70)
    try:
        print("\nGenerating synthetic datasets...")
        data1 = create_synthetic_classification_data(
            n_samples=1500, miscalibration_factor=1.8, random_state=42
        )
        data2 = create_synthetic_classification_data(
            n_samples=1200, miscalibration_factor=0.9, random_state=43
        )
        data3 = create_synthetic_classification_data(
            n_samples=1800, miscalibration_factor=2.2, random_state=44
        )

        config = AnalysisConfig(
            performance_mode="balanced",
            calibration_enabled=True,
            dimensionality_reduction=True,
            verbose=True,
        )
        pipeline = UltimateModelAnalysisPipeline(config)

        print("\n1. Single Model Analysis")
        print("-" * 35)
        results = pipeline.analyze_classification_model(
            data1["y_true"], data1["y_prob"], data1["X"], "Primary_Model"
        )
        if "calibration" in results and "original" in results["calibration"] and "calibrated" in results["calibration"]:
            cal = results["calibration"]
            print(f"Original ECE: {cal['original']['ece']:.4f}")
            print(f"Calibrated ECE: {cal['calibrated']['ece']:.4f}")
            print(f"ECE Improvement: {cal['improvement']['ece_reduction']:+.4f}")
        else:
            print("Calibration analysis incomplete")

        print("\n2. Multi-Model Comparison")
        print("-" * 30)
        models_data = {
            "Conservative_Model": {
                "y_true": data1["y_true"],
                "y_prob": data1["y_prob"],
                "X_features": data1["X"],
            },
            "Balanced_Model": {
                "y_true": data2["y_true"],
                "y_prob": data2["y_prob"],
                "X_features": data2["X"],
            },
            "Aggressive_Model": {
                "y_true": data3["y_true"],
                "y_prob": data3["y_prob"],
                "X_features": data3["X"],
            },
        }
        comparison_results = pipeline.compare_models(models_data)
        print("\nModel Comparison Results:")
        for model_name, model_results in comparison_results.items():
            if model_name == "meta_analysis":
                continue
            if "calibration" in model_results and "original" in model_results["calibration"]:
                cal = model_results["calibration"]
                improvement = cal.get("improvement", {}).get("ece_reduction", 0.0)
                print(
                    f"{model_name:18} | ECE: {cal['original']['ece']:.4f} | Improvement: {improvement:+.4f}"
                )
            elif "error" in model_results:
                print(f"{model_name:18} | Error: {model_results['error']}")

        if "meta_analysis" in comparison_results:
            meta = comparison_results["meta_analysis"]
            if "model_embedding" in meta:
                print("\nMeta-analysis completed successfully:")
                print(f"  - Comparison matrix shape: {meta['comparison_matrix'].shape}")
                print(f"  - Embedding method: {meta['reducer_metrics']['method']}")
                if "manifold_stress" in meta["reducer_metrics"]:
                    print(f"  - Manifold stress: {meta['reducer_metrics']['manifold_stress']:.6f}")
            elif "note" in meta:
                print(f"\nMeta-analysis note: {meta['note']}")
            elif "error" in meta:
                print(f"\nMeta-analysis error: {meta['error']}")

        print("\n3. Visualization Generation")
        print("-" * 30)
        try:
            cal_fig = pipeline.generate_calibration_plot(
                data1["y_true"],
                data1["y_prob"],
                results.get("calibrated_probabilities"),
                "Comprehensive Model Calibration",
            )
            print("✓ Calibration plot generated")
            pipeline.visualizations_["calibration"] = cal_fig
            if "dimensionality_reduction" in results:
                manifold_fig = pipeline.generate_manifold_plot(
                    results["dimensionality_reduction"]["embedding"],
                    data1["y_true"],
                    "Feature Space Manifold",
                )
                pipeline.visualizations_["manifold"] = manifold_fig
                print("✓ Manifold plot generated")
            else:
                print("⚠ Manifold plot skipped (no embedding data)")
        except Exception as exc:  # pragma: no cover
            print(f"✗ Visualization generation failed: {exc}")

        print("\n4. Calibration Parameter Optimization")
        print("-" * 45)
        try:
            best_params = optimize_calibration_parameters(data1["y_true"], data1["y_prob"])
            if "error" not in best_params:
                print(f"Optimal calibration method: {best_params['method']}")
                print(
                    "Cross-validated ECE: "
                    f"{best_params['cross_val_ece']:.4f} ± {best_params['cross_val_std']:.4f}"
                )
            else:
                print(f"Optimization failed: {best_params['error']}")
        except Exception as exc:
            print(f"Calibration optimization error: {exc}")

        print("\n5. Edge Case Testing")
        print("-" * 25)
        try:
            tiny_data = create_synthetic_classification_data(
                n_samples=8, n_features=3, random_state=45
            )
            tiny_pipeline = UltimateModelAnalysisPipeline(AnalysisConfig(dr_method="auto"))
            tiny_results = tiny_pipeline.analyze_classification_model(
                tiny_data["y_true"], tiny_data["y_prob"], tiny_data["X"], "Tiny_Model"
            )
            dr_method = "None"
            if "dimensionality_reduction" in tiny_results:
                dr_method = tiny_results["dimensionality_reduction"].get("metrics", {}).get(
                    "method", "Failed"
                )
            print(f"✓ Tiny dataset (n=8) handled successfully - DR method: {dr_method}")
        except Exception as exc:
            print(f"✗ Tiny dataset test failed: {exc}")

        print("\n6. Results Export")
        print("-" * 20)
        try:
            export_data = pipeline.export_results()
            if "error" not in export_data:
                print(f"✓ Results exported successfully ({len(export_data)} keys)")
            else:
                print(f"✗ Export failed: {export_data['error']}")
        except Exception as exc:
            print(f"✗ Export error: {exc}")

        print("\n" + "=" * 70)
        print("Comprehensive pipeline demonstration completed!")
        print("Pipeline demonstrates robust error handling and mathematical correctness.")
        return pipeline, results, comparison_results
    except Exception as exc:
        LOGGER.error("Demonstration failed: %s", exc)
        print(f"Demonstration failed with error: {exc}")
        return None, None, None


def demonstrate_competition_submission() -> Tuple[
    Optional[UltimateModelAnalysisPipeline],
    Optional[Dict[str, Any]],
]:
    print("Ultimate AI Model Analysis Pipeline v3.0 - Competition Submission Demo")
    print("=" * 75)
    try:
        print("\nGenerating synthetic competition datasets...")
        train_data = create_synthetic_classification_data(
            n_samples=2000, n_features=15, miscalibration_factor=1.8, random_state=42
        )
        test_data = create_synthetic_classification_data(
            n_samples=800, n_features=15, miscalibration_factor=1.8, random_state=123
        )
        test_ids = np.arange(len(test_data["X"]))
        print(
            f"Training set: {train_data['X'].shape[0]} samples, {train_data['X'].shape[1]} features"
        )
        print(
            f"Test set: {test_data['X'].shape[0]} samples, {test_data['X'].shape[1]} features"
        )

        config = AnalysisConfig(
            performance_mode="best_accuracy",
            calibration_enabled=True,
            dimensionality_reduction=False,
            verbose=True,
        )
        pipeline = UltimateModelAnalysisPipeline(config)

        print("\n1. Running Complete Competition Pipeline")
        print("-" * 45)
        competition_results = pipeline.fit_and_predict(
            train_features=train_data["X"],
            train_targets=train_data["y_true"],
            train_predictions=train_data["y_prob"],
            test_features=test_data["X"],
            test_predictions=test_data["y_prob"],
            test_ids=test_ids,
            model_name="competition_model",
        )

        if (
            "analysis" in competition_results
            and "calibration" in competition_results["analysis"]
            and "original" in competition_results["analysis"]["calibration"]
        ):
            cal = competition_results["analysis"]["calibration"]
            improvement = cal.get("improvement", {}).get("ece_reduction", 0.0)
            print(f"Training ECE: {cal['original']['ece']:.4f}")
            print(f"ECE Improvement: {improvement:+.4f}")
            print(f"Calibration method: {competition_results['calibration_method']}")

        if "submission" in competition_results:
            submission_df = competition_results["submission"]
            print(f"Submission shape: {submission_df.shape}")
            print("Prediction statistics:")
            print(f"  Mean: {submission_df['prediction'].mean():.4f}")
            print(f"  Std:  {submission_df['prediction'].std():.4f}")
            print(f"  Min:  {submission_df['prediction'].min():.4f}")
            print(f"  Max:  {submission_df['prediction'].max():.4f}")
            print("First 5 rows of submission:")
            print(submission_df.head())

        print("\n2. Direct Submission Generation")
        print("-" * 35)
        pipeline.calibrator.fit_calibrator(train_data["y_true"], train_data["y_prob"])
        submission_df_custom = pipeline.generate_submission(
            test_features=test_data["X"],
            model_predictions=test_data["y_prob"],
            test_ids=test_ids,
            submission_path="custom_submission.csv",
            id_column="row_id",
            target_column="target",
        )
        print(f"Custom submission saved with columns: {list(submission_df_custom.columns)}")

        print("\n3. Submission Validation")
        print("-" * 25)
        import os

        required_files = [
            "competition_model_submission.csv",
            "custom_submission.csv",
        ]
        for filename in required_files:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                print(f"✓ {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                if df.isnull().sum().sum() == 0:
                    print("  ✓ No missing values")
                else:
                    print(f"  ✗ Contains {df.isnull().sum().sum()} missing values")
                pred_col = df.columns[1]
                if df[pred_col].min() >= 0 and df[pred_col].max() <= 1:
                    print("  ✓ Predictions in valid range [0, 1]")
                else:
                    print(
                        f"  ⚠ Predictions outside [0, 1]: [{df[pred_col].min():.4f}, {df[pred_col].max():.4f}]"
                    )
            else:
                print(f"✗ {filename}: File not found")

        default_submission = pipeline.generate_submission(
            test_features=test_data["X"],
            model_predictions=test_data["y_prob"],
            test_ids=test_ids,
            submission_path="submission.csv",
            id_column="id",
            target_column="prediction",
        )
        print(f"\n✓ Default submission.csv created: {default_submission.shape}")

        print("\n4. Calibration Impact Analysis")
        print("-" * 32)
        val_size = 200
        val_indices = np.random.choice(len(train_data["y_true"]), val_size, replace=False)
        val_true = train_data["y_true"][val_indices]
        val_pred_raw = train_data["y_prob"][val_indices]
        val_pred_calibrated = pipeline.calibrator.transform(val_pred_raw)
        raw_logloss = log_loss(val_true, val_pred_raw)
        cal_logloss = log_loss(val_true, val_pred_calibrated)
        raw_auc = roc_auc_score(val_true, val_pred_raw)
        cal_auc = roc_auc_score(val_true, val_pred_calibrated)
        print("Validation Metrics Comparison:")
        print(f"  Log Loss:  {raw_logloss:.4f} → {cal_logloss:.4f} (Δ: {cal_logloss - raw_logloss:+.4f})")
        print(f"  AUC Score: {raw_auc:.4f} → {cal_auc:.4f} (Δ: {cal_auc - raw_auc:+.4f})")

        print("\n" + "=" * 75)
        print("Competition submission pipeline completed successfully!")
        print("Files created: submission.csv, competition_model_submission.csv, custom_submission.csv")
        return pipeline, competition_results
    except Exception as exc:
        LOGGER.error("Competition demonstration failed: %s", exc)
        print(f"Competition demo failed: {exc}")
        return None, None
