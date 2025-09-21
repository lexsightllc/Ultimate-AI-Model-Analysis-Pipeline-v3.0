"""Feature importance utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .vectorizers import VectorizerBase


LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    positive: List[Dict[str, float]]
    negative: List[Dict[str, float]]


def compute_linear_importance(models, word_vectorizer: VectorizerBase, char_vectorizer: VectorizerBase, top_n: int = 15) -> FeatureImportanceResult:
    coefs: List[np.ndarray] = []
    for model in models or []:
        coef = _extract_coef(model)
        if coef is not None:
            coefs.append(coef)
    if not coefs:
        LOGGER.warning(
            "Linear feature importance could not be computed because no coefficients were available.",
        )
        return FeatureImportanceResult(positive=[], negative=[])

    avg_coef = np.mean(coefs, axis=0)
    features: List[str] = []
    features.extend(_get_feature_names(word_vectorizer, "word"))
    features.extend(_get_feature_names(char_vectorizer, "character"))
    features.extend(["feat_length", "feat_punct", "feat_caps_ratio", "feat_word_count"])
    top_positive_idx = np.argsort(avg_coef)[-top_n:][::-1]
    top_negative_idx = np.argsort(avg_coef)[:top_n]
    fallback_needed = len(features) < avg_coef.shape[0]
    if fallback_needed:
        LOGGER.info(
            "Feature names were incomplete (%s provided for %s coefficients); using positional placeholders.",
            len(features),
            avg_coef.shape[0],
        )

    positive = [
        {"name": _resolve_feature_name(features, idx), "weight": float(avg_coef[idx])}
        for idx in top_positive_idx
    ]
    negative = [
        {"name": _resolve_feature_name(features, idx), "weight": float(avg_coef[idx])}
        for idx in top_negative_idx
    ]
    return FeatureImportanceResult(positive=positive, negative=negative)


def _extract_coef(model) -> Optional[np.ndarray]:
    """Extract a 1-D coefficient vector from a linear model if available."""

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            return coef
        if coef.ndim == 2:
            return coef[0]
        LOGGER.warning("Unsupported coefficient shape %s for model %s", coef.shape, type(model))
        return None

    base_model = getattr(model, "base_estimator", None)
    if base_model is None:
        base_model = getattr(model, "base_estimator_", None)
    if base_model is None:
        base_model = getattr(model, "estimator", None)
    if base_model is not None:
        return _extract_coef(base_model)

    calibrated_classifiers: Iterable = getattr(model, "calibrated_classifiers_", [])
    coefs: List[np.ndarray] = []
    for calibrated in calibrated_classifiers:
        base = getattr(calibrated, "base_estimator", None)
        if base is None:
            base = getattr(calibrated, "estimator", None)
        if base is None:
            continue
        base_coef = _extract_coef(base)
        if base_coef is not None:
            coefs.append(base_coef)
    if coefs:
        return np.mean(coefs, axis=0)

    return None


def _get_feature_names(vectorizer: VectorizerBase, description: str) -> List[str]:
    names: Optional[Iterable[str]] = None
    if hasattr(vectorizer, "get_feature_names_out"):
        names = vectorizer.get_feature_names_out()
    elif hasattr(vectorizer, "get_feature_names"):
        names = vectorizer.get_feature_names()

    if names is None:
        LOGGER.warning(
            "%s vectorizer %s does not expose feature names; positional placeholders will be used.",
            description.capitalize(),
            type(vectorizer).__name__,
        )
        return []

    return list(names)


def _resolve_feature_name(features: List[str], idx: int) -> str:
    if idx < len(features):
        return features[idx]
    return f"feature_{idx}"
