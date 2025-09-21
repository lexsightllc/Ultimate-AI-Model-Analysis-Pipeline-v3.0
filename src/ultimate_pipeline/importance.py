"""Feature importance utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .vectorizers import VectorizerBase


@dataclass
class FeatureImportanceResult:
    positive: List[Dict[str, float]]
    negative: List[Dict[str, float]]


def compute_linear_importance(models, word_vectorizer: VectorizerBase, char_vectorizer: VectorizerBase, top_n: int = 15) -> FeatureImportanceResult:
    if not models or not hasattr(models[0], "coef_"):
        return FeatureImportanceResult(positive=[], negative=[])
    avg_coef = np.mean([model.coef_[0] for model in models], axis=0)
    features = []
    if hasattr(word_vectorizer, "get_feature_names"):
        features.extend(word_vectorizer.get_feature_names() or [])
    if hasattr(char_vectorizer, "get_feature_names"):
        features.extend(char_vectorizer.get_feature_names() or [])
    features.extend(["feat_length", "feat_punct", "feat_caps_ratio", "feat_word_count"])
    top_positive_idx = np.argsort(avg_coef)[-top_n:][::-1]
    top_negative_idx = np.argsort(avg_coef)[:top_n]
    positive = [
        {"name": features[idx], "weight": float(avg_coef[idx])}
        for idx in top_positive_idx
        if idx < len(features)
    ]
    negative = [
        {"name": features[idx], "weight": float(avg_coef[idx])}
        for idx in top_negative_idx
        if idx < len(features)
    ]
    return FeatureImportanceResult(positive=positive, negative=negative)
