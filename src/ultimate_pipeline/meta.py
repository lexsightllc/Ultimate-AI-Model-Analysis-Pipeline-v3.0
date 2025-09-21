"""Meta-feature engineering components."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


META_FEATURE_COLUMNS = ("length", "punctuation", "caps_ratio", "word_count")


def _compute_meta(series: pd.Series) -> np.ndarray:
    series = series.fillna("").astype(str)
    lengths = series.str.len().to_numpy(dtype=np.float32)
    punct = series.apply(lambda x: len(re.findall(r"[^\w\s]", x))).to_numpy(dtype=np.float32)
    caps = series.apply(lambda x: len(re.findall(r"[A-Z]", x))).to_numpy(dtype=np.float32)
    letters = series.apply(lambda x: len(re.findall(r"[A-Za-z]", x))).to_numpy(dtype=np.float32)
    caps_ratio = np.divide(caps, np.maximum(letters, 1.0), out=np.zeros_like(caps), where=letters > 0)
    words = series.apply(lambda x: len(x.split())).to_numpy(dtype=np.float32)
    features = np.vstack([np.log1p(lengths), np.log1p(punct), caps_ratio, np.log1p(words)])
    return features.T


@dataclass
class MetaFeaturizer(BaseEstimator, TransformerMixin):
    """Compute and scale lightweight meta-features on text columns."""

    with_scaler: bool = True

    def __post_init__(self) -> None:
        self._scaler = StandardScaler(with_mean=False) if self.with_scaler else None

    def fit(self, texts: pd.Series, y=None):
        matrix = _compute_meta(texts)
        if self._scaler is not None:
            self._scaler.fit(matrix)
        return self

    def transform(self, texts: pd.Series):
        matrix = _compute_meta(texts)
        if self._scaler is not None:
            matrix = self._scaler.transform(matrix)
        return sparse.csr_matrix(matrix, dtype=np.float32)

    def get_feature_names(self) -> list[str]:  # pragma: no cover - simple mapping
        return list(META_FEATURE_COLUMNS)
