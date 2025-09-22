"""Dimensionality reduction strategies."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


LOGGER = logging.getLogger(__name__)


@dataclass
class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Wrapper around TruncatedSVD with optional explained variance target."""

    n_components: Optional[int] = None
    explained_variance: Optional[float] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_components is None and self.explained_variance is None:
            raise ValueError("Specify n_components or explained_variance for dimensionality reduction.")
        self._svd: Optional[TruncatedSVD] = None

    def fit(self, X, y=None):
        if not sparse.issparse(X):
            raise ValueError("DimensionalityReducer expects a sparse input matrix.")
        max_dim = min(X.shape) - 1
        if max_dim <= 0:
            raise ValueError("DimensionalityReducer requires at least two features for SVD.")

        requested = self.n_components or max_dim
        requested = min(requested, max_dim)

        if self.explained_variance is None:
            self._svd = TruncatedSVD(n_components=requested, random_state=self.random_state)
            self._svd.fit(X)
            return self

        probe_components = min(max_dim, max(requested, min(256, max_dim)))
        probe_svd = TruncatedSVD(n_components=probe_components, random_state=self.random_state)
        probe_svd.fit(X)

        cumulative = np.cumsum(probe_svd.explained_variance_ratio_)
        target_idx = int(np.searchsorted(cumulative, self.explained_variance, side="left"))

        if target_idx >= len(cumulative):
            LOGGER.warning(
                "Unable to reach explained variance %.3f with %s components; using %s components.",
                self.explained_variance,
                probe_components,
                probe_components,
            )
            target_components = probe_components
        else:
            target_components = max(requested, target_idx + 1)
            target_components = min(target_components, max_dim)

        if target_components <= probe_components:
            if target_components < probe_components:
                self._svd = TruncatedSVD(n_components=target_components, random_state=self.random_state)
                self._svd.fit(X)
            else:
                self._svd = probe_svd
        else:
            self._svd = TruncatedSVD(n_components=target_components, random_state=self.random_state)
            self._svd.fit(X)
        return self

    def transform(self, X):
        if self._svd is None:  # pragma: no cover - guarded by fit
            raise RuntimeError("DimensionalityReducer must be fitted before calling transform().")
        return self._svd.transform(X)
