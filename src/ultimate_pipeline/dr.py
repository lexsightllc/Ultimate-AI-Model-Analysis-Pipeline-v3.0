"""Dimensionality reduction strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


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
        n_components = self.n_components
        if self.explained_variance is not None:
            max_dim = min(X.shape) - 1
            n_components = min(max_dim, n_components or max_dim)
        self._svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state,
        )
        self._svd.fit(X)
        if self.explained_variance is not None:
            cumulative = self._svd.explained_variance_ratio_.sum()
            if cumulative < self.explained_variance:
                # Increase components if variance target not met
                additional = int(self.explained_variance / max(cumulative, 1e-6) * self._svd.n_components)
                additional = min(additional, X.shape[1] - 1)
                if additional > self._svd.n_components:
                    self._svd = TruncatedSVD(
                        n_components=additional,
                        random_state=self.random_state,
                    )
                    self._svd.fit(X)
        return self

    def transform(self, X):
        if self._svd is None:  # pragma: no cover - guarded by fit
            raise RuntimeError("DimensionalityReducer must be fitted before calling transform().")
        return self._svd.transform(X)
