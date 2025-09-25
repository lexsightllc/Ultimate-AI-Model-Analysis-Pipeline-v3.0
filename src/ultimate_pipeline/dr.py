"""Dimensionality reduction strategies."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD

try:  # pragma: no cover - optional dependency
    import umap  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing library
    umap = None  # type: ignore


LOGGER = logging.getLogger(__name__)


def _ensure_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


@dataclass
class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Configurable dimensionality reduction helper.

    The implementation mirrors the techniques discussed in the accompanying
    technical review: truncated SVD (a sparse-friendly analogue of PCA),
    classical PCA for dense inputs, and optional UMAP for manifold learning.
    Each backend is selected via ``method`` and exposes a common ``transform``
    interface so it can be dropped into the broader pipeline without extra
    glue code.
    """

    method: str = "svd"
    n_components: Optional[int] = None
    explained_variance: Optional[float] = None
    random_state: int = 42
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    def __post_init__(self) -> None:
        method = (self.method or "svd").lower()
        if method not in {"svd", "truncated_svd", "pca", "umap"}:
            raise ValueError("method must be one of {'svd','truncated_svd','pca','umap'}")
        self.method = method
        if self.n_components is None and self.explained_variance is None:
            if method in {"svd", "truncated_svd", "pca"}:
                raise ValueError(
                    "Specify n_components or explained_variance for dimensionality reduction."
                )
        self._estimator = None

    # ------------------------------------------------------------------
    # Fitting utilities
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        if self.method in {"svd", "truncated_svd"}:
            self._fit_truncated_svd(X)
        elif self.method == "pca":
            self._fit_pca(X)
        elif self.method == "umap":
            self._fit_umap(X)
        else:  # pragma: no cover - unreachable due to validation
            raise RuntimeError(f"Unsupported dimensionality reduction method: {self.method}")
        return self

    def transform(self, X):
        if self._estimator is None:  # pragma: no cover - guarded by fit
            raise RuntimeError("DimensionalityReducer must be fitted before calling transform().")
        if self.method in {"svd", "truncated_svd"}:
            return self._estimator.transform(X)
        if self.method == "pca":
            dense = _ensure_dense(X)
            return self._estimator.transform(dense)
        if self.method == "umap":
            dense = _ensure_dense(X)
            return self._estimator.transform(dense)
        raise RuntimeError(f"Unsupported method: {self.method}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Backend-specific helpers
    # ------------------------------------------------------------------
    def _fit_truncated_svd(self, X) -> None:
        if not sparse.issparse(X):
            raise ValueError("Truncated SVD expects a sparse input matrix.")
        max_dim = min(X.shape) - 1
        if max_dim <= 0:
            raise ValueError("DimensionalityReducer requires at least two features for SVD.")

        requested = self.n_components or max_dim
        requested = min(requested, max_dim)

        if self.explained_variance is None:
            estimator = TruncatedSVD(n_components=requested, random_state=self.random_state)
            estimator.fit(X)
            self._estimator = estimator
            return

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
                estimator = TruncatedSVD(n_components=target_components, random_state=self.random_state)
                estimator.fit(X)
                self._estimator = estimator
            else:
                self._estimator = probe_svd
        else:
            estimator = TruncatedSVD(n_components=target_components, random_state=self.random_state)
            estimator.fit(X)
            self._estimator = estimator

    def _fit_pca(self, X) -> None:
        dense = _ensure_dense(X)
        n_components = self.n_components
        if n_components is None and self.explained_variance is not None:
            n_components = self.explained_variance
        estimator = PCA(n_components=n_components, random_state=self.random_state)
        estimator.fit(dense)
        self._estimator = estimator

    def _fit_umap(self, X) -> None:
        if umap is None:  # pragma: no cover - optional dependency not installed
            raise ImportError("umap-learn is required when using the 'umap' dimensionality reduction method.")
        dense = _ensure_dense(X)
        n_components = self.n_components or 2
        estimator = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.random_state,
        )
        estimator.fit(dense)
        self._estimator = estimator
