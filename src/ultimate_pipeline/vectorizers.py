"""Vectorizer abstractions that expose a unified interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer


class VectorizerBase(BaseEstimator, TransformerMixin):
    """Base class implementing the sklearn transformer protocol."""

    def fit(self, X, y=None):  # pragma: no cover - delegated
        raise NotImplementedError

    def transform(self, X):  # pragma: no cover - delegated
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> Optional[list[str]]:  # pragma: no cover - optional
        return None


@dataclass
class TfidfVectorizerWrapper(VectorizerBase):
    """Standard scikit-learn TF–IDF vectorizer."""

    params: Dict[str, Any]
    max_features: Optional[int] = None

    def __post_init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=self.max_features, **self.params)

    def fit(self, X, y=None):
        self._vectorizer.fit(X)
        return self

    def transform(self, X):
        return self._vectorizer.transform(X)

    def get_feature_names(self) -> Optional[list[str]]:
        return list(self._vectorizer.get_feature_names_out())


@dataclass
class HashingWithIDFVectorizer(VectorizerBase):
    """Hashing vectorizer with post-hoc IDF re-weighting."""

    n_features: int = 2 ** 20
    alternate_sign: bool = False
    norm: Optional[str] = "l2"

    def __post_init__(self) -> None:
        self._hashing = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=self.alternate_sign,
            norm=None,
        )
        self._idf = TfidfTransformer(norm=self.norm)

    def fit(self, X, y=None):
        matrix = self._hashing.transform(X)
        self._idf.fit(matrix)
        return self

    def transform(self, X):
        matrix = self._hashing.transform(X)
        return self._idf.transform(matrix)


@dataclass
class CuMLTfidfVectorizer(VectorizerBase):
    """GPU-accelerated TF–IDF using RAPIDS cuML when available."""

    params: Dict[str, Any]
    max_features: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            import cudf  # type: ignore
            from cuml.feature_extraction.text import TfidfVectorizer as CuMLTfidf  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("cuML is not available. Install RAPIDS or choose another vectorizer.") from exc
        self._cudf = cudf
        self._vectorizer = CuMLTfidf(max_features=self.max_features, **self.params)

    def fit(self, X, y=None):
        series = self._cudf.Series(X)
        self._vectorizer.fit(series)
        return self

    def transform(self, X):
        series = self._cudf.Series(X)
        return self._vectorizer.transform(series)


def make_vectorizer(name: str, params: Dict[str, Any], max_features: Optional[int]) -> VectorizerBase:
    normalized = (name or "tfidf").lower()
    if normalized == "tfidf":
        return TfidfVectorizerWrapper(params=params, max_features=max_features)
    if normalized in {"hashing", "hashing_tfidf"}:
        n_features = max_features or 2 ** 20
        return HashingWithIDFVectorizer(n_features=n_features)
    if normalized == "cuml":  # pragma: no branch
        return CuMLTfidfVectorizer(params=params, max_features=max_features)
    raise ValueError(f"Unknown vectorizer '{name}'.")
