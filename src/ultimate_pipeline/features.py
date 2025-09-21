"""Feature assembly pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from joblib import Memory
from scipy import sparse

from .config import AnalysisConfig
from .dr import DimensionalityReducer
from .meta import MetaFeaturizer
from .text import TextPreprocessor
from .vectorizers import VectorizerBase, make_vectorizer


def _fit_transform_vectorizer(vectorizer: VectorizerBase, texts):
    return vectorizer.fit_transform(texts)


def _transform_vectorizer(vectorizer: VectorizerBase, texts):
    return vectorizer.transform(texts)


def _to_dense_if_needed(matrix):  # pragma: no cover - defensive
    if sparse.issparse(matrix):
        return matrix
    return sparse.csr_matrix(matrix)


@dataclass
class FeatureAssembler:
    config: AnalysisConfig
    text_preprocessor: TextPreprocessor
    cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        word_params = {**self.config.tfidf_word_params, "dtype": np.float32}
        if "ngram_range" in word_params:
            word_params["ngram_range"] = tuple(word_params["ngram_range"])
        char_params = {**self.config.tfidf_char_params, "dtype": np.float32}
        if "ngram_range" in char_params:
            char_params["ngram_range"] = tuple(char_params["ngram_range"])
        self.word_vectorizer = make_vectorizer(
            self.config.vectorizer,
            word_params,
            self.config.max_tfidf_features,
        )
        self.char_vectorizer = make_vectorizer(
            self.config.vectorizer,
            char_params,
            self.config.max_tfidf_features,
        )
        self.meta_featurizer = MetaFeaturizer()
        self.reducer: Optional[DimensionalityReducer] = None
        if self.config.dimensionality_reduction:
            self.reducer = DimensionalityReducer(
                n_components=self.config.dimensionality_reduction_components,
                explained_variance=self.config.explained_variance,
                random_state=self.config.seed,
            )
        self.memory = Memory(self.cache_dir, verbose=0) if self.cache_dir else None

    def _cache(self, func, *args):
        if self.memory is None:
            return func(*args)
        cached = self.memory.cache(func)
        return cached(*args)

    def fit_transform(self, train_frame) -> sparse.csr_matrix:
        start = time.time()
        train_text = self.text_preprocessor.fit_transform(train_frame)
        train_body = train_frame[self.config.text_columns[0]].fillna("")
        word = self._cache(_fit_transform_vectorizer, self.word_vectorizer, train_text)
        char = self._cache(_fit_transform_vectorizer, self.char_vectorizer, train_text)
        meta = self.meta_featurizer.fit(train_body).transform(train_body)
        features = sparse.hstack([_to_dense_if_needed(word), _to_dense_if_needed(char), meta], format="csr")
        if self.reducer is not None:
            features = self.reducer.fit_transform(features)
        elapsed = time.time() - start
        print(f"Transforming text features... done in {elapsed:.1f}s")
        return features

    def transform(self, frame) -> sparse.csr_matrix:
        text = self.text_preprocessor.transform(frame)
        body = frame[self.config.text_columns[0]].fillna("")
        word = self._cache(_transform_vectorizer, self.word_vectorizer, text)
        char = self._cache(_transform_vectorizer, self.char_vectorizer, text)
        meta = self.meta_featurizer.transform(body)
        features = sparse.hstack([_to_dense_if_needed(word), _to_dense_if_needed(char), meta], format="csr")
        if self.reducer is not None:
            features = self.reducer.transform(features)
        return features
