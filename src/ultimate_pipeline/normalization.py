"""Text normalization utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class TextNormalizer(Protocol):
    """Protocol defining the interface for text normalization components."""

    def fit(self, texts: pd.Series) -> "TextNormalizer":
        return self

    def transform(self, texts: pd.Series) -> pd.Series:
        ...


@dataclass
class NoOpNormalizer:
    """A normalizer that returns text unchanged."""

    def fit(self, texts: pd.Series) -> "NoOpNormalizer":  # pragma: no cover - trivial
        return self

    def transform(self, texts: pd.Series) -> pd.Series:
        return texts.fillna("").astype(str)


@dataclass
class RegexNormalizer:
    """A lightweight regex-based normalizer for high-throughput scenarios."""

    url_pattern: re.Pattern[str] = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    html_pattern: re.Pattern[str] = re.compile(r"<[^>]+>")
    whitespace_pattern: re.Pattern[str] = re.compile(r"\s+")

    def fit(self, texts: pd.Series) -> "RegexNormalizer":  # pragma: no cover - stateless
        return self

    def transform(self, texts: pd.Series) -> pd.Series:
        cleaned = texts.fillna("").astype(str)
        cleaned = cleaned.str.replace(self.url_pattern, " ", regex=True)
        cleaned = cleaned.str.replace(self.html_pattern, " ", regex=True)
        cleaned = cleaned.str.replace(self.whitespace_pattern, " ", regex=True)
        return cleaned.str.strip()


@dataclass
class SpaCyNormalizer:
    """A spaCy-powered normalizer with tokenization and lemmatization."""

    model: str = "en_core_web_sm"
    disable: tuple[str, ...] = ("ner", "parser")

    def __post_init__(self) -> None:
        try:
            import spacy  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "spaCy is required for SpaCyNormalizer. Install it or switch to regex mode."
            ) from exc
        self._nlp = spacy.load(self.model, disable=list(self.disable))  # type: ignore[attr-defined]

    def fit(self, texts: pd.Series) -> "SpaCyNormalizer":  # pragma: no cover - stateless
        return self

    def transform(self, texts: pd.Series) -> pd.Series:
        docs = self._nlp.pipe(texts.fillna("").astype(str), batch_size=256)
        return pd.Series(" ".join(token.lemma_.lower() for token in doc if not token.is_space) for doc in docs)


def make_normalizer(name: str) -> TextNormalizer:
    normalized = (name or "noop").lower()
    if normalized in {"regex", "default"}:
        return RegexNormalizer()
    if normalized in {"spaCy", "spacy", "lemmatize"}:  # pragma: no branch - small set
        return SpaCyNormalizer()
    if normalized in {"none", "noop"}:
        return NoOpNormalizer()
    raise ValueError(f"Unknown normalizer '{name}'.")
