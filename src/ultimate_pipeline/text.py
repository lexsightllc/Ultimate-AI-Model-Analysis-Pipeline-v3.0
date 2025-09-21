"""Text preprocessing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from .normalization import TextNormalizer, make_normalizer


@dataclass
class TextPreprocessor:
    columns: Iterable[str]
    prefixes: Dict[str, str]
    normalizer: TextNormalizer

    def concatenate(self, frame: pd.DataFrame) -> pd.Series:
        prefixed_columns = []
        for column in self.columns:
            prefix = self.prefixes.get(column, f"{column}:")
            values = frame.get(column, "").fillna("").astype(str)
            prefixed_columns.append(prefix + " " + values)
        combined = pd.concat(prefixed_columns, axis=1)
        concatenated = combined.apply(lambda row: " ".join(part for part in row if part.strip()), axis=1)
        return concatenated.str.strip()

    def fit_transform(self, frame: pd.DataFrame) -> pd.Series:
        text = self.concatenate(frame)
        self.normalizer.fit(text)
        return self.normalizer.transform(text)

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        text = self.concatenate(frame)
        return self.normalizer.transform(text)


def build_preprocessor(columns: Iterable[str], prefixes: Dict[str, str], normalizer_name: str) -> TextPreprocessor:
    normalizer = make_normalizer(normalizer_name)
    return TextPreprocessor(columns=tuple(columns), prefixes=dict(prefixes), normalizer=normalizer)
