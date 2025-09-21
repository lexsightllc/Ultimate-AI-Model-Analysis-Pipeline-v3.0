"""Cross-validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:  # pragma: no cover - optional dependency
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore


@dataclass
class CrossValidatorFactory:
    n_splits: int
    seed: int

    def make(self, strategy: str, y: np.ndarray, groups: Optional[np.ndarray]):
        strategy = (strategy or "auto").lower()
        if strategy == "auto":
            if groups is not None and len(np.unique(groups)) >= self.n_splits:
                return GroupKFold(n_splits=self.n_splits), f"GroupKFold({self.n_splits})"
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed), f"StratifiedKFold({self.n_splits})"
        if strategy == "group":
            if groups is None:
                raise ValueError("Group cross-validation requires group labels.")
            return GroupKFold(n_splits=self.n_splits), f"GroupKFold({self.n_splits})"
        if strategy == "stratified":
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed), f"StratifiedKFold({self.n_splits})"
        if strategy == "stratified_group":
            if StratifiedGroupKFold is None:
                raise RuntimeError("StratifiedGroupKFold is unavailable in this scikit-learn version.")
            if groups is None:
                raise ValueError("StratifiedGroupKFold requires group labels.")
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv, f"StratifiedGroupKFold({self.n_splits})"
        raise ValueError(f"Unknown CV strategy '{strategy}'.")
