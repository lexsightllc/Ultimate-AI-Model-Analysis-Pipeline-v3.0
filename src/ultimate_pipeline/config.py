"""Configuration management for the Ultimate AI Model Analysis pipeline."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def _load_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - safe guard when PyYAML missing
            raise RuntimeError("PyYAML is required to load YAML configuration files.")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Immutable configuration container with convenience helpers."""

    version: str = "3.1.0"
    seed: int = 42
    performance_mode: str = "balanced"
    n_splits_max: int = 5
    min_group_k_folds: int = 3
    max_tfidf_features: int = 100_000
    calibration_enabled: bool = True
    calibration_method: str = "isotonic"
    epsilon_prob_clip: float = 1e-6
    n_ece_bins: int = 10
    top_n_features_display: int = 15
    text_columns: tuple[str, ...] = (
        "body",
        "rule",
        "subreddit",
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    )
    text_prefixes: Dict[str, str] = field(
        default_factory=lambda: {
            "rule": "rule:",
            "subreddit": "subreddit: r/",
            "positive_example_1": "positive1:",
            "positive_example_2": "positive2:",
            "negative_example_1": "negative1:",
            "negative_example_2": "negative2:",
            "body": "comment:",
        }
    )
    label_column: str = "rule_violation"
    id_column: str = "row_id"
    group_column: str | None = "rule"
    tfidf_word_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "strip_accents": "unicode",
            "lowercase": True,
            "ngram_range": (1, 2),
            "min_df": 3,
            "max_df": 0.95,
            "sublinear_tf": True,
        }
    )
    tfidf_char_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "strip_accents": None,
            "lowercase": False,
            "analyzer": "char",
            "ngram_range": (3, 5),
            "min_df": 3,
            "max_df": 0.95,
            "sublinear_tf": True,
        }
    )
    large_data_threshold: int = 10_000
    batch_size: int = 2_000
    n_jobs: int = -1
    early_stopping: bool = True
    patience: int = 5
    tol: float = 1e-4
    use_sgd: bool = False
    normalizer: str = "regex"
    vectorizer: str = "tfidf"
    vectorizer_mode: str = "word_char"
    dimensionality_reduction: Optional[str] = None
    dimensionality_reduction_components: Optional[int] = None
    explained_variance: Optional[float] = None
    cache_dir: Path = Path("runs")
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        self.text_columns = tuple(self.text_columns)
        self.text_prefixes = dict(self.text_prefixes)
        # Ensure every declared text column has a prefix to avoid KeyError later on.
        for column in self.text_columns:
            self.text_prefixes.setdefault(column, f"{column}:")
        self.tfidf_word_params = dict(self.tfidf_word_params)
        self.tfidf_char_params = dict(self.tfidf_char_params)
        self.epsilon_prob_clip = float(self.epsilon_prob_clip)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if self.group_column is not None and not isinstance(self.group_column, str):
            raise TypeError("group_column must be a string or None")
        if isinstance(self.dimensionality_reduction, str) and self.dimensionality_reduction.lower() == "none":
            self.dimensionality_reduction = None
        if self.dimensionality_reduction_components is not None:
            self.dimensionality_reduction_components = int(self.dimensionality_reduction_components)
        if self.explained_variance is not None:
            self.explained_variance = float(self.explained_variance)
        self.vectorizer_mode = (self.vectorizer_mode or "word_char").lower()
        valid_modes = {
            "word_char",
            "tfidf_char",
            "tfidf_char_wb",
            "tfidf_word_char_union",
            "char_only",
            "word_only",
        }
        if self.vectorizer_mode not in valid_modes:
            raise ValueError(
                "vectorizer_mode must be one of "
                "{'word_char','tfidf_char','tfidf_char_wb','tfidf_word_char_union','char_only','word_only'}"
            )

    @classmethod
    def from_file(cls, path: Path | str) -> "AnalysisConfig":
        params = _load_dict(Path(path))
        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        fields = set(cls.__dataclass_fields__.keys())
        unknown = [key for key in data if key not in fields]
        if unknown:
            raise KeyError(f"Unknown configuration keys: {unknown}")
        return cls(**data)

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "AnalysisConfig":
        if not overrides:
            return self
        data = self.as_dict()
        data.update(overrides)
        return AnalysisConfig.from_dict(data)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "performance_mode": self.performance_mode,
            "n_splits_max": self.n_splits_max,
            "min_group_k_folds": self.min_group_k_folds,
            "max_tfidf_features": self.max_tfidf_features,
            "calibration_enabled": self.calibration_enabled,
            "calibration_method": self.calibration_method,
            "epsilon_prob_clip": self.epsilon_prob_clip,
            "n_ece_bins": self.n_ece_bins,
            "top_n_features_display": self.top_n_features_display,
            "text_columns": self.text_columns,
            "text_prefixes": self.text_prefixes,
            "label_column": self.label_column,
            "id_column": self.id_column,
            "group_column": self.group_column,
            "tfidf_word_params": self.tfidf_word_params,
            "tfidf_char_params": self.tfidf_char_params,
            "large_data_threshold": self.large_data_threshold,
            "batch_size": self.batch_size,
            "n_jobs": self.n_jobs,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "tol": self.tol,
            "use_sgd": self.use_sgd,
            "normalizer": self.normalizer,
            "vectorizer": self.vectorizer,
            "vectorizer_mode": self.vectorizer_mode,
            "dimensionality_reduction": self.dimensionality_reduction,
            "dimensionality_reduction_components": self.dimensionality_reduction_components,
            "explained_variance": self.explained_variance,
            "cache_dir": str(self.cache_dir),
            "cache_enabled": self.cache_enabled,
        }

    def _apply_performance_mode(self) -> "AnalysisConfig":
        """Return a new configuration with performance mode tweaks applied."""

        data = self.as_dict()

        mode = self.performance_mode
        if mode == "best_accuracy":
            data["max_tfidf_features"] = 300_000
            data["tfidf_word_params"].update({
                "ngram_range": (1, 3),
                "min_df": 2,
            })
            data["tfidf_char_params"].update({
                "ngram_range": (2, 6),
                "min_df": 2,
            })
            data["calibration_method"] = "sigmoid"
            data["early_stopping"] = False
            data["use_sgd"] = False
        elif mode == "max_speed":
            data["max_tfidf_features"] = 50_000
            data["tfidf_word_params"].update({
                "ngram_range": (1, 1),
                "min_df": 5,
            })
            data["tfidf_char_params"].update({
                "ngram_range": (3, 4),
                "min_df": 5,
            })
            data["calibration_enabled"] = False
            data["n_splits_max"] = 3
            data["batch_size"] = 5_000
            data["use_sgd"] = True
        elif mode in {"balanced", "default", None}:
            data["use_sgd"] = data["large_data_threshold"] > 5_000
        else:
            LOGGER.warning(
                "Unknown performance mode '%s'; applying balanced defaults.",
                mode,
            )
            data["use_sgd"] = data["large_data_threshold"] > 5_000

        return AnalysisConfig.from_dict(data)


def load_default_config(overrides: Optional[Dict[str, Any]] = None) -> AnalysisConfig:
    if DEFAULT_CONFIG_PATH.exists():
        try:
            base = AnalysisConfig.from_file(DEFAULT_CONFIG_PATH)
        except RuntimeError as exc:
            LOGGER.warning(
                "Falling back to in-memory defaults because the config file could not be loaded: %s",
                exc,
            )
            base = AnalysisConfig()
    else:
        base = AnalysisConfig()
    return base.with_overrides(overrides)
