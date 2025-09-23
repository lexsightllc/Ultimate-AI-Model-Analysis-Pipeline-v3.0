"""Implementation of a configurable multi-label text classification pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Settings controlling text normalisation."""

    lowercase: bool = True
    remove_punctuation: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_numbers: bool = False
    normalize_whitespace: bool = True
    remove_stopwords: bool = False
    min_word_length: int = 2
    max_word_length: int = 50
    handle_emojis: str = "remove"  # "remove", "keep", or "convert"


@dataclass
class VectorizerConfig:
    """Configuration for converting raw text into numerical features."""

    type: str = "tfidf"  # tfidf or count
    max_features: int = 10_000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    use_char_ngrams: bool = False
    char_ngram_range: Tuple[int, int] = (3, 5)
    sublinear_tf: bool = True


@dataclass
class ModelConfig:
    """Estimator choices for multi-label classification."""

    type: str = "logistic"  # logistic, rf, or gbm
    use_class_weights: bool = True
    calibration_method: Optional[str] = None  # None, "sigmoid", "isotonic"
    max_iter: int = 1000
    random_state: int = 42
    n_estimators: int = 100  # used by tree-based models


@dataclass
class TrainingConfig:
    """Parameters controlling the cross-validation strategy."""

    n_splits: int = 5
    stratify: bool = True
    validation_size: float = 0.2
    early_stopping: bool = False
    patience: int = 3


@dataclass
class AnalysisConfig:
    """Root configuration for the multi-label pipeline."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    performance_mode: str = "standard"  # fast, standard, enhance
    output_dir: str = "runs"
    competition_name: str = "jigsaw-agile-community-rules"
    debug: bool = False

    def apply_performance_mode(self) -> None:
        """Mutate configuration to respect the requested performance preset."""

        mode = (self.performance_mode or "standard").lower()
        if mode == "fast":
            self.vectorizer.max_features = 5_000
            self.vectorizer.ngram_range = (1, 1)
            self.training.n_splits = 3
        elif mode == "enhance":
            self.vectorizer.max_features = 20_000
            self.vectorizer.ngram_range = (1, 3)
            self.vectorizer.use_char_ngrams = True
            self.training.n_splits = max(self.training.n_splits, 7)
            self.model.use_class_weights = True


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


class TextNormalizer:
    """Apply deterministic preprocessing for community-sourced text."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.mention_pattern = re.compile(r"@\w+")
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols
            "\U0001F680-\U0001F6FF"  # transport
            "\U0001F1E0-\U0001F1FF"  # flags
            "]",
            flags=re.UNICODE,
        )
        self.punct_pattern = re.compile(r"[^\w\s]")
        self.number_pattern = re.compile(r"\d+")

    def normalize(self, text: Union[str, float]) -> str:
        if pd.isna(text):
            return ""

        value = str(text)

        if self.config.remove_urls:
            value = self.url_pattern.sub(" URL ", value)
        if self.config.remove_mentions:
            value = self.mention_pattern.sub(" USER ", value)

        if self.config.handle_emojis == "remove":
            value = self.emoji_pattern.sub(" ", value)
        elif self.config.handle_emojis == "convert":
            value = self.emoji_pattern.sub(" EMOJI ", value)

        if self.config.lowercase:
            value = value.lower()
        if self.config.remove_punctuation:
            value = self.punct_pattern.sub(" ", value)
        if self.config.remove_numbers:
            value = self.number_pattern.sub(" ", value)
        if self.config.normalize_whitespace:
            value = " ".join(value.split())

        if self.config.min_word_length > 1 or self.config.max_word_length < 50:
            tokens = value.split()
            tokens = [
                token
                for token in tokens
                if self.config.min_word_length
                <= len(token)
                <= self.config.max_word_length
            ]
            value = " ".join(tokens)

        return value


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Convert normalised text into sparse feature matrices."""

    def __init__(self, config: VectorizerConfig):
        self.config = config
        self.vectorizer: Optional[Union[TfidfVectorizer, CountVectorizer]] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, texts: Sequence[str]) -> "FeatureExtractor":
        LOGGER.info(
            "Fitting %s vectorizer with max_features=%d",
            self.config.type,
            self.config.max_features,
        )
        if self.config.type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                sublinear_tf=self.config.sublinear_tf,
            )
        elif self.config.type == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
            )
        else:
            raise ValueError(f"Unsupported vectorizer type: {self.config.type}")

        self.vectorizer.fit(texts)

        if self.config.use_char_ngrams:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.config.char_ngram_range,
                max_features=min(2_000, self.config.max_features),
            )
            self.char_vectorizer.fit(texts)

        return self

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer must be fit before calling transform")

        word_features = self.vectorizer.transform(texts)

        if not self.char_vectorizer:
            return word_features

        char_features = self.char_vectorizer.transform(texts)
        return sparse.hstack([word_features, char_features], format="csr")

    def fit_transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)


# ---------------------------------------------------------------------------
# Multi-label model wrapper
# ---------------------------------------------------------------------------


class MultiLabelClassifierWrapper:
    """Train one estimator per label with optional calibration."""

    def __init__(self, config: ModelConfig, n_labels: int):
        self.config = config
        self.n_labels = n_labels
        self.estimators: List[Union[CalibratedClassifierCV, LogisticRegression, RandomForestClassifier, GradientBoostingClassifier]] = []

    def _make_base_estimator(self):
        if self.config.type == "logistic":
            return LogisticRegression(
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                solver="liblinear",
            )
        if self.config.type == "rf":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        if self.config.type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
            )
        raise ValueError(f"Unsupported model type: {self.config.type}")

    def fit(self, X: sparse.csr_matrix, y: np.ndarray) -> "MultiLabelClassifierWrapper":
        self.estimators = []
        for index in range(self.n_labels):
            LOGGER.info("Training model for label %d/%d", index + 1, self.n_labels)
            target = y[:, index]
            estimator = self._make_base_estimator()

            fit_kwargs: Dict[str, np.ndarray] = {}
            if self.config.use_class_weights:
                unique_classes = np.unique(target)
                if len(unique_classes) > 1:
                    class_weights = compute_class_weight(
                        "balanced",
                        classes=unique_classes,
                        y=target,
                    )
                    sample_weight = np.zeros_like(target, dtype=float)
                    for cls, weight in zip(unique_classes, class_weights):
                        sample_weight[target == cls] = weight
                    fit_kwargs["sample_weight"] = sample_weight

            if self.config.calibration_method:
                estimator = CalibratedClassifierCV(
                    estimator,
                    method=self.config.calibration_method,
                    cv=3,
                )
                estimator.fit(X, target)
            else:
                estimator.fit(X, target, **fit_kwargs)

            self.estimators.append(estimator)
        return self

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        if not self.estimators:
            raise RuntimeError("Model has not been fit")

        probabilities = np.zeros((X.shape[0], self.n_labels), dtype=float)
        for index, estimator in enumerate(self.estimators):
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    probabilities[:, index] = proba[:, 1]
                else:
                    probabilities[:, index] = proba.ravel()
            elif hasattr(estimator, "decision_function"):
                decision = estimator.decision_function(X)
                probabilities[:, index] = 1.0 / (1.0 + np.exp(-decision))
            else:
                raise AttributeError("Estimator lacks probability outputs")
        return probabilities


# ---------------------------------------------------------------------------
# Main pipeline implementation
# ---------------------------------------------------------------------------


class UltimateMultilabelPipeline:
    """End-to-end training loop for the Kaggle Jigsaw competition."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.config.apply_performance_mode()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        LOGGER.info(
            "Initialised UltimateMultilabelPipeline with mode=%s", self.config.performance_mode
        )
        self.normalizer = TextNormalizer(self.config.preprocessing)
        self.features = FeatureExtractor(self.config.vectorizer)
        self.label_columns: List[str] = []
        self.cv_scores: List[float] = []
        self.model: Optional[MultiLabelClassifierWrapper] = None

    def _possible_data_roots(self) -> List[Path]:
        return [
            Path("/kaggle/input") / self.config.competition_name,
            Path("/kaggle/input/jigsaw-agile-community-rules"),
            Path("."),
        ]

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for root in self._possible_data_roots():
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            if train_path.exists() and test_path.exists():
                LOGGER.info("Loading competition data from %s", train_path.parent)
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                break
        else:  # pragma: no cover - safety net for environments without data
            raise FileNotFoundError("Could not locate train/test CSVs")

        self.label_columns = [
            column
            for column in train_df.columns
            if column not in {"id", "text", "comment_text"}
        ]
        if not self.label_columns:
            raise ValueError("No label columns detected in training data")

        LOGGER.info(
            "Detected %d labels: %s", len(self.label_columns), ", ".join(self.label_columns)
        )
        return train_df, test_df

    def preprocess(self, texts: Sequence[str]) -> List[str]:
        LOGGER.info("Preprocessing %d texts", len(texts))
        start = time.time()
        processed = [self.normalizer.normalize(text) for text in texts]
        LOGGER.info("Preprocessing completed in %.1fs", time.time() - start)
        return processed

    def build_features(self, texts: Sequence[str], fit: bool) -> sparse.csr_matrix:
        mode = self.config.performance_mode.capitalize()
        LOGGER.info("%s mode: transforming text features", mode)
        start = time.time()
        matrix = self.features.fit_transform(texts) if fit else self.features.transform(texts)
        LOGGER.info(
            "%s Transforming text features... done in %.1fs", mode, time.time() - start
        )
        return matrix

    def _build_cv(self, y: np.ndarray) -> Union[KFold, StratifiedKFold]:
        if self.config.training.stratify and y.shape[1] > 0:
            return StratifiedKFold(
                n_splits=self.config.training.n_splits,
                shuffle=True,
                random_state=self.config.model.random_state,
            )
        return KFold(
            n_splits=self.config.training.n_splits,
            shuffle=True,
            random_state=self.config.model.random_state,
        )

    def cross_validate(self, X: sparse.csr_matrix, y: np.ndarray) -> float:
        splitter = self._build_cv(y)
        scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, np.clip(y[:, 0], 0, 1) if y.shape[1] else None),
            start=1,
        ):
            LOGGER.info("Training fold %d/%d", fold, splitter.get_n_splits())
            model = MultiLabelClassifierWrapper(self.config.model, y.shape[1])
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[val_idx])
            fold_scores = []
            for column in range(y.shape[1]):
                try:
                    score = roc_auc_score(y[val_idx, column], preds[:, column])
                except ValueError:
                    score = 0.5
                fold_scores.append(score)
            mean_auc = float(np.mean(fold_scores))
            LOGGER.info("Fold %d AUC=%.6f", fold, mean_auc)
            scores.append(mean_auc)

        self.cv_scores = scores
        return float(np.mean(scores))

    def train_final(self, X: sparse.csr_matrix, y: np.ndarray) -> None:
        LOGGER.info("Training final model on %d samples", X.shape[0])
        self.model = MultiLabelClassifierWrapper(self.config.model, y.shape[1])
        self.model.fit(X, y)

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Final model has not been trained")
        return self.model.predict_proba(X)

    def save_submission(self, test_df: pd.DataFrame, predictions: np.ndarray) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({"id": test_df["id"]})
        for idx, column in enumerate(self.label_columns):
            submission[column] = predictions[:, idx]
        path = output_dir / "submission.csv"
        submission.to_csv(path, index=False)
        LOGGER.info("Submission saved to %s", path)
        return path

    def run(self) -> float:
        LOGGER.info("=" * 60)
        LOGGER.info("Ultimate AI Model Analysis Pipeline v3.0")
        LOGGER.info("Performance mode: %s", self.config.performance_mode)
        LOGGER.info("=" * 60)

        train_df, test_df = self.load_data()
        text_column = "comment_text" if "comment_text" in train_df.columns else "text"
        train_texts = self.preprocess(train_df[text_column])
        test_texts = self.preprocess(test_df[text_column])
        X_train = self.build_features(train_texts, fit=True)
        X_test = self.build_features(test_texts, fit=False)
        y_train = train_df[self.label_columns].astype(int).values

        cv_auc = self.cross_validate(X_train, y_train)
        LOGGER.info("Final fold AUC: %.12f", cv_auc)

        self.train_final(X_train, y_train)
        predictions = self.predict(X_test)
        self.save_submission(test_df, predictions)

        LOGGER.info("Pipeline execution completed successfully")
        return cv_auc


# ---------------------------------------------------------------------------
# Configuration loading utilities
# ---------------------------------------------------------------------------


def _load_nested_config(data: Dict[str, object]) -> AnalysisConfig:
    return AnalysisConfig(
        preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
        vectorizer=VectorizerConfig(**data.get("vectorizer", {})),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        performance_mode=data.get("performance_mode", "standard"),
        output_dir=data.get("output_dir", "runs"),
        competition_name=data.get("competition_name", "jigsaw-agile-community-rules"),
        debug=bool(data.get("debug", False)),
    )


def load_config(config_path: Optional[Union[str, Path]]) -> AnalysisConfig:
    if not config_path:
        return AnalysisConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # Lazily import to avoid dependency when unused

        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    else:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)

    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")

    return _load_nested_config(raw)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-label Ultimate Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON configuration file")
    parser.add_argument("--performance-mode", type=str, choices=["fast", "standard", "enhance"])
    parser.add_argument("--n-splits", type=int, help="Cross-validation folds")
    parser.add_argument("--max-features", type=int, help="TF-IDF feature cap")
    parser.add_argument("--model-type", type=str, choices=["logistic", "rf", "gbm"])
    parser.add_argument("--calibration", type=str, choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output-config", type=str, help="Optional path to save resolved config")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> AnalysisConfig:
    config = load_config(args.config)
    if args.performance_mode:
        config.performance_mode = args.performance_mode
    if args.n_splits:
        config.training.n_splits = args.n_splits
    if args.max_features:
        config.vectorizer.max_features = args.max_features
    if args.model_type:
        config.model.type = args.model_type
    if args.calibration:
        config.model.calibration_method = None if args.calibration == "none" else args.calibration
    if args.debug:
        config.debug = True
        logging.getLogger().setLevel(logging.DEBUG)

    if args.output_config:
        path = Path(args.output_config)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)
        LOGGER.info("Resolved configuration saved to %s", path)

    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = resolve_config(args)
    pipeline = UltimateMultilabelPipeline(config)
    score = pipeline.run()

    print("\nFinal Results:")
    print(f"Average CV AUC: {score:.6f}")
    if pipeline.cv_scores:
        print("Per-fold scores:", pipeline.cv_scores)


if __name__ == "__main__":  # pragma: no cover - guard for script execution
    main()
