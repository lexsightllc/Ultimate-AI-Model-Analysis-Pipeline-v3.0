"""Main orchestration for the Ultimate AI Model Analysis Pipeline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .calibration import Calibrator
from .config import AnalysisConfig, load_default_config
from .cv import CrossValidatorFactory
from .data import DatasetBundle, load_datasets
from .features import FeatureAssembler
from .importance import FeatureImportanceResult, compute_linear_importance
from .metrics import MetricResult, compute_metrics
from .models import ModelFactory
from .reporting import prepare_run_directory, save_dashboard, save_json, save_submission
from .text import build_preprocessor

LOGGER = logging.getLogger("ultimate_pipeline")


@dataclass
class PipelineResult:
    metrics: List[MetricResult]
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_importance: FeatureImportanceResult
    cv_strategy: str


class AnalysisPipeline:
    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or load_default_config()
        self.config._apply_performance_mode()
        self.artifacts = prepare_run_directory(self.config.cache_dir)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        LOGGER.info("Initialized pipeline with mode=%s", self.config.performance_mode)
        text_preprocessor = build_preprocessor(self.config.text_columns, self.config.text_prefixes, self.config.normalizer)
        cache_dir = self.artifacts.cache_dir if self.config.cache_enabled else None
        self.features = FeatureAssembler(self.config, text_preprocessor, cache_dir=cache_dir)
        self.model_factory = ModelFactory(
            use_sgd=self.config.use_sgd,
            seed=self.config.seed,
            max_iter=self.config.n_splits_max * 1000,
            tol=self.config.tol,
            early_stopping=self.config.early_stopping,
            patience=self.config.patience,
            n_jobs=self.config.n_jobs,
        )
        self.calibrator = Calibrator(self.config.calibration_method, self.config.epsilon_prob_clip)
        self.cv_factory = CrossValidatorFactory(self.config.n_splits_max, self.config.seed)

    def _validate_inputs(self, bundle: DatasetBundle) -> None:
        required = set(self.config.text_columns) | {"rule_violation"}
        missing = [col for col in required if col not in bundle.train.columns]
        if missing:
            raise ValueError(f"Training data missing columns: {missing}")
        for column in self.config.text_columns:
            if column not in bundle.test.columns:
                bundle.test[column] = ""
        for frame in [bundle.train, bundle.test]:
            for column in self.config.text_columns:
                frame[column] = frame[column].fillna("").astype(str)

    def _determine_cv(self, y: np.ndarray, groups: Optional[np.ndarray]):
        n_splits = min(self.config.n_splits_max, max(2, int(np.bincount(y).min())))
        factory = CrossValidatorFactory(n_splits=n_splits, seed=self.config.seed)
        groups_array = groups if groups is not None else None
        cv, strategy = factory.make("auto", y, groups_array)
        LOGGER.info("CV Strategy: %s", strategy)
        return cv, strategy

    def run(self, bundle: Optional[DatasetBundle] = None) -> PipelineResult:
        start = time.time()
        bundle = bundle or load_datasets(self.config)
        self._validate_inputs(bundle)
        groups = bundle.train.get("rule", "").astype(str).values
        y = bundle.train["rule_violation"].astype(int).values
        LOGGER.info("Data loaded: %d train, %d test samples", len(bundle.train), len(bundle.test))

        X_train = self.features.fit_transform(bundle.train)
        X_test = self.features.transform(bundle.test)
        LOGGER.info("Feature matrix: Train=%s, Test=%s", X_train.shape, X_test.shape)

        cv, strategy = self._determine_cv(y, groups)
        metrics: List[MetricResult] = []
        oof = np.zeros_like(y, dtype=np.float32)
        test_preds: List[np.ndarray] = []
        models = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y, groups), 1):
            LOGGER.info("Fold %d/%d | Training on %d samples", fold, cv.get_n_splits(), len(tr_idx))
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = self.model_factory.make_estimator()
            model.fit(X_tr, y_tr)
            calibrator = self.calibrator.calibrate(model, X_val, y_val) if self.config.calibration_enabled else None
            if calibrator is not None:
                val_probs = calibrator.predict(X_val)
                test_prob = calibrator.predict(X_test)
            else:
                val_probs = model.predict_proba(X_val)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]
            val_probs = np.clip(val_probs, self.config.epsilon_prob_clip, 1 - self.config.epsilon_prob_clip)
            test_prob = np.clip(test_prob, self.config.epsilon_prob_clip, 1 - self.config.epsilon_prob_clip)
            oof[val_idx] = val_probs
            test_preds.append(test_prob)
            fold_metrics = compute_metrics(y_val, val_probs, self.config.n_ece_bins)
            metrics.append(fold_metrics)
            LOGGER.info("Fold %d | AUC=%.4f", fold, fold_metrics.auc)
            models.append(model)

        test_prediction = np.mean(test_preds, axis=0)
        feature_importance = compute_linear_importance(models, self.features.word_vectorizer, self.features.char_vectorizer, self.config.top_n_features_display)

        duration = time.time() - start
        LOGGER.info("Analysis completed in %.2fs", duration)

        row_ids = bundle.test.get("row_id", np.arange(len(bundle.test)))
        submission_path = self.artifacts.run_dir / "submission.csv"
        dashboard_path = self.artifacts.reports_dir / "dashboard.html"
        summary_path = self.artifacts.reports_dir / "analysis_results.json"
        save_submission(submission_path, row_ids, test_prediction)
        save_dashboard(dashboard_path, metrics)
        summary = {
            "version": self.config.version,
            "performance_mode": self.config.performance_mode,
            "cv_strategy": strategy,
            "metrics": [m.as_dict() for m in metrics],
            "overall": compute_metrics(y, oof, self.config.n_ece_bins).as_dict(),
            "duration_seconds": duration,
        }
        save_json(summary_path, summary)
        # Mirror high-value artefacts to working directory root for convenience
        save_submission(Path(submission_path.name), row_ids, test_prediction)
        save_dashboard(Path(dashboard_path.name), metrics)
        save_json(Path(summary_path.name), summary)

        return PipelineResult(
            metrics=metrics,
            oof_predictions=oof,
            test_predictions=test_prediction,
            feature_importance=feature_importance,
            cv_strategy=strategy,
        )


def run_pipeline(config_overrides: Optional[Dict] = None) -> PipelineResult:
    config = load_default_config(config_overrides)
    pipeline = AnalysisPipeline(config)
    bundle = load_datasets(config)
    return pipeline.run(bundle)
