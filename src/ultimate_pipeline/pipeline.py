"""Main orchestration for the Ultimate AI Model Analysis Pipeline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .calibration import Calibrator, make_identity_calibrator
from .config import AnalysisConfig, load_default_config
from .cv import CrossValidatorFactory
from .data import DatasetBundle, load_datasets
from .features import FeatureAssembler
from .importance import FeatureImportanceResult, compute_linear_importance
from .metrics import MetricResult, compute_metrics
from .models import ModelFactory
from .reporting import (
    prepare_run_directory,
    save_dashboard,
    save_json,
    save_oof_predictions,
    save_submission,
)
from .tracking import build_tracker
from .text import build_preprocessor

LOGGER = logging.getLogger("ultimate_pipeline")


@dataclass
class PipelineResult:
    metrics: List[MetricResult]
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_importance: FeatureImportanceResult
    cv_strategy: str
    duration: float


class AnalysisPipeline:
    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        base_config = config or load_default_config()
        self.config = base_config._apply_performance_mode()
        self.run_id = self.config.run_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_s{self.config.seed}"
        self.artifacts = prepare_run_directory(self.config.cache_dir, self.run_id)
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
        self.tracker = build_tracker(
            self.config.tracker,
            run_id=self.run_id,
            artifacts_dir=self.artifacts.run_dir,
            tracker_uri=self.config.tracker_uri,
            project=self.config.tracker_project,
        )
        config_payload = self.config.as_dict()
        config_payload["run_id"] = self.run_id
        self.tracker.log_params(config_payload)

    def _validate_inputs(self, bundle: DatasetBundle) -> DatasetBundle:
        required = set(self.config.text_columns) | {self.config.label_column}
        missing = [col for col in required if col not in bundle.train.columns]
        if missing:
            raise ValueError(f"Training data missing columns: {missing}")
        train_df = bundle.train.copy()
        test_df = bundle.test.copy()
        for column in self.config.text_columns:
            if column not in test_df.columns:
                test_df[column] = ""
        if self.config.id_column and self.config.id_column not in test_df.columns:
            test_df[self.config.id_column] = np.arange(len(test_df))
        for frame in [train_df, test_df]:
            for column in self.config.text_columns:
                frame[column] = frame[column].fillna("").astype(str)

        return DatasetBundle(
            train=train_df,
            test=test_df,
            sample_submission=bundle.sample_submission,
            work_dir=bundle.work_dir,
            is_synthetic=bundle.is_synthetic,
        )

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
        bundle = self._validate_inputs(bundle)
        groups = None
        if self.config.group_column and self.config.group_column in bundle.train.columns:
            groups = bundle.train[self.config.group_column].astype(str).values
        if self.config.label_column not in bundle.train.columns:
            raise ValueError(
                f"Label column '{self.config.label_column}' not found in training data."
            )
        y = bundle.train[self.config.label_column].astype(int).values
        LOGGER.info("Data loaded: %d train, %d test samples", len(bundle.train), len(bundle.test))

        X_train = self.features.fit_transform(bundle.train)
        X_test = self.features.transform(bundle.test)
        reducer_metrics = getattr(self.features, "reducer_quality_metrics_", None)
        if reducer_metrics:
            self.tracker.log_metrics({f"dr_{key}": value for key, value in reducer_metrics.items()}, step=0)
        LOGGER.info("Feature matrix: Train=%s, Test=%s", X_train.shape, X_test.shape)

        cv, strategy = self._determine_cv(y, groups)
        metrics: List[MetricResult] = []
        classes = np.unique(y)
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        n_classes = len(classes)
        oof = np.zeros((len(y), n_classes), dtype=np.float32)
        test_preds: List[np.ndarray] = []
        models = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y, groups), 1):
            LOGGER.info("Fold %d/%d | Training on %d samples", fold, cv.get_n_splits(), len(tr_idx))
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = self.model_factory.make_estimator()
            model.fit(X_tr, y_tr)
            if self.config.calibration_enabled:
                calibrated = self.calibrator.calibrate(model, X_val, y_val)
            else:
                calibrated = make_identity_calibrator(model, self.config.epsilon_prob_clip, X_val, y_val)
            val_probs = calibrated.predict_proba(X_val)
            test_prob = calibrated.predict_proba(X_test)
            aligned_val = np.zeros((len(val_idx), n_classes), dtype=np.float32)
            aligned_test = np.zeros((len(test_prob), n_classes), dtype=np.float32)
            for col, cls in enumerate(calibrated.classes_):
                target_idx = class_to_index.get(cls, None)
                if target_idx is None:
                    continue
                aligned_val[:, target_idx] = val_probs[:, col]
                aligned_test[:, target_idx] = test_prob[:, col]
            oof[val_idx] = aligned_val
            test_preds.append(aligned_test)
            fold_metrics = compute_metrics(
                y_val,
                aligned_val,
                self.config.n_ece_bins,
                class_labels=classes,
            )
            metrics.append(fold_metrics)
            LOGGER.info("Fold %d | AUC=%.4f", fold, fold_metrics.auc)
            self.tracker.log_metrics(
                {"auc": fold_metrics.auc, "brier": fold_metrics.brier, "ece": fold_metrics.ece},
                step=fold,
            )
            models.append(model)

        test_prediction = np.mean(test_preds, axis=0)
        feature_importance = compute_linear_importance(
            models,
            self.features.word_vectorizer,
            self.features.char_vectorizer,
            self.config.top_n_features_display,
        )

        duration = time.time() - start
        LOGGER.info("Analysis completed in %.2fs", duration)

        row_ids = bundle.test.get(
            self.config.id_column, np.arange(len(bundle.test))
        )
        submission_path = self.artifacts.run_dir / "submission.csv"
        dashboard_path = self.artifacts.reports_dir / "dashboard.html"
        summary_path = self.artifacts.reports_dir / "analysis_results.json"
        save_submission(
            submission_path,
            row_ids,
            test_prediction,
            id_column=self.config.id_column,
            label_column=self.config.label_column,
            class_labels=classes,
        )
        train_ids = (
            bundle.train[self.config.id_column].values
            if self.config.id_column and self.config.id_column in bundle.train
            else np.arange(len(bundle.train))
        )
        oof_path = self.artifacts.artifacts_dir / "oof_predictions.csv"
        save_oof_predictions(
            oof_path,
            train_ids,
            y,
            oof,
            id_column=self.config.id_column or "row_id",
            label_column=self.config.label_column,
            class_labels=classes,
        )
        save_dashboard(dashboard_path, metrics)
        overall_metrics = compute_metrics(y, oof, self.config.n_ece_bins, class_labels=classes)
        summary = {
            "version": self.config.version,
            "performance_mode": self.config.performance_mode,
            "cv_strategy": strategy,
            "metrics": [m.as_dict() for m in metrics],
            "overall": overall_metrics.as_dict(),
            "duration_seconds": duration,
            "run_id": self.run_id,
        }
        save_json(summary_path, summary)
        self.tracker.log_metrics(
            {
                "overall_auc": overall_metrics.auc,
                "overall_brier": overall_metrics.brier,
                "overall_ece": overall_metrics.ece,
                "duration_seconds": duration,
            },
            step=cv.get_n_splits() + 1,
        )
        self.tracker.log_artifact(submission_path, name="submission.csv")
        self.tracker.log_artifact(oof_path, name="oof_predictions.csv")
        self.tracker.log_artifact(summary_path, name="analysis_results.json")
        self.tracker.log_artifact(dashboard_path, name="dashboard.html")

        result = PipelineResult(
            metrics=metrics,
            oof_predictions=oof,
            test_predictions=test_prediction,
            feature_importance=feature_importance,
            cv_strategy=strategy,
            duration=duration,
        )
        self.tracker.close()
        return result


def run_pipeline(config_overrides: Optional[Dict] = None) -> PipelineResult:
    config = load_default_config(config_overrides)
    pipeline = AnalysisPipeline(config)
    bundle = load_datasets(config)
    return pipeline.run(bundle)
