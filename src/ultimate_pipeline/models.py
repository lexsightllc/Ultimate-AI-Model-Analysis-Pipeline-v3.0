"""Model factory and calibration helpers."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC


@dataclass
class ModelFactory:
    """Create estimators with configuration-aware defaults."""

    use_sgd: bool
    seed: int
    max_iter: int = 1_000
    tol: float = 1e-4
    early_stopping: bool = True
    patience: int = 5
    n_jobs: int = -1

    def make_estimator(self, name: str | None = None):
        target = (name or ("sgd" if self.use_sgd else "logreg")).lower()
        if target == "logreg":
            return LogisticRegression(
                solver="saga",
                penalty="l2",
                C=1.0,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                random_state=self.seed,
                class_weight="balanced",
            )
        if target == "sgd":
            return SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=0.0001,
                l1_ratio=0.15,
                max_iter=self.max_iter,
                tol=self.tol,
                learning_rate="optimal",
                early_stopping=self.early_stopping,
                n_iter_no_change=self.patience,
                random_state=self.seed,
            )
        if target == "linearsvc":
            base = LinearSVC(
                C=1.0,
                max_iter=self.max_iter,
                dual=False,
                random_state=self.seed,
            )
            return CalibratedClassifierCV(base, method="sigmoid", cv=5)
        raise ValueError(f"Unknown model '{name}'.")
