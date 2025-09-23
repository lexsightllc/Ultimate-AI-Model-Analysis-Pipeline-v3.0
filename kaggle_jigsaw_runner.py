#!/usr/bin/env python3
"""
AI Model Analysis — actionable diagnostics and upgrades translated into code.

What this script does (end-to-end, no placeholders):
  • Loads the Jigsaw Agile Community Rules dataset from Kaggle paths (or custom paths via CLI).
  • Builds a strong sparse text representation: union of word n-grams (1–2) and char_wb n-grams (3–5).
  • Optionally includes subreddit features; supports an ablation run that drops them to detect shortcut reliance.
  • Runs cross-validation with StratifiedKFold or StratifiedGroupKFold (group=subreddit) to harden validation.
  • Computes ROC AUC, Brier score, and a robust ECE with fixed bins; exports a reliability curve table.
  • Optionally calibrates probabilities INSIDE each CV fold (Platt/‘sigmoid’ or isotonic) with an inner calibration split.
  • Selects operational thresholds (0.5, Youden-J, max-F1) per fold and aggregates.
  • Fits on all training data, predicts on test, and GUARANTEES a valid /kaggle/working/submission.csv.
  • Writes concise artifacts under /kaggle/working:
        - submission.csv
        - diagnostics.json
        - reliability_curve.csv
        - cv_fold_metrics.csv

Defaults are safe for Kaggle. You can override via CLI flags (see parse_args).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------- Paths & env defaults -----------------------

KAGGLE_WORKING = Path("/kaggle/working")
# Ensure the working directory exists even outside Kaggle notebooks.
KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
COMP_DIR = Path("/kaggle/input/jigsaw-agile-community-rules")

DEFAULT_TRAIN = COMP_DIR / "train.csv"
DEFAULT_TEST = COMP_DIR / "test.csv"
DEFAULT_SAMPLE = COMP_DIR / "sample_submission.csv"

# ----------------------- Utilities -----------------------

def seed_everything(seed: Optional[int] = 42) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def _infer_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Return (text_col, subreddit_col, target_col) with robust fallbacks.
    """
    # text column candidates
    text_candidates = ["comment_text", "text", "comment", "body", "content"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        # fallback: first object column
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            raise ValueError("No suitable text column found.")
        text_col = obj_cols[0]

    subreddit_col = "subreddit" if "subreddit" in df.columns else None

    # target column candidates (only for train)
    target_candidates = ["rule_violation", "target", "label", "y"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    return text_col, subreddit_col, target_col

def _infer_id_column(df: pd.DataFrame) -> str:
    if "row_id" in df.columns:
        return "row_id"
    # otherwise first column
    return df.columns[0]

def prob_pos_from_estimator(estimator, X) -> np.ndarray:
    """
    Extracts p(y=1). Works with predict_proba or decision_function.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            return proba
        # pick column corresponding to class 1 / True
        if hasattr(estimator, "classes_"):
            classes = list(estimator.classes_)
            if 1 in classes:
                idx = classes.index(1)
            elif True in classes:
                idx = classes.index(True)
            else:
                idx = 1 if proba.shape[1] > 1 else 0
        else:
            idx = 1 if proba.shape[1] > 1 else 0
        return proba[:, idx]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X).ravel()
        lo, hi = np.min(scores), np.max(scores)
        return (scores - lo) / (hi - lo + 1e-12)
    raise NotFittedError("Estimator has neither predict_proba nor decision_function.")

def _calibration_table(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_pred, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    frame = pd.DataFrame({
        "bin": bin_ids,
        "y": y_true,
        "p": p_pred,
    })
    grouped = frame.groupby("bin", sort=True)

    counts = grouped.size().reindex(range(n_bins), fill_value=0).astype(float)
    sum_pred = grouped["p"].sum().reindex(range(n_bins), fill_value=0.0)
    sum_true = grouped["y"].sum().reindex(range(n_bins), fill_value=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = sum_pred / counts
        frac_pos = sum_true / counts

    weights = counts / max(float(len(frame)), 1.0)

    calib_df = pd.DataFrame({
        "bin": np.arange(n_bins),
        "bin_lower": bins[:-1],
        "bin_upper": bins[1:],
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "weight": weights,
        "count": counts,
    })
    return calib_df

def ece_fixed_bins(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 20) -> Tuple[float, pd.DataFrame]:
    """
    Robust ECE with fixed, uniform probability bins and weights by bin mass.
    Returns (ece, reliability_df) where reliability_df has mean_pred, frac_pos, weight per bin.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)

    calib_df = _calibration_table(y_true, p_pred, n_bins)
    valid = calib_df["count"] > 0
    ece = float(np.sum(
        calib_df.loc[valid, "weight"]
        * np.abs(calib_df.loc[valid, "mean_pred"] - calib_df.loc[valid, "frac_pos"])
    ))
    return ece, calib_df

def choose_thresholds(y_true: np.ndarray, p_pos: np.ndarray) -> Dict[str, Any]:
    """
    Returns thresholds and confusion-derived metrics for 0.5, Youden-J, and max-F1.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    fpr, tpr, roc_th = roc_curve(y_true, p_pos)
    prec, rec, pr_th = precision_recall_curve(y_true, p_pos)

    j_scores = tpr - fpr
    thr_j = roc_th[int(np.argmax(j_scores))]
    pr_thr = np.r_[pr_th, 1.0]
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = pr_thr[int(np.argmax(f1s))]

    def cm_at(thr: float) -> Dict[str, Any]:
        y_hat = (p_pos >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        recall = tp / (tp + fn + 1e-12)
        fpr_val = fp / (fp + tn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        return {
            "threshold": float(thr),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "recall": float(recall),
            "fpr": float(fpr_val),
            "precision": float(precision),
            "f1": float(f1_score(y_true, y_hat))
        }

    return {
        "thresholds": {
            "fixed_0_5": cm_at(0.5),
            "youden_j": cm_at(thr_j),
            "max_f1": cm_at(thr_f1),
        }
    }

# ----------------------- Modeling blocks -----------------------

def build_preprocessor(text_col: str,
                       include_subreddit: bool,
                       subreddit_col: Optional[str],
                       min_df: int,
                       word_max_features: int,
                       char_max_features: int) -> ColumnTransformer:
    """
    ColumnTransformer that:
      - applies TF-IDF word (1-2) and char_wb (3-5) to the text column
      - (optional) OHE on subreddit
    """
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=word_max_features, lowercase=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, max_features=char_max_features, lowercase=True)

    transformers = [
        ("word_ngrams", word_vec, text_col),
        ("charwb_ngrams", char_vec, text_col),
    ]
    if include_subreddit and subreddit_col is not None:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        transformers.append(("subreddit", ohe, [subreddit_col]))

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
    return pre

def build_estimator(C: float, penalty: str = "l2", solver: str = "liblinear", n_jobs: Optional[int] = None) -> LogisticRegression:
    """
    Logistic Regression suited for sparse high-dim text.
    """
    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=2000,
        n_jobs=n_jobs if solver in ("saga",) else None,  # liblinear ignores n_jobs
        verbose=0,
    )
    return clf

@dataclass
class CVResult:
    fold: int
    auc: float
    brier: float
    ece: float
    n: int
    thresholds: Dict[str, Any]

# ----------------------- Cross-validation with optional calibration -----------------------

def cross_validate(df: pd.DataFrame,
                   text_col: str,
                   target_col: str,
                   subreddit_col: Optional[str],
                   include_subreddit: bool,
                   use_group_cv: bool,
                   n_splits: int,
                   C: float,
                   min_df: int,
                   word_max_features: int,
                   char_max_features: int,
                   calibration: str = "none",
                   calib_size: float = 0.15,
                   ece_bins: int = 20,
                   random_state: int = 42) -> Tuple[List[CVResult], float, float, float]:
    """
    Performs CV, returns per-fold results and aggregate (mean AUC, mean Brier, mean ECE).
    """
    X = df.copy()
    y = df[target_col].astype(int).values
    groups = X[subreddit_col].values if (use_group_cv and subreddit_col and subreddit_col in X.columns) else None

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state) if groups is not None \
               else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: List[CVResult] = []
    aucs, briers, eces = [], [], []

    fold_idx = 0
    for train_idx, valid_idx in splitter.split(X, y, groups if groups is not None else None):
        fold_idx += 1
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        # Optional inner split for calibration INSIDE the fold
        X_fit, X_cal, y_fit, y_cal = X_tr, None, y_tr, None
        use_cal = calibration.lower() in ("platt", "sigmoid", "isotonic")
        if use_cal:
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X_tr, y_tr, test_size=calib_size, stratify=y_tr, random_state=random_state
            )

        pre = build_preprocessor(text_col, include_subreddit, subreddit_col, min_df, word_max_features, char_max_features)
        clf = build_estimator(C=C, solver="liblinear")  # consistent & robust on small-ish data

        base = Pipeline([( "pre", pre), ("clf", clf)])
        base.fit(X_fit, y_fit)

        model = base
        if use_cal and X_cal is not None:
            # Calibrate a prefit model using only the inner calibration chunk
            calibrated = CalibratedClassifierCV(estimator=model, method="isotonic" if calibration.lower() == "isotonic" else "sigmoid", cv="prefit")
            # Fit on the SAME pipeline object; it will call predict_proba/decision_function of the base
            calibrated.fit(X_cal, y_cal)
            model = calibrated

        # Predict on validation
        p_va = prob_pos_from_estimator(model, X_va)
        auc = roc_auc_score(y_va, p_va)
        brier = brier_score_loss(y_va, p_va)
        ece, _ = ece_fixed_bins(y_va, p_va, n_bins=ece_bins)
        thr = choose_thresholds(y_va, p_va)

        results.append(CVResult(fold=fold_idx, auc=float(auc), brier=float(brier), ece=float(ece), n=len(valid_idx), thresholds=thr["thresholds"]))
        aucs.append(auc); briers.append(brier); eces.append(ece)

    return results, float(np.mean(aucs)), float(np.mean(briers)), float(np.mean(eces))

# ----------------------- Ablation: drop subreddit features -----------------------

def run_ablation(train_df: pd.DataFrame,
                 text_col: str,
                 target_col: str,
                 subreddit_col: Optional[str],
                 **cv_kwargs) -> Dict[str, Any]:
    """
    Train/validate with subreddit features included vs. excluded, to detect shortcut reliance.
    Returns dict with both mean AUCs and delta.
    """
    res_incl, auc_incl, _, _ = cross_validate(
        train_df, text_col, target_col, subreddit_col,
        include_subreddit=True,
        **cv_kwargs
    )
    res_excl, auc_excl, _, _ = cross_validate(
        train_df, text_col, target_col, subreddit_col,
        include_subreddit=False,
        **cv_kwargs
    )
    return {
        "with_subreddit": {"mean_auc": auc_incl, "folds": [r.__dict__ for r in res_incl]},
        "without_subreddit": {"mean_auc": auc_excl, "folds": [r.__dict__ for r in res_excl]},
        "delta_auc": auc_incl - auc_excl
    }

# ----------------------- Fit on full data & predict test -----------------------

def fit_full_and_predict(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         text_col: str,
                         target_col: str,
                         subreddit_col: Optional[str],
                         include_subreddit: bool,
                         C: float,
                         min_df: int,
                         word_max_features: int,
                         char_max_features: int,
                         calibration: str = "none",
                         calib_size: float = 0.15,
                         random_state: int = 42) -> np.ndarray:
    """
    Fit on all training rows, optionally with an internal calibration split, then predict p(y=1) for test.
    """
    X_all = train_df.copy()
    y_all = train_df[target_col].astype(int).values

    # Optional inner calibration split (train vs calibrate), then refit calibrator on a holdout of training
    use_cal = calibration.lower() in ("platt", "sigmoid", "isotonic")
    if use_cal and len(X_all) > 5:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_all, y_all, test_size=calib_size, stratify=y_all, random_state=random_state
        )
    else:
        X_fit, X_cal, y_fit, y_cal = X_all, None, y_all, None

    pre = build_preprocessor(text_col, include_subreddit, subreddit_col, min_df, word_max_features, char_max_features)
    clf = build_estimator(C=C, solver="liblinear")
    base = Pipeline([("pre", pre), ("clf", clf)])
    base.fit(X_fit, y_fit)

    model = base
    if use_cal and X_cal is not None:
        calibrated = CalibratedClassifierCV(estimator=model, method="isotonic" if calibration.lower() == "isotonic" else "sigmoid", cv="prefit")
        calibrated.fit(X_cal, y_cal)
        model = calibrated

    p_test = prob_pos_from_estimator(model, test_df)
    return p_test

# ----------------------- Submission helpers -----------------------

def ensure_submission(sub: pd.DataFrame, sample_path: Path, test_path: Path, sub_path: Path) -> Path:
    """
    Validates and writes submission.csv to sub_path. Aligns header to sample and order to sample/test.
    """
    # Align header
    if sample_path.exists():
        sample_cols = list(pd.read_csv(sample_path, nrows=0).columns)
        if list(sub.columns) != sample_cols and len(sub.columns) >= 2:
            sub.columns = sample_cols[:len(sub.columns)]
        id_col = sample_cols[0]; target_col = sample_cols[1]
    else:
        id_col = sub.columns[0]; target_col = sub.columns[1]

    # Coerce numeric & clip
    sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Enforce order by sample or test if available
    if sample_path.exists():
        order = pd.read_csv(sample_path, usecols=[0])
        sub = order.merge(sub, left_on=order.columns[0], right_on=id_col, how="left")
        sub = sub[[order.columns[0], target_col]]
        sub.columns = [id_col, target_col]
    elif test_path.exists():
        order = pd.read_csv(test_path, usecols=[0])
        sub = order.merge(sub, left_on=order.columns[0], right_on=id_col, how="left")
        sub = sub[[order.columns[0], target_col]]
        sub.columns = [id_col, target_col]

    # Final checks
    if sub[id_col].duplicated().any():
        raise ValueError("Duplicate IDs in submission.")
    if sub[target_col].isna().any():
        raise ValueError("NaNs in submission target.")

    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    return sub_path

# ----------------------- Main orchestrator -----------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostics + upgrades for AUC/Calibration with robust CV")
    parser.add_argument("--train-csv", type=str, default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST))
    parser.add_argument("--sample-csv", type=str, default=str(DEFAULT_SAMPLE))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--use-group-cv", action="store_true", help="Use StratifiedGroupKFold grouped by subreddit")
    parser.add_argument("--include-subreddit", action="store_true", help="Include subreddit features (OHE)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation: with vs without subreddit features")
    parser.add_argument("--calibration", type=str, default="none", choices=["none", "platt", "sigmoid", "isotonic"])
    parser.add_argument("--ece-bins", type=int, default=20)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=200000)
    parser.add_argument("--char-max-features", type=int, default=200000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-path", type=str, default=str(KAGGLE_WORKING / "submission.csv"))
    parser.add_argument("--write-artifacts", action="store_true", help="Write diagnostics.json, cv_fold_metrics.csv, reliability_curve.csv")
    args, unknown = parser.parse_known_args(argv)
    # Silence Jupyter's -f arg etc.
    _ = [u for u in unknown if u]  # ignore
    return args

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    sample_path = Path(args.sample_csv)
    sub_path = Path(args.submission_path)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test CSVs. train={train_path} test={test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    text_col, subreddit_col, target_col = _infer_columns(train_df)
    if target_col is None:
        raise ValueError("Could not infer target column (expected one of: rule_violation, target, label, y).")

    # Light robustness: clip extreme caps ratio if present
    if "feat_caps_ratio" in train_df.columns:
        q = train_df["feat_caps_ratio"].quantile(0.99)
        train_df["feat_caps_ratio"] = np.clip(train_df["feat_caps_ratio"].values, None, float(q))
        if "feat_caps_ratio" in test_df.columns:
            test_df["feat_caps_ratio"] = np.clip(test_df["feat_caps_ratio"].values, None, float(q))

    # -------- Cross-validation diagnostics --------
    cv_kwargs = dict(
        use_group_cv=args.use_group_cv,
        n_splits=args.n_splits,
        C=args.C,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        calibration=args.calibration,
        calib_size=0.15,
        ece_bins=args.ece_bins,
        random_state=args.seed,
    )

    if args.ablation:
        print("[info] Running subreddit ablation (with vs without subreddit features)...")
        ab_res = run_ablation(train_df, text_col, target_col, subreddit_col, **cv_kwargs)
        print(json.dumps({
            "with_subreddit_mean_auc": round(ab_res["with_subreddit"]["mean_auc"], 6),
            "without_subreddit_mean_auc": round(ab_res["without_subreddit"]["mean_auc"], 6),
            "delta_auc": round(ab_res["delta_auc"], 6),
        }, indent=2))
        # Proceed with user choice for final fit (respect --include-subreddit flag)

    print(f"[info] Cross-validating with include_subreddit={args.include_subreddit} "
          f"and {'StratifiedGroupKFold' if args.use_group_cv else 'StratifiedKFold'}...")
    fold_res, mean_auc, mean_brier, mean_ece = cross_validate(
        train_df, text_col, target_col, subreddit_col,
        include_subreddit=args.include_subreddit,
        **cv_kwargs
    )
    print(json.dumps({
        "mean_auc": round(mean_auc, 6),
        "mean_brier": round(mean_brier, 6),
        "mean_ece": round(mean_ece, 6),
    }, indent=2))

    # Aggregate thresholds for information
    agg_thr = {
        "fixed_0_5": {k: float(np.mean([fr.thresholds["fixed_0_5"][k] for fr in fold_res])) for k in ["threshold", "tp", "fp", "tn", "fn", "recall", "fpr", "precision", "f1"] if k in fold_res[0].thresholds["fixed_0_5"]},
        "youden_j": {k: float(np.mean([fr.thresholds["youden_j"][k] for fr in fold_res])) for k in ["threshold", "tp", "fp", "tn", "fn", "recall", "fpr", "precision", "f1"] if k in fold_res[0].thresholds["youden_j"]},
        "max_f1": {k: float(np.mean([fr.thresholds["max_f1"][k] for fr in fold_res])) for k in ["threshold", "tp", "fp", "tn", "fn", "recall", "fpr", "precision", "f1"] if k in fold_res[0].thresholds["max_f1"]},
    }
    print("[info] Aggregated thresholds (mean across folds):")
    print(json.dumps(agg_thr, indent=2))

    # -------- Fit full model & predict test --------
    print("[info] Fitting on all training data and predicting test...")
    p_test = fit_full_and_predict(
        train_df, test_df, text_col, target_col, subreddit_col,
        include_subreddit=args.include_subreddit,
        C=args.C,
        min_df=args.min_df,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        calibration=args.calibration,
        calib_size=0.15,
        random_state=args.seed
    )

    # Build submission
    test_id_col = _infer_id_column(test_df)
    submission = pd.DataFrame({
        test_id_col: test_df[test_id_col],
        "rule_violation": p_test.astype(float)
    })
    sub_written = ensure_submission(submission, sample_path, test_path, Path(args.submission_path))
    print(f"[ok ] submission.csv saved to {sub_written.resolve()}")

    # -------- Optional artifacts --------
    if args.write_artifacts:
        # Export CV folds
        cv_df = pd.DataFrame([{
            "fold": r.fold,
            "auc": r.auc,
            "brier": r.brier,
            "ece": r.ece,
            "n": r.n,
            "thr_fixed_0_5": r.thresholds["fixed_0_5"]["threshold"],
            "f1_fixed_0_5": r.thresholds["fixed_0_5"]["f1"],
            "thr_youden_j": r.thresholds["youden_j"]["threshold"],
            "f1_youden_j": r.thresholds["youden_j"]["f1"],
            "thr_max_f1": r.thresholds["max_f1"]["threshold"],
            "f1_max_f1": r.thresholds["max_f1"]["f1"],
        } for r in fold_res])
        cv_df.to_csv(KAGGLE_WORKING / "cv_fold_metrics.csv", index=False)

        # A simple reliability curve using a held-out slice of train
        tr_idx, va_idx = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(train_df, train_df[target_col].astype(int).values))
        X_tr, X_va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        y_va = train_df[target_col].astype(int).values[va_idx]
        pre = build_preprocessor(text_col, args.include_subreddit, subreddit_col, args.min_df, args.word_max_features, args.char_max_features)
        clf = build_estimator(C=args.C, solver="liblinear")
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, train_df[target_col].astype(int).values[tr_idx])
        p_va = prob_pos_from_estimator(pipe, X_va)
        ece_val, calib_df = ece_fixed_bins(y_va, p_va, n_bins=args.ece_bins)
        calib_df.to_csv(KAGGLE_WORKING / "reliability_curve.csv", index=False)

        diag = {
            "mean_auc": float(mean_auc),
            "mean_brier": float(mean_brier),
            "mean_ece": float(mean_ece),
            "aggregated_thresholds": agg_thr,
            "ece_val_on_holdout": float(ece_val),
            "settings": {
                "include_subreddit": bool(args.include_subreddit),
                "use_group_cv": bool(args.use_group_cv),
                "n_splits": int(args.n_splits),
                "calibration": str(args.calibration),
                "ece_bins": int(args.ece_bins),
                "min_df": int(args.min_df),
                "word_max_features": int(args.word_max_features),
                "char_max_features": int(args.char_max_features),
                "C": float(args.C),
                "seed": int(args.seed),
            }
        }
        with open(KAGGLE_WORKING / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        print("[ok ] artifacts written: cv_fold_metrics.csv, reliability_curve.csv, diagnostics.json")


if __name__ == "__main__":
    main()
