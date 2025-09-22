"""Command line interface for the Ultimate AI Model Analysis Pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import AnalysisConfig, load_default_config
from .pipeline import AnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultimate AI Model Analysis Pipeline")
    parser.add_argument("--config", type=str, help="Path to a YAML/JSON configuration file", default=None)
    parser.add_argument("--performance-mode", type=str, help="Performance profile (balanced|max_speed|best_accuracy)")
    parser.add_argument("--calibration", type=str, help="Calibration method (isotonic|sigmoid|none)")
    parser.add_argument("--normalizer", type=str, help="Text normalizer (regex|spacy|none)")
    parser.add_argument("--vectorizer", type=str, help="Vectorizer backend (tfidf|hashing|cuml)")
    parser.add_argument("--n-splits", type=int, help="Number of CV splits")
    parser.add_argument("--max-features", type=int, help="Maximum TF-IDF features")
    parser.add_argument("--dim-reduction", type=str, help="Dimensionality reduction strategy (svd|none)")
    parser.add_argument("--components", type=int, help="Number of SVD components")
    parser.add_argument("--explained-variance", type=float, help="Target explained variance for SVD")
    parser.add_argument("--n-jobs", type=int, help="Parallel jobs for estimators")
    parser.add_argument("--output-config", type=str, help="Path to dump the resolved configuration", default=None)
    parser.add_argument("--text-cols", nargs="+", help="Text columns to use for modelling")
    parser.add_argument("--label-col", type=str, help="Target label column name")
    parser.add_argument("--id-col", type=str, help="Identifier column for submissions")
    parser.add_argument("--train-csv", type=str, help="Explicit path to train.csv")
    parser.add_argument("--test-csv", type=str, help="Explicit path to test.csv")
    parser.add_argument(
        "--sample-submission-csv",
        type=str,
        help="Optional path to sample_submission.csv for schema reference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where run artefacts should be stored",
        default=None,
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.performance_mode:
        overrides["performance_mode"] = args.performance_mode
    if args.calibration:
        overrides["calibration_method"] = args.calibration
        overrides["calibration_enabled"] = args.calibration.lower() != "none"
    if args.normalizer:
        overrides["normalizer"] = args.normalizer
    if args.vectorizer:
        overrides["vectorizer"] = args.vectorizer
    if args.n_splits:
        overrides["n_splits_max"] = args.n_splits
    if args.max_features:
        overrides["max_tfidf_features"] = args.max_features
    if args.dim_reduction and args.dim_reduction.lower() != "none":
        overrides["dimensionality_reduction"] = args.dim_reduction
    if args.components:
        overrides["dimensionality_reduction_components"] = args.components
    if args.explained_variance:
        overrides["explained_variance"] = args.explained_variance
    if args.n_jobs is not None:
        overrides["n_jobs"] = args.n_jobs
    if args.text_cols:
        overrides["text_columns"] = tuple(args.text_cols)
        overrides["text_prefixes"] = {col: f"{col}:" for col in args.text_cols}
    if args.label_col:
        overrides["label_column"] = args.label_col
    if args.id_col:
        overrides["id_column"] = args.id_col
    if args.output_dir:
        overrides["cache_dir"] = args.output_dir
    return overrides


def main() -> None:
    args = parse_args()
    overrides = build_overrides(args)
    if args.config:
        config = AnalysisConfig.from_file(Path(args.config))
        if overrides:
            config = config.with_overrides(overrides)
    else:
        config = load_default_config(overrides)
    pipeline = AnalysisPipeline(config)

    bundle = None
    if args.train_csv or args.test_csv or args.sample_submission_csv:
        from .data import load_custom_datasets

        if not args.train_csv:
            raise ValueError("--train-csv must be provided when using explicit dataset paths")
        bundle = load_custom_datasets(
            config,
            train_path=args.train_csv,
            test_path=args.test_csv,
            sample_path=args.sample_submission_csv,
        )

    result = pipeline.run(bundle)
    if args.output_config:
        Path(args.output_config).write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    print("Final AUC:", result.metrics[-1].auc if result.metrics else "N/A")


if __name__ == "__main__":
    main()
