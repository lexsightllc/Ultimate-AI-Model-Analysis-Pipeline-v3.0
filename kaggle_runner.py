#!/usr/bin/env python3
"""Convenience entrypoint for running the Ultimate Pipeline on Kaggle."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

# __file__ may be missing when the module is executed in certain Kaggle
# notebook contexts. Fall back to the current working directory which, on
# Kaggle, is the project root where this script lives. We also guard against
# symlinked paths by resolving the fallback path when possible.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:  # pragma: no cover - environment dependent
    PROJECT_ROOT = Path.cwd().resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from ultimate_pipeline.config import AnalysisConfig, load_default_config
from ultimate_pipeline.pipeline import AnalysisPipeline
from ultimate_pipeline.data import DatasetBundle, load_custom_datasets

KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_CODE_DATASET = Path("/kaggle/input/ultimate-ai-model-analysis-pipeline-v3-0")
DEFAULT_COMP_DATASET = Path("/kaggle/input/jigsaw-agile-community-rules")


def _default_output_dir() -> Path:
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING / "ultimate_pipeline_runs"
    return PROJECT_ROOT / "runs"


def _resolve_dataset_paths(
    config: AnalysisConfig, args: argparse.Namespace
) -> Optional[DatasetBundle]:
    train_csv = Path(args.train_csv) if args.train_csv else None
    test_csv = Path(args.test_csv) if args.test_csv else None
    sample_csv = Path(args.sample_csv) if args.sample_csv else None

    if train_csv:
        return load_custom_datasets(
            config,
            train_path=train_csv,
            test_path=test_csv,
            sample_path=sample_csv,
        )

    inferred_train = DEFAULT_COMP_DATASET / "train.csv"
    inferred_test = DEFAULT_COMP_DATASET / "test.csv"
    inferred_sample = DEFAULT_COMP_DATASET / "sample_submission.csv"
    if inferred_train.exists() and inferred_test.exists():
        return load_custom_datasets(
            config,
            train_path=inferred_train,
            test_path=inferred_test,
            sample_path=inferred_sample if inferred_sample.exists() else None,
        )

    return None


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    overrides: Dict[str, object] = {}
    if args.performance_mode:
        overrides["performance_mode"] = args.performance_mode
    if args.calibration:
        overrides["calibration_method"] = args.calibration
        overrides["calibration_enabled"] = args.calibration.lower() != "none"
    if args.normalizer:
        overrides["normalizer"] = args.normalizer
    if args.vectorizer:
        normalized_vectorizer = args.vectorizer.lower()
        char_modes = {
            "tfidf_char": "tfidf_char",
            "tfidf_char_wb": "tfidf_char_wb",
            "tfidf_word_char_union": "tfidf_word_char_union",
        }
        if normalized_vectorizer in {"tfidf", "hashing", "hashing_tfidf", "cuml"}:
            overrides["vectorizer"] = normalized_vectorizer
        elif normalized_vectorizer in char_modes:
            overrides["vectorizer"] = "tfidf"
            overrides["vectorizer_mode"] = char_modes[normalized_vectorizer]
        else:
            overrides["vectorizer"] = args.vectorizer
    if args.n_splits:
        overrides["n_splits_max"] = args.n_splits
    if args.max_features:
        overrides["max_tfidf_features"] = args.max_features
    if args.n_jobs is not None:
        overrides["n_jobs"] = args.n_jobs
    if args.output_dir:
        overrides["cache_dir"] = str(Path(args.output_dir))
    else:
        overrides["cache_dir"] = str(_default_output_dir())

    base_config: AnalysisConfig
    if args.config:
        base_config = AnalysisConfig.from_file(Path(args.config))
    else:
        base_config = load_default_config()
    return base_config.with_overrides(overrides)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ultimate Pipeline on Kaggle datasets")
    parser.add_argument("--config", type=str, help="Path to custom YAML/JSON config file")
    parser.add_argument("--performance-mode", type=str, help="Performance profile override")
    parser.add_argument("--calibration", type=str, help="Calibration strategy override")
    parser.add_argument("--normalizer", type=str, help="Text normalizer override")
    parser.add_argument(
        "--vectorizer",
        type=str,
        help=(
            "Vectorizer override (e.g. 'tfidf', 'tfidf_char_wb', 'tfidf_word_char_union', "
            "'hashing')"
        ),
    )
    parser.add_argument("--n-splits", type=int, help="Number of CV splits to use")
    parser.add_argument("--max-features", type=int, help="Maximum TF-IDF features")
    parser.add_argument("--n-jobs", type=int, help="Parallel jobs for estimators", default=None)
    parser.add_argument("--output-dir", type=str, help="Directory for pipeline artefacts")
    parser.add_argument("--train-csv", type=str, help="Explicit path to train.csv")
    parser.add_argument("--test-csv", type=str, help="Explicit path to test.csv")
    parser.add_argument("--sample-csv", type=str, help="Optional path to sample_submission.csv")
    parser.add_argument(
        "--final-submission",
        type=str,
        help="Destination path for the submission.csv copy",
        default=str((KAGGLE_WORKING if KAGGLE_WORKING.exists() else PROJECT_ROOT) / "submission.csv"),
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(
            "Ignoring unrecognized arguments: " + " ".join(unknown),
            file=sys.stderr,
        )
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    config = _build_config(args)
    pipeline = AnalysisPipeline(config)
    bundle = _resolve_dataset_paths(config, args)
    result = pipeline.run(bundle)

    submission_src = pipeline.artifacts.run_dir / "submission.csv"
    final_submission = Path(args.final_submission)
    final_submission.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(submission_src, final_submission)

    print(f"Submission saved to {final_submission.resolve()}")
    if result.metrics:
        mean_auc = sum(m.auc for m in result.metrics) / len(result.metrics)
        print("Mean CV AUC:", round(mean_auc, 6))
        print("Last fold AUC:", round(result.metrics[-1].auc, 6))

    oof_src = pipeline.artifacts.artifacts_dir / "oof_predictions.csv"
    if oof_src.exists():
        oof_target_root = KAGGLE_WORKING if KAGGLE_WORKING.exists() else PROJECT_ROOT
        oof_target = oof_target_root / "oof_predictions.csv"
        oof_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(oof_src, oof_target)
        print(f"OOF predictions saved to {oof_target.resolve()}")


if __name__ == "__main__":
    main()
