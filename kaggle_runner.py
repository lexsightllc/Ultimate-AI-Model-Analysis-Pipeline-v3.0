#!/usr/bin/env python3
"""Convenience entrypoint for running the Ultimate Pipeline on Kaggle."""

import argparse
import os
import shutil
import sys
import pathlib
import typing as t

# __file__ may be missing when the module is executed in certain Kaggle
# notebook contexts. Fall back to the current working directory which, on
# Kaggle, is the project root where this script lives. We also guard against
# symlinked paths by resolving the fallback path when possible.
try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
except NameError:  # pragma: no cover - environment dependent
    PROJECT_ROOT = pathlib.Path.cwd().resolve()

KAGGLE_WORKING = pathlib.Path("/kaggle/working")
DEFAULT_CODE_DATASET = pathlib.Path("/kaggle/input/ultimate-ai-model-analysis-pipeline-v3-0")
DEFAULT_COMP_DATASET = pathlib.Path("/kaggle/input/jigsaw-agile-community-rules")


def _ensure_code_on_sys_path() -> t.List[str]:
    """Ensure the source package is importable by appending plausible roots."""
    attempted: t.List[str] = []
    seen_attempted: t.Set[str] = set()
    seen_sys_path: t.Set[str] = set(sys.path)

    def _register_path(path: pathlib.Path) -> None:
        """Adds a path to sys.path if not already present."""
        path_str = str(path)
        if path_str not in seen_sys_path:
            sys.path.insert(0, path_str)
            seen_sys_path.add(path_str)

    candidates: t.List[pathlib.Path] = []
    env_src = os.environ.get("ULTIMATE_PIPELINE_SRC")
    if env_src:
        # Prioritize explicit environment variable
        candidates.append(pathlib.Path(env_src).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT,  # Current script's root
            pathlib.Path.cwd(),  # Current working directory
            KAGGLE_WORKING,  # Kaggle's standard working directory
            DEFAULT_CODE_DATASET,  # Kaggle's default code dataset path
        ]
    )

    for candidate in candidates:
        resolved = str(candidate.resolve(strict=False))
        if resolved not in seen_attempted:
            attempted.append(resolved)
            seen_attempted.add(resolved)
        
        # Add 'src' subdirectory and the candidate itself
        _register_path(candidate / "src")
        _register_path(candidate)

        # Look for wheels (.whl files) within the candidate path
        if candidate.exists():
            for wheel in sorted(candidate.rglob("*.whl")):
                _register_path(wheel)

    return attempted


_ATTEMPTED_CODE_PATHS = _ensure_code_on_sys_path()


def _default_output_dir() -> pathlib.Path:
    """Return the default cache/output directory for pipeline artefacts."""
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING / "ultimate_pipeline_runs"
    return PROJECT_ROOT / "runs"


try:
    import ultimate_pipeline.config as pipeline_config
    import ultimate_pipeline.pipeline as pipeline_runtime
    import ultimate_pipeline.data as pipeline_data
except ModuleNotFoundError as exc:
    if exc.name and not exc.name.startswith("ultimate_pipeline"):
        raise
    if _ATTEMPTED_CODE_PATHS:
        checked_locations = "\n  - " + "\n  - ".join(_ATTEMPTED_CODE_PATHS)
    else:
        checked_locations = ""
    message = (
        "Unable to import 'ultimate_pipeline'. Checked the following locations:"
        f"{checked_locations}\n"
        "Set the ULTIMATE_PIPELINE_SRC environment variable or attach the "
        "Ultimate Pipeline code dataset."
    )
    raise SystemExit(message) from exc

AnalysisConfig = pipeline_config.AnalysisConfig
AnalysisPipeline = pipeline_runtime.AnalysisPipeline
DatasetBundle = pipeline_data.DatasetBundle
load_default_config = pipeline_config.load_default_config
load_custom_datasets = pipeline_data.load_custom_datasets


def _resolve_dataset_paths(
    config: AnalysisConfig, args: argparse.Namespace
) -> t.Optional[DatasetBundle]:
    """Resolve dataset file paths prioritising explicit CLI arguments."""
    train_csv = pathlib.Path(args.train_csv) if args.train_csv else None
    test_csv = pathlib.Path(args.test_csv) if args.test_csv else None
    sample_csv = pathlib.Path(args.sample_csv) if args.sample_csv else None

    if train_csv:
        # If train_csv is explicitly provided, use custom dataset loader
        return load_custom_datasets(
            config,
            train_path=train_csv,
            test_path=test_csv,
            sample_path=sample_csv,
        )

    # Attempt to infer dataset paths from default Kaggle competition dataset
    inferred_train = DEFAULT_COMP_DATASET / "train.csv"
    inferred_test = DEFAULT_COMP_DATASET / "test.csv"
    inferred_sample = DEFAULT_COMP_DATASET / "sample_submission.csv"
    if inferred_train.exists() and inferred_test.exists():
        return load_custom_datasets(
            config,
            train_path=inferred_train,
            test_path=inferred_test,
            # Sample submission is optional
            sample_path=inferred_sample if inferred_sample.exists() else None,
        )

    return None  # No datasets found or provided


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    """Compose an AnalysisConfig instance applying CLI overrides."""
    overrides: t.Dict[str, t.Any] = {}

    # Apply performance mode override
    if args.performance_mode:
        overrides["performance_mode"] = args.performance_mode
    
    # Apply calibration strategy override
    if args.calibration:
        overrides["calibration_method"] = args.calibration
        overrides["calibration_enabled"] = args.calibration.lower() != "none"
    
    # Apply normalizer override
    if args.normalizer:
        overrides["normalizer"] = args.normalizer
    
    # Apply vectorizer override, handling special character modes
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
            overrides["vectorizer"] = "tfidf" # Set base vectorizer
            overrides["vectorizer_mode"] = char_modes[normalized_vectorizer] # Set mode
        else:
            # Fallback for unrecognized vectorizer string, pass as is
            overrides["vectorizer"] = args.vectorizer
    
    # Apply cross-validation splits override
    if args.n_splits:
        overrides["n_splits_max"] = args.n_splits
    
    # Apply max TF-IDF features override
    if args.max_features:
        overrides["max_tfidf_features"] = args.max_features
    
    # Apply parallel jobs override
    if args.n_jobs is not None:
        overrides["n_jobs"] = args.n_jobs
    
    # Determine cache directory
    if args.output_dir:
        overrides["cache_dir"] = str(pathlib.Path(args.output_dir))
    else:
        overrides["cache_dir"] = str(_default_output_dir())

    # Load base configuration (from file or default)
    base_config: AnalysisConfig
    if args.config:
        base_config = AnalysisConfig.from_file(pathlib.Path(args.config))
    else:
        base_config = load_default_config()
    
    # Apply all collected overrides
    return base_config.with_overrides(overrides)


def parse_args(argv: t.Optional[t.List[str]] = None) -> argparse.Namespace:
    """Parse and sanitise CLI arguments."""
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
    
    # Parse known arguments and capture any unknown ones
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(
            "Ignoring unrecognized arguments: " + " ".join(unknown),
            file=sys.stderr,
        )
    return args


def main(argv: t.Optional[t.List[str]] = None) -> None:
    """Run the configured analysis pipeline and persist Kaggle-friendly artefacts."""
    args = parse_args(argv)
    config = _build_config(args)
    pipeline = AnalysisPipeline(config)
    bundle = _resolve_dataset_paths(config, args)
    
    print("Transforming text features... ", end="")
    result = pipeline.run(bundle)
    print(f"done in {result.duration:.1f}s")

    # Save final submission file
    submission_src = pipeline.artifacts.run_dir / "submission.csv"
    final_submission = pathlib.Path(args.final_submission)
    final_submission.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(submission_src, final_submission)
    print(f"Submission saved to {final_submission.resolve()}")
    
    # Report mean CV AUC if metrics are available
    if result.metrics:
        mean_auc = sum(m.auc for m in result.metrics) / len(result.metrics)
        print("Mean CV AUC:", round(mean_auc, 6))
        # Assuming metrics are ordered, show the last fold's AUC
        print("Last fold AUC:", round(result.metrics[-1].auc, 6))

    # Save OOF predictions if they exist
    oof_src = pipeline.artifacts.artifacts_dir / "oof_predictions.csv"
    if oof_src.exists():
        oof_target_root = KAGGLE_WORKING if KAGGLE_WORKING.exists() else PROJECT_ROOT
        oof_target = oof_target_root / "oof_predictions.csv"
        oof_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(oof_src, oof_target)
        print(f"OOF predictions saved to {oof_target.resolve()}")


if __name__ == "__main__":
    # Example of how the script would be called within a Kaggle notebook,
    # often with default arguments or specific overrides.
    # The `sys.argv[1:]` handles typical command-line execution.
    # The `Ignoring unrecognized arguments...` line in the original output
    # suggests this script might be run with extra Jupyter/IPython specific
    # arguments which `parse_args` is designed to filter.
    main()
