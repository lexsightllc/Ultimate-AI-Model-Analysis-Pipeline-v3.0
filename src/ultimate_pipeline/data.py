"""Data loading utilities for the analysis pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisConfig


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: Optional[pd.DataFrame]
    work_dir: Path
    is_synthetic: bool


def _discover_file(base_dirs: list[Path], filename: str) -> Optional[Path]:
    search_paths = []
    for base in base_dirs:
        search_paths.append(base / filename)
    for base in base_dirs:
        for child in base.glob("*/"):
            search_paths.append(child / filename)

    for candidate in search_paths:
        if candidate.exists():
            return candidate
    return None


def locate_data_files() -> Tuple[Optional[Path], Optional[Path], Optional[Path], Path]:
    input_dirs = [Path("/kaggle/input"), Path("/mnt/data"), Path.cwd()]
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    base_dirs = [d for d in input_dirs if d.exists()]
    train_path = _discover_file(base_dirs, "train.csv")
    test_path = _discover_file(base_dirs, "test.csv")
    sample_path = _discover_file(base_dirs, "sample_submission.csv")
    return train_path, test_path, sample_path, work_dir


def generate_synthetic_data(
    config: AnalysisConfig, work_dir: Path, n_train: int = 5_000, n_test: int = 1_000
) -> DatasetBundle:
    rng = np.random.default_rng(config.seed)
    rules = [f"Rule_{i+1}" for i in range(10)]
    subs = [f"Subreddit_{chr(65 + i)}" for i in range(8)]
    vocab = [
        "help",
        "urgent",
        "spam",
        "hate",
        "friendly",
        "ban",
        "policy",
        "violation",
        "content",
        "moderate",
        "community",
        "guidelines",
        "respect",
        "remove",
    ]

    def create_row(index: int, is_train: bool) -> dict:
        rule = rng.choice(rules)
        subreddit = rng.choice(subs)
        tokens = rng.choice(vocab, size=rng.integers(15, 100)).tolist()
        if rng.random() < 0.3:
            spam_tokens = ["spam", "promo", "discount", "free", "offer", "click", "link"]
            tokens.extend(rng.choice(spam_tokens, size=rng.integers(3, 8)).tolist())
        row = {
            "row_id": index if is_train else index + n_train,
            "body": " ".join(tokens),
            "rule": rule,
            "subreddit": subreddit,
            "positive_example_1": " ".join(rng.choice(tokens, size=min(len(tokens), 20))),
            "positive_example_2": " ".join(rng.choice(tokens, size=min(len(tokens), 20))),
            "negative_example_1": " ".join(rng.choice(tokens, size=min(len(tokens), 20))),
            "negative_example_2": " ".join(rng.choice(tokens, size=min(len(tokens), 20))),
        }
        if is_train:
            row["rule_violation"] = rng.integers(0, 2)
        return row

    train = pd.DataFrame(create_row(i, True) for i in range(n_train))
    test = pd.DataFrame(create_row(i, False) for i in range(n_test))
    sample = pd.DataFrame({"row_id": test["row_id"], "rule_violation": 0.5})

    out_dir = work_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    sample_path = out_dir / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    return DatasetBundle(
        train=train,
        test=test,
        sample_submission=sample,
        work_dir=work_dir,
        is_synthetic=True,
    )


def load_datasets(config: AnalysisConfig) -> DatasetBundle:
    train_path, test_path, sample_path, work_dir = locate_data_files()
    if not all([train_path, test_path, sample_path]):
        return generate_synthetic_data(config, work_dir)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    return DatasetBundle(
        train=train,
        test=test,
        sample_submission=sample,
        work_dir=work_dir,
        is_synthetic=False,
    )


def load_custom_datasets(
    config: AnalysisConfig,
    train_path: Path | str,
    test_path: Optional[Path | str] = None,
    sample_path: Optional[Path | str] = None,
) -> DatasetBundle:
    """Load datasets from explicitly provided paths."""

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path is not None else pd.DataFrame()
    sample = pd.read_csv(sample_path) if sample_path is not None else None
    work_dir = Path(config.cache_dir)
    return DatasetBundle(
        train=train,
        test=test,
        sample_submission=sample,
        work_dir=work_dir,
        is_synthetic=False,
    )
