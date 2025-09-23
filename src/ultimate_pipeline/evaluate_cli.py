"""Standalone CLI for evaluating saved prediction files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .evaluation import EvaluationSummary, evaluate_prediction_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV files with calibration metrics.")
    parser.add_argument("csv", type=str, help="Path to the CSV file containing labels and predictions.")
    parser.add_argument("--label-col", type=str, default="rule_violation", help="Name of the label column.")
    parser.add_argument("--prediction-col", type=str, default="prediction", help="Name of the prediction column.")
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of bins for ECE computation.")
    parser.add_argument("--no-clip", action="store_true", help="Disable probability clipping before evaluation.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Clipping epsilon applied when probabilities are clipped.")
    parser.add_argument("--output-json", type=str, help="Optional path to write the evaluation summary as JSON.")
    parser.add_argument(
        "--calibration-csv",
        type=str,
        help="Optional path to export the calibration table (one row per probability bin).",
    )
    return parser.parse_args()


def _write_json(path: Path, summary: EvaluationSummary) -> None:
    path.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")


def _write_calibration(path: Path, summary: EvaluationSummary) -> None:
    import csv

    fieldnames = ["lower", "upper", "count", "accuracy", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.calibration_table:
            writer.writerow(row.as_dict())


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()
    summary = evaluate_prediction_file(
        parsed.csv,
        label_column=parsed.label_col,
        prediction_column=parsed.prediction_col,
        n_bins=parsed.ece_bins,
        clip=not parsed.no_clip,
        epsilon=parsed.epsilon,
    )

    print(f"Samples: {summary.n_samples}")
    print(f"Positive rate: {summary.positive_rate:.4f}")
    print(f"Mean prediction: {summary.prediction_mean:.4f}")
    print(f"AUC: {summary.metrics.auc:.6f}")
    print(f"Brier score: {summary.metrics.brier:.6f}")
    print(f"ECE ({parsed.ece_bins} bins): {summary.metrics.ece:.6f}")

    if parsed.output_json:
        _write_json(Path(parsed.output_json), summary)
    if parsed.calibration_csv:
        _write_calibration(Path(parsed.calibration_csv), summary)


if __name__ == "__main__":
    main()
