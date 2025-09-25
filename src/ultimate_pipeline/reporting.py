"""Reporting utilities for saving analysis artefacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .metrics import MetricResult


@dataclass
class RunArtifacts:
    run_dir: Path
    cache_dir: Path
    models_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    run_id: str


def prepare_run_directory(base_dir: Path, run_id: str | None = None) -> RunArtifacts:
    timestamp = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    cache_dir = run_dir / "cache"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    for path in [run_dir, cache_dir, models_dir, reports_dir, artifacts_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_dir, cache_dir, models_dir, reports_dir, artifacts_dir, timestamp)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_dashboard(path: Path, metrics: List[MetricResult]) -> None:
    rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{m.auc:.4f}</td><td>{m.brier:.4f}</td><td>{m.ece:.4f}</td></tr>"
        for i, m in enumerate(metrics)
    )
    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Ultimate AI Model Analysis Pipeline</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2rem; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Cross-Validation Summary</h1>
        <table>
            <thead>
                <tr><th>Fold</th><th>AUC</th><th>Brier</th><th>ECE</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    path.write_text(html, encoding="utf-8")


def save_submission(
    path: Path,
    row_ids: np.ndarray,
    predictions: np.ndarray,
    *,
    id_column: str = "row_id",
    label_column: str = "rule_violation",
    class_labels: np.ndarray | None = None,
) -> None:
    preds = np.asarray(predictions)
    columns: Dict[str, np.ndarray] = {id_column: row_ids}
    if preds.ndim == 1:
        columns[label_column] = preds
    else:
        label_array = np.asarray(class_labels) if class_labels is not None else np.arange(preds.shape[1])
        labels = label_array
        if preds.shape[1] == 2:
            # Preserve compatibility with binary competitions by writing the positive class only.
            positive_index = 1
            if class_labels is not None:
                try:
                    positive_index = int(np.where(label_array == 1)[0][0])
                except Exception:  # pragma: no cover - fallback when label 1 missing
                    positive_index = preds.shape[1] - 1
                columns[label_column] = preds[:, positive_index]
            else:
                columns[label_column] = preds[:, positive_index]
        else:
            for idx in range(preds.shape[1]):
                name = f"{label_column}_{labels[idx]}"
                columns[name] = preds[:, idx]
    df = pd.DataFrame(columns)
    df.to_csv(path, index=False)


def save_oof_predictions(
    path: Path,
    row_ids: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    id_column: str = "row_id",
    label_column: str = "rule_violation",
    class_labels: np.ndarray | None = None,
) -> None:
    preds = np.asarray(predictions)
    payload: Dict[str, Any] = {id_column: row_ids, label_column: y_true}
    if preds.ndim == 1:
        payload["prediction"] = preds
    else:
        labels = np.asarray(class_labels) if class_labels is not None else np.arange(preds.shape[1])
        for idx in range(preds.shape[1]):
            payload[f"prediction_{labels[idx]}"] = preds[:, idx]
    df = pd.DataFrame(payload)
    df.to_csv(path, index=False)
