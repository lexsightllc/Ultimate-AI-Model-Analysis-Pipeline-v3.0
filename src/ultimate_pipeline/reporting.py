"""Reporting utilities for saving analysis artefacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .metrics import MetricResult


@dataclass
class RunArtifacts:
    run_dir: Path
    cache_dir: Path
    models_dir: Path
    reports_dir: Path
    artifacts_dir: Path


def prepare_run_directory(base_dir: Path) -> RunArtifacts:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    cache_dir = run_dir / "cache"
    models_dir = run_dir / "models"
    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    for path in [run_dir, cache_dir, models_dir, reports_dir, artifacts_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_dir, cache_dir, models_dir, reports_dir, artifacts_dir)


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


def save_submission(path: Path, row_ids: np.ndarray, predictions: np.ndarray) -> None:
    import pandas as pd

    df = pd.DataFrame({"row_id": row_ids, "rule_violation": predictions})
    df.to_csv(path, index=False)
