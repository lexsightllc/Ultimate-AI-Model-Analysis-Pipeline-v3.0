#!/usr/bin/env python3
"""
Convenience entrypoint for running the Ultimate Pipeline on Kaggle, with
reproducibility, audit, artifact hygiene, an inlined dashboard generator,
and a GUARANTEED submission.csv at the Kaggle root (/kaggle/working).

This script will ALWAYS write /kaggle/working/submission.csv (or ./submission.csv
when running off-Kaggle). If the pipeline didn't produce a submission file,
it will fall back to copying the competition's sample_submission.csv so that
Kaggle can at least find a valid file for scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict as dataclass_asdict, is_dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from enum import Enum

# Optional numpy support for JSON sanitization
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

# __file__ may be missing in some Kaggle contexts; fall back to CWD and resolve symlinks.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:  # pragma: no cover - environment dependent
    PROJECT_ROOT = Path.cwd().resolve()

KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_CODE_DATASET = Path("/kaggle/input/ultimate-ai-model-analysis-pipeline-v3-0")
DEFAULT_COMP_DATASET = Path("/kaggle/input/jigsaw-agile-community-rules")

# --------------------------- JSON sanitization ---------------------------

def _json_sanitize(obj: Any) -> Any:
    """Recursively convert arbitrary Python objects into JSON-serializable ones."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _json_sanitize(dataclass_asdict(obj))
    if _np is not None:
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return _json_sanitize(obj.tolist())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return _json_sanitize(obj.value)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)

# ------------------------------ Import plumbing ------------------------------

def _ensure_code_on_sys_path() -> list[str]:
    attempted: list[str] = []
    seen_attempted: set[str] = set()
    seen_sys_path: set[str] = set(sys.path)

    def _register_path(path: Path) -> None:
        path_str = str(path)
        if path_str not in seen_sys_path:
            sys.path.insert(0, path_str)
            seen_sys_path.add(path_str)

    candidates: list[Path] = []
    env_src = os.environ.get("ULTIMATE_PIPELINE_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    candidates.extend([PROJECT_ROOT, Path.cwd(), KAGGLE_WORKING, DEFAULT_CODE_DATASET])

    for candidate in candidates:
        resolved = str(candidate.resolve(strict=False))
        if resolved not in seen_attempted:
            attempted.append(resolved)
            seen_attempted.add(resolved)
        _register_path(candidate / "src")
        _register_path(candidate)
        if candidate.exists():
            for wheel in sorted(candidate.rglob("*.whl")):
                _register_path(wheel)
    return attempted

_ATTEMPTED_CODE_PATHS = _ensure_code_on_sys_path()

try:
    from ultimate_pipeline.config import AnalysisConfig, load_default_config
    from ultimate_pipeline.pipeline import AnalysisPipeline
    from ultimate_pipeline.data import DatasetBundle, load_custom_datasets
except ModuleNotFoundError as exc:
    if exc.name and not exc.name.startswith("ultimate_pipeline"):
        raise
    checked_locations = "\n  - " + "\n  - ".join(_ATTEMPTED_CODE_PATHS) if _ATTEMPTED_CODE_PATHS else ""
    message = (
        "Unable to import 'ultimate_pipeline'. Checked the following locations:"
        f"{checked_locations}\n"
        "Set the ULTIMATE_PIPELINE_SRC environment variable or attach the Ultimate Pipeline code dataset."
    )
    raise SystemExit(message) from exc

# ------------------------------ Paths & datasets ------------------------------

def _default_output_dir() -> Path:
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING / "ultimate_pipeline_runs"
    return PROJECT_ROOT / "runs"

def _infer_comp_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    inferred_train = DEFAULT_COMP_DATASET / "train.csv"
    inferred_test = DEFAULT_COMP_DATASET / "test.csv"
    inferred_sample = DEFAULT_COMP_DATASET / "sample_submission.csv"
    return (
        inferred_train if inferred_train.exists() else None,
        inferred_test if inferred_test.exists() else None,
        inferred_sample if inferred_sample.exists() else None,
    )

def _resolve_dataset_paths(config: AnalysisConfig, args: argparse.Namespace) -> Optional[DatasetBundle]:
    train_csv = Path(args.train_csv) if args.train_csv else None
    test_csv = Path(args.test_csv) if args.test_csv else None
    sample_csv = Path(args.sample_csv) if args.sample_csv else None

    if train_csv:
        return load_custom_datasets(config, train_path=train_csv, test_path=test_csv, sample_path=sample_csv)

    inferred_train, inferred_test, inferred_sample = _infer_comp_paths()
    if inferred_train and inferred_test:
        return load_custom_datasets(
            config,
            train_path=inferred_train,
            test_path=inferred_test,
            sample_path=inferred_sample,
        )
    return None

# ------------------------------ Repro seeding ------------------------------

def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

# ------------------------------ Git metadata ------------------------------

def _git_commit(root: Path) -> Optional[str]:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return None

# ------------------------------ Manifests ------------------------------

def _export_manifest(run_dir: Path, args: argparse.Namespace, overrides: Dict[str, object], config_obj: AnalysisConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "cli_args.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(vars(args)), f, indent=2, sort_keys=True)
    with open(run_dir / "cli_overrides.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(overrides), f, indent=2, sort_keys=True)

    effective: Any = None
    for attr in ("to_dict", "dict", "model_dump"):
        if hasattr(config_obj, attr):
            try:
                effective = getattr(config_obj, attr)()
                break
            except Exception:
                continue
    if effective is None:
        try:
            if is_dataclass(config_obj):
                effective = dataclass_asdict(config_obj)
            else:
                effective = vars(config_obj)
        except Exception:
            effective = str(config_obj)
    with open(run_dir / "config_effective.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(effective), f, indent=2, sort_keys=True)

    meta = {
        "project_root": PROJECT_ROOT,
        "kaggle_working_exists": KAGGLE_WORKING.exists(),
        "attempted_code_paths": _ATTEMPTED_CODE_PATHS,
        "git_commit": _git_commit(PROJECT_ROOT),
        "python_version": sys.version,
        "platform": sys.platform,
    }
    with open(run_dir / "environment_meta.json", "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(meta), f, indent=2, sort_keys=True)

# ------------------------------ Config build ------------------------------

def _build_config(args: argparse.Namespace) -> tuple[AnalysisConfig, Dict[str, object]]:
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
    if args.cv_strategy:
        overrides["cv_strategy"] = args.cv_strategy
    if args.seed is not None:
        overrides["random_state"] = args.seed
        overrides["global_seed"] = args.seed
    if args.output_dir:
        overrides["cache_dir"] = str(Path(args.output_dir))
    else:
        overrides["cache_dir"] = str(_default_output_dir())

    base_config: AnalysisConfig = AnalysisConfig.from_file(Path(args.config)) if args.config else load_default_config()
    return base_config.with_overrides(overrides), overrides

# ------------------------------ CLI ------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ultimate Pipeline on Kaggle datasets")
    parser.add_argument("--config", type=str, help="Path to custom YAML/JSON config file")
    parser.add_argument("--performance-mode", type=str, help="Performance profile override")
    parser.add_argument("--calibration", type=str, help="Calibration strategy override (e.g. 'isotonic','platt','none')")
    parser.add_argument("--normalizer", type=str, help="Text normalizer override")
    parser.add_argument("--vectorizer", type=str, help="Vectorizer override: 'tfidf', 'tfidf_char_wb', 'tfidf_word_char_union', 'hashing', 'cuml'")
    parser.add_argument("--n-splits", type=int, help="Number of CV splits to use")
    parser.add_argument("--max-features", type=int, help="Maximum TF-IDF features")
    parser.add_argument("--n-jobs", type=int, help="Parallel jobs for estimators", default=None)
    parser.add_argument("--cv-strategy", type=str, help="CV strategy override (e.g. 'StratifiedKFold', 'GroupKFold', 'TimeSeriesSplit')")
    parser.add_argument("--seed", type=int, help="Global random seed for reproducibility", default=None)
    parser.add_argument("--output-dir", type=str, help="Directory for pipeline artefacts")
    parser.add_argument("--train-csv", type=str, help="Explicit path to train.csv")
    parser.add_argument("--test-csv", type=str, help="Explicit path to test.csv")
    parser.add_argument("--sample-csv", type=str, help="Optional path to sample_submission.csv")
    # DEFAULT: ALWAYS write to Kaggle root (or project root)
    default_target_root = KAGGLE_WORKING if KAGGLE_WORKING.exists() else PROJECT_ROOT
    parser.add_argument(
        "--final-submission",
        type=str,
        default=str(default_target_root / "submission.csv"),
        help="Destination path for the submission.csv copy (defaults to Kaggle root).",
    )
    parser.add_argument("--copy-root-artifacts", action="store_true", help="Also copy oof_predictions.csv to Kaggle root if present")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    args, unknown = parser.parse_known_args(argv)
    noisy = [u for u in unknown if not (u == "-f" or u.endswith(".json"))]
    if noisy:
        print("Ignoring unrecognized arguments: " + " ".join(noisy), file=sys.stderr)
    return args

def _setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ------------------------------ Artifact helpers ------------------------------

def _read_predictions(oof_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not oof_csv.exists():
        return rows
    with open(oof_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({"row_id": r.get("row_id", r.get("id", "")), "rule_violation": float(r.get("rule_violation", r.get("prediction", "0")))})
            except Exception:
                rows.append({"row_id": r.get("row_id", r.get("id", "")), "rule_violation": r.get("rule_violation", r.get("prediction", ""))})
    return rows

def _read_feature_importance(fi_csv: Path, limit: int = 500) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not fi_csv.exists():
        return items
    with open(fi_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= limit:
                break
            items.append({k: r[k] for k in r})
    return items

def _find_submission_candidate(run_dir: Path, artifacts_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "submission.csv",
        artifacts_dir / "submission.csv",
        run_dir / "artifacts" / "submission.csv",
        run_dir / "reports" / "submission.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _fallback_sample_submission(args: argparse.Namespace) -> Optional[Path]:
    # Prefer user-provided sample
    if args.sample_csv and Path(args.sample_csv).exists():
        return Path(args.sample_csv)
    # Then competition's default
    _, _, inferred_sample = _infer_comp_paths()
    return inferred_sample

def _ensure_submission_at_root(run_dir: Path, artifacts_dir: Path, final_path: Path, args: argparse.Namespace) -> Path:
    """
    Ensure a submission.csv exists at final_path. Try pipeline-produced file first,
    else copy sample_submission.csv as a last resort (so Kaggle always finds a file).
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Prefer pipeline-produced submission
    src = _find_submission_candidate(run_dir, artifacts_dir)
    if src and src.exists():
        shutil.copy2(src, final_path)
        return final_path

    # 2) Fallback to sample_submission.csv
    sample = _fallback_sample_submission(args)
    if sample and sample.exists():
        shutil.copy2(sample, final_path)
        return final_path

    # 3) As a last resort, create an empty-but-valid CSV if test exists (not ideal for scoring, but unblocks Kaggle)
    _, inferred_test, inferred_sample = _infer_comp_paths()
    if inferred_test and inferred_test.exists():
        # Try to infer header from sample; if no sample, write two generic columns
        if inferred_sample and inferred_sample.exists():
            shutil.copy2(inferred_sample, final_path)
            return final_path
        # Minimal fallback: assume first column is an id named "row_id" and output zeros
        try:
            import pandas as pd  # type: ignore
            df_test = pd.read_csv(inferred_test)
            id_col = df_test.columns[0]
            out = pd.DataFrame({id_col: df_test[id_col], "rule_violation": 0.0})
            out.to_csv(final_path, index=False)
            return final_path
        except Exception:
            pass

    # If everything failed, write a stub with the expected header used by the pipeline
    with open(final_path, "w", encoding="utf-8", newline="") as f:
        f.write("row_id,rule_violation\n")
    return final_path

# ------------------------------ Dashboard ------------------------------

def generate_dashboard(metrics: List[Dict[str, Any]], feature_importance: List[Dict[str, Any]], predictions: List[Dict[str, Any]], dashboard_path: Path) -> None:
    """
    Writes a self-contained HTML dashboard. No Python placeholders in the HTML; JSON is directly inlined.
    Avoid f-strings to prevent conflicts with JavaScript `${...}` template literals.
    """
    metrics_json = json.dumps(metrics, ensure_ascii=False)
    preds_json = json.dumps(predictions, ensure_ascii=False)
    fi_json = json.dumps(feature_importance, ensure_ascii=False)

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ultimate AI Model Analysis Pipeline</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --muted:#666; --bg:#ffffff; --border:#ddd; --header:#f2f2f2; }
    body { font-family: Arial, sans-serif; margin: 2rem; background: var(--bg); color:#111; }
    h1 { margin-bottom: 0.5rem; }
    h2 { margin-top: 2rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { border: 1px solid var(--border); padding: 8px; text-align: center; }
    th { background-color: var(--header); }
    .muted { color: var(--muted); font-size: 0.9rem; }
    .btn { display:inline-block; padding:0.5rem 0.75rem; border:1px solid #ccc; border-radius:6px; text-decoration:none; color:#111; }
    .grid { display:grid; grid-template-columns: 1fr; gap: 1.5rem; }
    @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
    .card { border:1px solid var(--border); border-radius:10px; padding:1rem; }
    .kpi { display:flex; gap:1rem; flex-wrap:wrap; }
    .kpi .item { flex:1 1 140px; border:1px solid var(--border); border-radius:8px; padding:0.75rem; background:#fafafa; }
    .right { text-align:right; }
    .small { font-size: 0.85rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  </style>
</head>
<body>
  <h1>Cross-Validation Summary</h1>
  <div id="cv-kpis" class="kpi"></div>
  <div id="cv-summary" class="card"></div>

  <div class="grid">
    <div class="card">
      <h2>Predictions</h2>
      <p class="muted small">You can download the out-of-fold predictions for offline inspection.</p>
      <p><a id="downloadCsv" class="btn" href="#" download="oof_predictions.csv">Download CSV</a></p>
      <div id="preds-info" class="muted small"></div>
      <div id="preds-preview" class="mono small"></div>
    </div>
    <div class="card">
      <h2>Feature Importance</h2>
      <div id="fi-table"></div>
    </div>
  </div>

  <script>
    const METRICS = """ + metrics_json + """;
    const PREDICTIONS = """ + preds_json + """;
    const FEATURE_IMPORTANCE = """ + fi_json + """;

    function mean(arr) {
      if (!arr || arr.length === 0) return null;
      let s = 0; for (const x of arr) s += Number(x) || 0;
      return s / arr.length;
    }

    (function renderKPIs() {
      if (!Array.isArray(METRICS) || METRICS.length === 0) return;
      const aucs = METRICS.map(m => Number(m.auc)).filter(v => !Number.isNaN(v));
      const briers = METRICS.map(m => Number(m.brier)).filter(v => !Number.isNaN(v));
      const eces = METRICS.map(m => Number(m.ece)).filter(v => !Number.isNaN(v));
      const kpis = [
        {label: "Mean AUC", value: (mean(aucs) ?? 0).toFixed(4)},
        {label: "Mean Brier", value: (mean(briers) ?? 0).toFixed(4)},
        {label: "Mean ECE", value: (mean(eces) ?? 0).toFixed(4)},
        {label: "Folds", value: String(METRICS.length)}
      ];
      const el = document.getElementById('cv-kpis');
      el.innerHTML = kpis.map(k => `<div class="item"><div class="muted small">${k.label}</div><div class="right"><strong>${k.value}</strong></div></div>`).join('');
    })();

    (function renderCV() {
      const container = document.getElementById('cv-summary');
      if (!Array.isArray(METRICS) || METRICS.length === 0) {
        container.innerText = 'No CV metrics available.';
        return;
      }
      let thead = '<thead><tr><th>Fold</th><th>AUC</th><th>Brier</th><th>ECE</th></tr></thead>';
      let rows = METRICS.map((m, i) => {
        const auc = (m.auc !== undefined) ? Number(m.auc).toFixed(4) : '';
        const brier = (m.brier !== undefined) ? Number(m.brier).toFixed(4) : '';
        const ece = (m.ece !== undefined) ? Number(m.ece).toFixed(4) : '';
        return `<tr><td>${i+1}</td><td>${auc}</td><td>${brier}</td><td>${ece}</td></tr>`;
      }).join('');
      container.innerHTML = `<table>${thead}<tbody>${rows}</tbody></table>`;
    })();

    function predsToCSV(rows) {
      if (!Array.isArray(rows) || rows.length === 0) return 'row_id,rule_violation\n';
      let csv = 'row_id,rule_violation\n';
      for (const row of rows) {
        csv += `${row.row_id},${row.rule_violation}\n`;
      }
      return csv;
    }

    (function setupCSV() {
      const csv = predsToCSV(PREDICTIONS);
      const blob = new Blob([csv], {type: 'text/csv'});
      const url = URL.createObjectURL(blob);
      const a = document.getElementById('downloadCsv');
      a.href = url;
      const n = Array.isArray(PREDICTIONS) ? PREDICTIONS.length : 0;
      document.getElementById('preds-info').innerText = `Rows: ${n}`;
      const prev = PREDICTIONS.slice(0, 10).map(r => `${r.row_id},${r.rule_violation}`).join('\n');
      document.getElementById('preds-preview').innerText = prev || '(no rows)';
    })();

    (function renderFI() {
      const el = document.getElementById('fi-table');
      if (!Array.isArray(FEATURE_IMPORTANCE) || FEATURE_IMPORTANCE.length === 0) {
        el.innerHTML = '<span class="muted">No feature importance available.</span>';
        return;
      }
      const keys = Object.keys(FEATURE_IMPORTANCE[0]);
      let thead = '<thead><tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr></thead>';
      let rows = FEATURE_IMPORTANCE.slice(0, 100).map(obj => {
        return '<tr>' + keys.map(k => `<td>${obj[k]}</td>`).join('') + '</tr>';
      }).join('');
      el.innerHTML = `<table>${thead}<tbody>${rows}</tbody></table>`;
    })();
  </script>
</body>
</html>
"""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(html, encoding="utf-8")

# ------------------------------ Main ------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    _seed_everything(args.seed)

    config, overrides = _build_config(args)
    pipeline = AnalysisPipeline(config)
    bundle = _resolve_dataset_paths(config, args)

    logging.info("Starting pipeline run...")
    result = pipeline.run(bundle)
    run_dir = pipeline.artifacts.run_dir
    artifacts_dir = pipeline.artifacts.artifacts_dir
    logging.info("Run directory: %s", run_dir)

    _export_manifest(run_dir, args, overrides, config)

    # ALWAYS ensure Kaggle can find submission.csv at the root
    final_submission = Path(args.final_submission)
    final_path = _ensure_submission_at_root(run_dir, artifacts_dir, final_submission, args)
    logging.info("Final submission path: %s", final_path.resolve())

    if args.copy_root_artifacts:
        # Optionally copy OOF to root for convenience
        oof_src = artifacts_dir / "oof_predictions.csv"
        target_root = final_path.parent
        if oof_src.exists():
            try:
                shutil.copy2(oof_src, target_root / "oof_predictions.csv")
                logging.info("oof_predictions.csv copied to %s", (target_root / "oof_predictions.csv").resolve())
            except Exception as e:
                logging.warning("Could not copy oof_predictions.csv to root: %s", e)

    # Summaries
    if result.metrics:
        try:
            mean_auc = sum(m.auc for m in result.metrics) / len(result.metrics)
            last_auc = result.metrics[-1].auc
            print("Mean CV AUC:", round(mean_auc, 6))
            print("Last fold AUC:", round(last_auc, 6))
            with open(run_dir / "cv_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    _json_sanitize({"mean_auc": float(mean_auc), "last_fold_auc": float(last_auc), "n_folds": len(result.metrics)}),
                    f,
                    indent=2,
                    sort_keys=True,
                )
        except Exception as e:
            logging.warning("Could not summarize CV metrics: %s", e)

    # Build Dashboard (reads existing OOF and feature importance files)
    dashboard_path = run_dir / "reports" / "dashboard.html"
    preds = _read_predictions(artifacts_dir / "oof_predictions.csv")
    fi = _read_feature_importance(artifacts_dir / "feature_importance.csv")
    metrics_payload: List[Dict[str, Any]] = []
    if result.metrics:
        for idx, metric in enumerate(result.metrics, start=1):
            metrics_payload.append({
                "fold": idx,
                "auc": getattr(metric, "auc", None),
                "brier": getattr(metric, "brier", None),
                "ece": getattr(metric, "ece", None),
            })
    try:
        generate_dashboard(metrics_payload, fi, preds, dashboard_path)
        logging.info("Dashboard written to %s", dashboard_path.resolve())
    except Exception as e:
        logging.warning("Failed to generate dashboard: %s", e)

    # Copy dashboard to root if requested
    if args.copy_root_artifacts:
        try:
            target_root = final_path.parent
            root_dashboard = target_root / "dashboard.html"
            shutil.copy2(dashboard_path, root_dashboard)
            logging.info("Dashboard copied to %s", root_dashboard.resolve())
        except Exception as e:
            logging.warning("Could not copy dashboard to root: %s", e)

    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
