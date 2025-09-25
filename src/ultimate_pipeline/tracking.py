"""Experiment tracking shim with local, MLflow and Weights & Biases backends."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class Tracker:
    """Minimal interface for logging parameters, metrics and artifacts."""

    def log_params(self, params: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def log_artifact(self, artifact: Any, name: Optional[str] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default noop
        return None


class NoOpTracker(Tracker):
    """Tracker backend that silently discards all logging requests."""

    def log_params(self, params: Dict[str, Any]) -> None:  # pragma: no cover - intentionally empty
        return None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:  # pragma: no cover
        return None

    def log_artifact(self, artifact: Any, name: Optional[str] = None) -> None:  # pragma: no cover
        return None


class LocalTracker(Tracker):
    """Persist parameters and metrics to the local run directory."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.params_path = self.run_dir / "params.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.artifacts_dir = self.run_dir / "tracked_artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: list[Dict[str, Any]] = []

    def log_params(self, params: Dict[str, Any]) -> None:
        serialisable = _serialise(params)
        with self.params_path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {"step": step}
        payload.update({key: float(value) for key, value in metrics.items()})
        self._metrics.append(payload)
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(self._metrics, handle, indent=2)

    def log_artifact(self, artifact: Any, name: Optional[str] = None) -> None:
        if artifact is None:
            return
        if hasattr(artifact, "savefig"):
            filename = name or "figure.png"
            path = self.artifacts_dir / filename
            artifact.savefig(path)
            return
        source = Path(artifact)
        if not source.exists():  # pragma: no cover - defensive branch
            LOGGER.warning("Artifact path %s does not exist; skipping.", source)
            return
        destination = self.artifacts_dir / (name or source.name)
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


class MLflowTracker(Tracker):
    """Wrapper around MLflow's logging API."""

    def __init__(self, run_name: str, tracker_uri: Optional[str] = None) -> None:
        import mlflow  # type: ignore

        if tracker_uri:
            mlflow.set_tracking_uri(tracker_uri)
        self._mlflow = mlflow
        self._run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params(_serialise(params))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self._mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, artifact: Any, name: Optional[str] = None) -> None:
        if hasattr(artifact, "savefig"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / (name or "figure.png")
                artifact.savefig(path)
                self._mlflow.log_artifact(path, artifact_path=name)
            return
        path = Path(artifact)
        if path.exists():
            self._mlflow.log_artifact(str(path), artifact_path=name)

    def close(self) -> None:
        self._mlflow.end_run()


class WeightsAndBiasesTracker(Tracker):
    """Thin adapter for Weights & Biases logging."""

    def __init__(self, run_name: str, project: Optional[str] = None) -> None:
        import wandb  # type: ignore

        self._wandb = wandb
        self._run = wandb.init(project=project or "ultimate-pipeline", name=run_name, reinit=True)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._run.config.update(_serialise(params), allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        payload = {key: float(value) for key, value in metrics.items()}
        if step is not None:
            payload["step"] = step
        self._wandb.log(payload)

    def log_artifact(self, artifact: Any, name: Optional[str] = None) -> None:
        if hasattr(artifact, "savefig"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / (name or "figure.png")
                artifact.savefig(path)
                self._wandb.save(str(path))
            return
        path = Path(artifact)
        if path.exists():
            self._wandb.save(str(path))

    def close(self) -> None:
        self._run.finish()


def build_tracker(
    kind: Optional[str],
    *,
    run_id: str,
    artifacts_dir: Path,
    tracker_uri: Optional[str] = None,
    project: Optional[str] = None,
) -> Tracker:
    """Factory returning a tracker instance based on ``kind``."""

    target = (kind or "local").lower()
    if target in {"none", "off"}:
        return NoOpTracker()
    if target in {"local", "offline"}:
        return LocalTracker(artifacts_dir)
    if target == "mlflow":
        try:
            return MLflowTracker(run_name=run_id, tracker_uri=tracker_uri)
        except Exception as exc:  # pragma: no cover - optional dependency missing
            LOGGER.warning("Falling back to LocalTracker because MLflow initialisation failed: %s", exc)
            return LocalTracker(artifacts_dir)
    if target in {"wandb", "weights_and_biases"}:
        try:
            return WeightsAndBiasesTracker(run_name=run_id, project=project)
        except Exception as exc:  # pragma: no cover - optional dependency missing
            LOGGER.warning("Falling back to LocalTracker because W&B initialisation failed: %s", exc)
            return LocalTracker(artifacts_dir)
    LOGGER.warning("Unknown tracker '%s'; using LocalTracker.", kind)
    return LocalTracker(artifacts_dir)


def _serialise(data: Dict[str, Any]) -> Dict[str, Any]:
    serialised: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialised[key] = value
        elif isinstance(value, Path):
            serialised[key] = str(value)
        elif isinstance(value, (list, tuple, set)):
            serialised[key] = [_maybe_serialise(item) for item in value]
        elif isinstance(value, dict):
            serialised[key] = {str(k): _maybe_serialise(v) for k, v in value.items()}
        else:
            serialised[key] = _maybe_serialise(value)
    return serialised


def _maybe_serialise(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return _serialise(value.__dict__)  # pragma: no cover - best-effort serialisation
    return str(value)
