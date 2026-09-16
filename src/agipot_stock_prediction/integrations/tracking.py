"""Local-only MLflow tracking without global URI changes or a tracking server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ._optional import disable_telemetry, finite_values, require


def record_mlflow_run(
    evidence: Mapping[str, Any], metrics: Mapping[str, float], *,
    output_dir: str | Path = ".agipot/mlflow", experiment_name: str = "agipot_offline",
    run_name: str = "research", status: str = "FINISHED",
) -> dict[str, str]:
    """Persist manifest + metrics, including FAILED/KILLED status, to local SQLite.

    A complete manifest must come from ExperimentEvidenceHarness. Logging a run
    here alone is not proof that every trial was registered or data was OOS.
    """
    if status not in {"FINISHED", "FAILED", "KILLED"}:
        raise ValueError("status must be FINISHED, FAILED, or KILLED")
    root = Path(output_dir).expanduser().resolve()
    if "://" in str(output_dir):
        raise ValueError("output_dir must be a local path, not a tracking URI")
    payload = json.dumps(dict(evidence), sort_keys=True, allow_nan=False, indent=2)
    numbers = {key: finite_values([value], name=f"metrics.{key}")[0] for key, value in metrics.items()}
    disable_telemetry()
    client_module = require("mlflow.tracking", "mlflow (or mlflow-skinny + sqlalchemy)")
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{root / 'tracking.db'}"
    client = client_module.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        experiment_name, artifact_location=(root / "artifacts").as_uri())
    run = client.create_run(experiment_id, tags={"mlflow.runName": run_name, "agipot.evidence": "research_only"})
    run_id = run.info.run_id
    try:
        client.log_dict(run_id, json.loads(payload), "experiment_evidence.json")
        for key, value in numbers.items():
            client.log_metric(run_id, key, value)
        client.set_terminated(run_id, status=status)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    return {"run_id": run_id, "tracking_uri": uri, "status": status, "evidence_level": "caller_supplied"}
