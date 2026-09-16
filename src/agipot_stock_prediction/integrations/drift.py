"""Optional offline Evidently distribution-drift report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._optional import disable_telemetry, require


def evidently_report(reference: Any, current: Any, *, output_html: str | Path | None = None) -> dict:
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current batches must be nonempty")
    if list(reference.columns) != list(current.columns):
        raise ValueError("reference and current columns must match in order")
    disable_telemetry()
    evidently = require("evidently")
    presets = require("evidently.presets")
    snapshot = evidently.Report([presets.DataDriftPreset()]).run(
        current_data=current.copy(), reference_data=reference.copy())
    if output_html is not None:
        destination = Path(output_html).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(destination))
    return {"framework": "evidently", "report": snapshot.dict(),
            "prediction_accuracy_verified": False, "point_in_time_verified": False}
