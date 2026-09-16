"""Shared lazy imports and explicit errors for optional frameworks."""

from __future__ import annotations

import importlib
import math
import os
from typing import Any


class IntegrationUnavailable(ImportError):
    """An optional dependency or its import-time prerequisites are unavailable."""


def require(module: str, package: str | None = None) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise IntegrationUnavailable(
            f"Optional integration {module!r} is unavailable. Install {package or module!r} "
            f"in an isolated environment; import failed: {exc}"
        ) from exc


def disable_telemetry() -> None:
    """Opt out before importing telemetry-enabled SDKs; do not touch credentials."""
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["GX_ANALYTICS_ENABLED"] = "False"
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"


def finite_values(values: Any, *, name: str, minimum: int = 1) -> list[float]:
    result = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain finite numbers, not booleans")
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(value)
    if len(result) < minimum:
        raise ValueError(f"{name} requires at least {minimum} observations")
    return result
