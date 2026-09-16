"""Minimal numeric contracts extracted from AGIPOT research contracts."""

from __future__ import annotations

import math


class ResearchDataError(ValueError):
    """A required research input is missing or invalid."""


def finite_number(value: float | int, field_name: str) -> float:
    if isinstance(value, bool):
        raise ResearchDataError(f"{field_name} must be numeric, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResearchDataError(f"{field_name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ResearchDataError(f"{field_name} must be finite")
    return numeric


__all__ = ["ResearchDataError", "finite_number"]
