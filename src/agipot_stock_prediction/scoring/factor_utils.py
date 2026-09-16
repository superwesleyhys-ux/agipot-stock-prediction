from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def coalesce(value: float | None, default: float) -> float:
    """Use the default only for missing values; a measured zero stays zero."""
    return default if value is None else value


def finite_float(value: object, default: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def feature_value(
    features: Mapping[str, object],
    names: Sequence[str],
    default: float | None = None,
) -> float | None:
    for name in names:
        value = finite_float(features.get(name))
        if value is not None:
            return value
    return default


def normalize_higher(
    value: float | None,
    low: float,
    high: float,
    *,
    default: float | None = None,
) -> float | None:
    if value is None or high <= low:
        return default
    return clamp((value - low) / (high - low))


def normalize_lower(
    value: float | None,
    good: float,
    bad: float,
    *,
    default: float | None = None,
) -> float | None:
    if value is None or bad <= good:
        return default
    return clamp((bad - value) / (bad - good))


def weighted_average(parts: Iterable[tuple[float | None, float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in parts:
        numeric = finite_float(value)
        if numeric is None or weight <= 0:
            continue
        numerator += numeric * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)
