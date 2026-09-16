from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Mapping, Sequence

from agipot_stock_prediction.research.contracts import finite_number


EPSILON = 1e-12


@dataclass(frozen=True)
class FormulaResult:
    score: float
    confidence: float
    pass_gate: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", clamp(finite_number(self.score, "formula.score"), 0.0, 1.0))
        object.__setattr__(self, "confidence", clamp(finite_number(self.confidence, "formula.confidence"), 0.0, 1.0))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "pass_gate": self.pass_gate,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
        }


def clamp(value: float, low: float, high: float) -> float:
    numeric = finite_number(value, "formula.value")
    if low > high:
        raise ValueError("clamp low cannot exceed high")
    return max(low, min(high, numeric))


def sigmoid(value: float) -> float:
    x = finite_number(value, "formula.sigmoid")
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    top = finite_number(numerator, "formula.numerator")
    bottom = finite_number(denominator, "formula.denominator")
    if abs(bottom) <= EPSILON:
        return default
    return top / bottom


def finite_series(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    return tuple(finite_number(value, f"{field_name}[{idx}]") for idx, value in enumerate(values))


def percentile(values: Sequence[float], pct: float) -> float:
    series = sorted(finite_series(values, "formula.percentile"))
    if not series:
        return 0.0
    if len(series) == 1:
        return series[0]
    position = clamp(pct, 0.0, 100.0) / 100.0 * (len(series) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return series[lower]
    weight = position - lower
    return series[lower] * (1.0 - weight) + series[upper] * weight


def median(values: Sequence[float]) -> float:
    series = finite_series(values, "formula.median")
    return statistics.median(series) if series else 0.0


def max_drawdown(values: Sequence[float]) -> float:
    series = finite_series(values, "formula.max_drawdown")
    if not series:
        return 0.0
    peak = series[0]
    worst = 0.0
    for value in series:
        peak = max(peak, value)
        if peak > EPSILON:
            worst = min(worst, value / peak - 1.0)
    return abs(worst)


def weighted_average(pairs: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(max(0.0, finite_number(weight, "formula.weight")) for _, weight in pairs)
    if total_weight <= EPSILON:
        return 0.0
    return sum(finite_number(value, "formula.weighted_value") * max(0.0, weight) for value, weight in pairs) / total_weight


def result_from_blockers(
    *,
    score: float,
    confidence: float,
    blockers: Sequence[str],
    warnings: Sequence[str] = (),
    evidence: Mapping[str, Any] | None = None,
) -> FormulaResult:
    return FormulaResult(
        score=score,
        confidence=confidence,
        pass_gate=not tuple(blockers),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        evidence=dict(evidence or {}),
    )


__all__ = [
    "EPSILON",
    "FormulaResult",
    "clamp",
    "finite_series",
    "max_drawdown",
    "median",
    "percentile",
    "result_from_blockers",
    "safe_div",
    "sigmoid",
    "weighted_average",
]
