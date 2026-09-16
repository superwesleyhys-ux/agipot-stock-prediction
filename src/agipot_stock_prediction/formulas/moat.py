from __future__ import annotations

from ._validation import validated_inputs

import statistics
from typing import Sequence

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, clamp, result_from_blockers, safe_div, sigmoid


def _persistence(values: Sequence[float]) -> float:
    series = tuple(float(item) for item in values)
    if len(series) < 2:
        return 0.0
    mean = statistics.mean(series)
    std = statistics.pstdev(series)
    return clamp(1.0 - safe_div(std, abs(mean), default=1.0), 0.0, 1.0)


@validated_inputs
def evaluate_moat(
    *,
    gross_margin_history: Sequence[float],
    roic_history: Sequence[float],
    revenue_history: Sequence[float],
    gross_margin_trend: float = 0.0,
    revenue_growth: float = 0.0,
    unit_volume_growth: float = 0.0,
    reinvestment_runway: float = 0.5,
    low_customer_concentration: float = 0.5,
    fundamental_coverage: float = 1.0,
    required_fundamental_coverage: float = 0.70,
    proxy_mode_score: float = 0.40,
    min_moat_score: float = 0.50,
) -> FormulaResult:
    warnings: list[str] = []
    coverage = finite_number(fundamental_coverage, "moat.fundamental_coverage")
    if coverage < required_fundamental_coverage:
        score = clamp(proxy_mode_score, 0.0, 1.0)
        warnings.append("price_proxy_used")
    else:
        margin_persistence = _persistence(gross_margin_history)
        roic_series = tuple(float(item) for item in roic_history)
        roic_persistence = clamp((statistics.median(roic_series) - statistics.pstdev(roic_series)) / 0.20, 0.0, 1.0) if roic_series else 0.0
        revenue_stability = _persistence(revenue_history)
        pricing_power = sigmoid((gross_margin_trend + revenue_growth - unit_volume_growth) * 10.0)
        score = (
            0.25 * margin_persistence
            + 0.20 * roic_persistence
            + 0.20 * revenue_stability
            + 0.15 * pricing_power
            + 0.10 * clamp(reinvestment_runway, 0.0, 1.0)
            + 0.10 * clamp(low_customer_concentration, 0.0, 1.0)
        )
    blockers = [] if score >= min_moat_score else ["moat_below_threshold"]
    if coverage < required_fundamental_coverage:
        blockers.append("insufficient_fundamental_coverage")
    if min(len(gross_margin_history), len(roic_history), len(revenue_history)) < 2:
        blockers.append("insufficient_fundamental_history")
    return result_from_blockers(
        score=score,
        confidence=clamp(coverage, 0.0, 1.0),
        blockers=blockers,
        warnings=warnings,
        evidence={"fundamental_coverage": coverage, "required_fundamental_coverage": required_fundamental_coverage},
    )


__all__ = ["evaluate_moat"]
