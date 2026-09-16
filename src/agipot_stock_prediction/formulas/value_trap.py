from __future__ import annotations

from ._validation import validated_inputs

from .contracts import FormulaResult, clamp, result_from_blockers


@validated_inputs
def evaluate_value_trap(
    *,
    negative_revision: float,
    margin_deterioration: float,
    debt_stress: float,
    dilution_risk: float,
    price_breakdown: float,
    accounting_risk: float,
    max_value_trap_risk: float = 0.45,
) -> FormulaResult:
    risk = (
        0.25 * clamp(negative_revision, 0.0, 1.0)
        + 0.20 * clamp(margin_deterioration, 0.0, 1.0)
        + 0.20 * clamp(debt_stress, 0.0, 1.0)
        + 0.15 * clamp(dilution_risk, 0.0, 1.0)
        + 0.10 * clamp(price_breakdown, 0.0, 1.0)
        + 0.10 * clamp(accounting_risk, 0.0, 1.0)
    )
    blockers = [] if risk <= max_value_trap_risk else ["value_trap_risk_above_max"]
    return result_from_blockers(
        score=1.0 - risk,
        confidence=1.0,
        blockers=blockers,
        evidence={"value_trap_risk": risk, "max_value_trap_risk": max_value_trap_risk},
    )


__all__ = ["evaluate_value_trap"]
