from __future__ import annotations

from ._validation import validated_inputs

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, clamp, result_from_blockers


@validated_inputs
def evaluate_sizing(
    *,
    alpha: float,
    reliability_weighted_confidence: float,
    data_confidence: float,
    liquidity_multiplier: float,
    regime_multiplier: float,
    margin_of_safety: float,
    target_margin_of_safety: float,
    quality_score: float,
    permanent_loss_risk: float,
    expected_excess_return: float,
    variance: float,
    max_symbol_weight: float,
    liquidity_capacity_weight: float,
    kelly_fraction: float = 0.15,
    min_target_weight: float = 0.0001,
) -> FormulaResult:
    if variance <= 0 or target_margin_of_safety <= 0:
        raise ValueError("sizing variance and target_margin_of_safety must be positive")
    if min(max_symbol_weight, liquidity_capacity_weight, kelly_fraction, min_target_weight) < 0:
        raise ValueError("sizing weight limits and fractions cannot be negative")
    if max(max_symbol_weight, liquidity_capacity_weight, kelly_fraction) > 1:
        raise ValueError("sizing weight limits and kelly_fraction cannot exceed 1")
    raw_weight = (
        finite_number(max_symbol_weight, "sizing.max_symbol_weight")
        * clamp(alpha, 0.0, 1.0)
        * clamp(reliability_weighted_confidence, 0.0, 1.0)
        * clamp(data_confidence, 0.0, 1.0)
        * clamp(liquidity_multiplier, 0.0, 1.0)
        * clamp(regime_multiplier, 0.0, 1.5)
    )
    mos_multiplier = clamp(margin_of_safety / max(target_margin_of_safety, 1e-12), 0.0, 1.5)
    permanent_loss_multiplier = clamp(1.0 - permanent_loss_risk, 0.0, 1.0)
    formula_weight = raw_weight * mos_multiplier * clamp(quality_score, 0.0, 1.0) * permanent_loss_multiplier
    kelly_weight = kelly_fraction * max(0.0, expected_excess_return) / max(variance, 1e-12)
    target_weight = min(formula_weight, kelly_weight, max_symbol_weight, liquidity_capacity_weight)
    blockers = [] if target_weight >= min_target_weight else ["target_weight_below_min"]
    return result_from_blockers(
        score=clamp(target_weight / max(max_symbol_weight, 1e-12), 0.0, 1.0),
        confidence=min(clamp(reliability_weighted_confidence, 0.0, 1.0), clamp(data_confidence, 0.0, 1.0)),
        blockers=blockers,
        evidence={
            "raw_weight": raw_weight,
            "mos_multiplier": mos_multiplier,
            "permanent_loss_multiplier": permanent_loss_multiplier,
            "formula_weight": formula_weight,
            "kelly_weight": kelly_weight,
            "target_weight": target_weight,
        },
    )


__all__ = ["evaluate_sizing"]
