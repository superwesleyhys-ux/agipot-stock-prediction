from __future__ import annotations

from ._validation import validated_inputs


from .contracts import FormulaResult, clamp, result_from_blockers, sigmoid


@validated_inputs
def evaluate_general_sleeve(
    *,
    circle_pass: bool,
    margin_of_safety: float,
    quality_score: float,
    moat_score: float,
    fcf_yield: float,
    leverage_risk: float,
    cyclicality_risk: float,
    permanent_loss_risk: float,
    min_general_score: float = 0.65,
    min_margin_of_safety: float = 0.30,
    max_permanent_loss_risk: float = 0.35,
) -> FormulaResult:
    raw = (
        0.35 * margin_of_safety
        + 0.25 * quality_score
        + 0.15 * moat_score
        + 0.10 * fcf_yield
        - 0.10 * leverage_risk
        - 0.05 * cyclicality_risk
    )
    score = sigmoid(raw * 4.0) if circle_pass else 0.0
    blockers: list[str] = []
    if not circle_pass:
        blockers.append("outside_circle_of_competence")
    if margin_of_safety < min_margin_of_safety:
        blockers.append("margin_of_safety_below_general_min")
    if permanent_loss_risk > max_permanent_loss_risk:
        blockers.append("permanent_loss_risk_above_general_max")
    if score < min_general_score:
        blockers.append("general_score_below_threshold")
    return result_from_blockers(
        score=score,
        confidence=clamp((quality_score + moat_score) / 2.0, 0.0, 1.0),
        blockers=blockers,
        evidence={"general_raw": raw, "permanent_loss_risk": permanent_loss_risk},
    )


@validated_inputs
def evaluate_workout_sleeve(
    *,
    probability_close: float,
    deal_spread: float,
    break_loss: float,
    cost: float,
    days_to_close: int,
    liquidity_pass: bool,
    source_verified: bool,
    min_event_probability: float = 0.75,
    max_days: int = 180,
    max_break_loss: float = 0.25,
) -> FormulaResult:
    if days_to_close <= 0 or max_days <= 0:
        raise ValueError("workout day counts must be positive")
    if break_loss < 0 or cost < 0:
        raise ValueError("workout loss and cost cannot be negative")
    p_close = clamp(probability_close, 0.0, 1.0)
    days = max(1, int(days_to_close))
    expected_return = p_close * deal_spread - (1.0 - p_close) * break_loss - cost
    annualized = (1.0 + expected_return) ** (252.0 / days) - 1.0 if expected_return > -0.99 else -1.0
    blockers: list[str] = []
    if p_close < min_event_probability:
        blockers.append("event_probability_below_min")
    if days > max_days:
        blockers.append("days_to_close_above_max")
    if break_loss > max_break_loss:
        blockers.append("break_loss_above_max")
    if not liquidity_pass:
        blockers.append("liquidity_failed")
    if not source_verified:
        blockers.append("source_not_verified")
    return result_from_blockers(
        score=clamp(annualized, 0.0, 1.0),
        confidence=1.0 if source_verified else 0.25,
        blockers=blockers,
        evidence={"expected_return": expected_return, "annualized_return": annualized, "days_to_close": days},
    )


@validated_inputs
def evaluate_controls_sleeve(
    *,
    controls_enabled: bool = False,
    insider_ownership_high: bool = False,
    buyback_authorized: bool = False,
    capital_allocation_improving: bool = False,
    activist_13d_verified: bool = False,
) -> FormulaResult:
    proxy_score = sum(
        1
        for item in (
            insider_ownership_high,
            buyback_authorized,
            capital_allocation_improving,
            activist_13d_verified,
        )
        if item
    ) / 4.0
    blockers = [] if controls_enabled else ["controls_disabled_for_retail_autonomy"]
    return result_from_blockers(
        score=proxy_score if controls_enabled else 0.0,
        confidence=0.25 if not controls_enabled else 0.75,
        blockers=blockers,
        warnings=("activist_proxy_not_true_control",) if proxy_score else (),
        evidence={"control_proxy_score": proxy_score},
    )


__all__ = ["evaluate_controls_sleeve", "evaluate_general_sleeve", "evaluate_workout_sleeve"]
