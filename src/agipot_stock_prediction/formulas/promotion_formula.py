from __future__ import annotations

from ._validation import validated_inputs

from typing import Sequence

from .modes import AutonomyMode, coerce_autonomy_mode

from .contracts import FormulaResult, result_from_blockers


@validated_inputs
def evaluate_promotion_formula(
    *,
    current_mode: AutonomyMode | str,
    target_mode: AutonomyMode | str,
    validation_manifest_pass: bool = False,
    shadow_days: int = 0,
    shadow_order_match_rate: float = 0.0,
    paper_days: int = 0,
    paper_reconcile_rate: float = 0.0,
    human_approval_days: int = 0,
    critical_incidents: int = 0,
    submission_unknown_unresolved: int = 0,
    all_risk_tests_pass: bool = False,
    activation_token_valid: bool = False,
    legal_account_owner_confirmed: bool = False,
    kill_switch: bool = False,
    broker_integrity_failure: bool = False,
    synthetic_demo: bool = False,
) -> FormulaResult:
    """Evaluate supplied research evidence only; this never activates any account."""
    if min(shadow_days, paper_days, human_approval_days, critical_incidents, submission_unknown_unresolved) < 0:
        raise ValueError("promotion day and incident counts cannot be negative")
    if not 0 <= shadow_order_match_rate <= 1 or not 0 <= paper_reconcile_rate <= 1:
        raise ValueError("promotion evidence rates must be in [0, 1]")
    current = coerce_autonomy_mode(current_mode)
    target = coerce_autonomy_mode(target_mode)
    blockers: list[str] = []
    next_mode = target.value
    if broker_integrity_failure:
        return result_from_blockers(
            score=0.0,
            confidence=1.0,
            blockers=("broker_integrity_failure",),
            evidence={"current_mode": current.value, "target_mode": target.value, "next_mode": AutonomyMode.LOCKDOWN.value},
        )
    if kill_switch:
        return result_from_blockers(
            score=0.0,
            confidence=1.0,
            blockers=("kill_switch_triggered",),
            evidence={"current_mode": current.value, "target_mode": target.value, "next_mode": AutonomyMode.REDUCE_ONLY.value},
        )
    if synthetic_demo and target != AutonomyMode.SHADOW:
        blockers.append("synthetic_demo_cannot_promote_beyond_shadow")
        next_mode = AutonomyMode.SHADOW.value
    if target == AutonomyMode.SHADOW and not validation_manifest_pass:
        blockers.append("validation_manifest_not_passed")
    elif target == AutonomyMode.PAPER:
        if shadow_days < 20:
            blockers.append("shadow_days_below_min")
        if shadow_order_match_rate < 0.95:
            blockers.append("shadow_order_match_rate_below_min")
        if critical_incidents:
            blockers.append("critical_incidents_present")
    elif target == AutonomyMode.HUMAN_APPROVAL:
        if paper_days < 20:
            blockers.append("paper_days_below_min")
        if paper_reconcile_rate < 0.98:
            blockers.append("paper_reconcile_rate_below_min")
        if submission_unknown_unresolved:
            blockers.append("submission_unknown_unresolved")
    elif target == AutonomyMode.LIMITED_AUTONOMY:
        if human_approval_days < 20:
            blockers.append("human_approval_days_below_min")
        if not all_risk_tests_pass:
            blockers.append("risk_tests_not_passed")
        if not activation_token_valid:
            blockers.append("activation_token_invalid")
        if not legal_account_owner_confirmed:
            blockers.append("legal_account_owner_not_confirmed")
    return result_from_blockers(
        score=1.0 if not blockers else 0.0,
        confidence=1.0,
        blockers=blockers,
        evidence={"current_mode": current.value, "target_mode": target.value, "next_mode": next_mode},
    )


@validated_inputs
def evaluate_trade_permission(
    *,
    mode: AutonomyMode | str,
    direction: str,
    data_integrity_pass: bool,
    universe_pass: bool = True,
    value_pass: bool = True,
    value_trap_pass: bool = True,
    cost_pass: bool = True,
    liquidity_pass: bool = True,
    risk_pass: bool = True,
    alpha: float = 0.0,
    min_alpha: float = 0.75,
    promotion_gate_allowed: bool = False,
    activation_token_valid: bool = False,
    no_unresolved_orders: bool = True,
    no_submission_unknown: bool = True,
    broker_review_pass: bool = False,
    owned_qty: float = 0.0,
    sell_qty: float = 0.0,
    quote_sanity_pass: bool = True,
    no_duplicate_economic_intent: bool = True,
) -> FormulaResult:
    """Return an offline policy score; it is not a broker execution permission."""
    mode_value = coerce_autonomy_mode(mode)
    side = direction.upper()
    blockers: list[str] = []
    if side == "BUY":
        allowed_modes = {AutonomyMode.PAPER, AutonomyMode.HUMAN_APPROVAL, AutonomyMode.LIMITED_AUTONOMY}
        checks: Sequence[tuple[str, bool]] = (
            ("data_integrity_failed", data_integrity_pass),
            ("universe_failed", universe_pass),
            ("value_failed", value_pass),
            ("value_trap_failed", value_trap_pass),
            ("cost_failed", cost_pass),
            ("liquidity_failed", liquidity_pass),
            ("risk_failed", risk_pass),
            ("alpha_below_threshold", alpha >= min_alpha),
            ("mode_not_buy_enabled", mode_value in allowed_modes),
            ("promotion_gate_not_allowed", promotion_gate_allowed),
            ("activation_token_invalid", activation_token_valid),
            ("unresolved_orders_present", no_unresolved_orders),
            ("submission_unknown_present", no_submission_unknown),
            ("broker_review_failed", broker_review_pass),
        )
    elif side == "SELL":
        sell_enabled_modes = {AutonomyMode.PAPER, AutonomyMode.HUMAN_APPROVAL, AutonomyMode.LIMITED_AUTONOMY, AutonomyMode.REDUCE_ONLY}
        checks = (
            ("mode_not_live_sell_enabled", mode_value in sell_enabled_modes),
            ("data_integrity_failed", data_integrity_pass),
            ("no_owned_qty", owned_qty > 0),
            ("sell_qty_not_positive", sell_qty > 0),
            ("sell_qty_exceeds_owned_qty", sell_qty <= owned_qty),
            ("quote_sanity_failed", quote_sanity_pass),
            ("duplicate_economic_intent", no_duplicate_economic_intent),
        )
    else:
        raise ValueError("direction must be BUY or SELL")
    blockers.extend(name for name, passed in checks if not passed)
    return result_from_blockers(
        score=1.0 if not blockers else 0.0,
        confidence=1.0,
        blockers=blockers,
        evidence={"mode": mode_value.value, "direction": side, "alpha": alpha, "min_alpha": min_alpha},
    )


__all__ = ["evaluate_promotion_formula", "evaluate_trade_permission"]
