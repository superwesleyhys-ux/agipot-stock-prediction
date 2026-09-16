from __future__ import annotations

from ._validation import validated_inputs

from .contracts import FormulaResult, clamp, result_from_blockers


@validated_inputs
def evaluate_add_reduce_exit(
    *,
    current_weight: float,
    target_weight: float,
    account_equity: float,
    alpha: float,
    prior_alpha: float,
    owned_qty: float,
    limit_price: float,
    drawdown_state: str = "NORMAL",
    liquidity_pass: bool = True,
    cost_pass: bool = True,
    no_unresolved_order: bool = True,
    post_entry_loss: float = 0.0,
    permanent_loss_risk: float = 0.0,
    thesis_invalidated: bool = False,
    data_integrity_failed: bool = False,
    security_no_longer_eligible: bool = False,
    rebalance_band: float = 0.0025,
    theta_add: float = 0.75,
    theta_reduce: float = 0.45,
    theta_exit: float = 0.25,
    max_alpha_decay: float = 0.10,
    max_add_per_run: float = 2_000.0,
    daily_turnover_remaining: float = 5_000.0,
    liquidity_capacity_remaining: float = 5_000.0,
    cash_available_after_floor: float = 5_000.0,
    hard_exit_risk: float = 0.80,
) -> FormulaResult:
    if account_equity <= 0 or limit_price <= 0 or owned_qty < 0:
        raise ValueError("position equity and price must be positive, quantity nonnegative")
    if min(current_weight, target_weight, max_add_per_run, daily_turnover_remaining,
           liquidity_capacity_remaining, cash_available_after_floor) < 0:
        raise ValueError("position weights and available capacities cannot be negative")
    blockers: list[str] = []
    action = "HOLD"
    target_gap = target_weight - current_weight
    exit_signal = (
        alpha < theta_exit
        or thesis_invalidated
        or permanent_loss_risk >= hard_exit_risk
        or data_integrity_failed
        or security_no_longer_eligible
        or post_entry_loss <= -0.12
    )
    if exit_signal and owned_qty > 0:
        action = "EXIT"
        notional = owned_qty * limit_price
    elif target_gap > rebalance_band and alpha >= theta_add:
        action = "ADD"
        if alpha - prior_alpha < -max_alpha_decay:
            blockers.append("alpha_decay_too_large")
        if drawdown_state != "NORMAL":
            blockers.append("drawdown_state_not_normal")
        if not liquidity_pass:
            blockers.append("liquidity_failed")
        if not cost_pass:
            blockers.append("cost_failed")
        if not no_unresolved_order:
            blockers.append("unresolved_order")
        if post_entry_loss <= -0.05:
            blockers.append("post_entry_loss_blocks_add")
        notional = min(
            account_equity * target_gap,
            max_add_per_run,
            daily_turnover_remaining,
            liquidity_capacity_remaining,
            cash_available_after_floor,
        )
    elif target_gap < -rebalance_band or alpha < theta_reduce or permanent_loss_risk > hard_exit_risk * 0.75:
        action = "REDUCE"
        notional = account_equity * max(0.0, current_weight - target_weight)
    else:
        notional = 0.0
    sell_qty = min(owned_qty, notional / max(limit_price, 1e-12)) if action in {"REDUCE", "EXIT"} else 0.0
    if action in {"REDUCE", "EXIT"} and sell_qty > owned_qty:
        blockers.append("sell_qty_exceeds_owned_qty")
    executable_notional = 0.0 if blockers else max(0.0, notional)
    return result_from_blockers(
        score=1.0 if action != "HOLD" else 0.5,
        confidence=clamp(alpha, 0.0, 1.0),
        blockers=blockers,
        evidence={
            "action": action,
            "notional": max(0.0, notional),
            "executable_notional": executable_notional,
            "sell_qty": sell_qty,
            "target_gap": target_gap,
        },
    )


__all__ = ["evaluate_add_reduce_exit"]
