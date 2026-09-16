from __future__ import annotations

from ._validation import validated_inputs

from .contracts import FormulaResult, result_from_blockers, safe_div


@validated_inputs
def evaluate_kill_switch(
    *,
    equity: float,
    start_of_day_equity: float,
    rolling_peak_equity: float,
    daily_loss_limit: float = 0.03,
    rolling_drawdown_limit: float = 0.10,
    submission_unknown: bool = False,
    reconcile_mismatch: bool = False,
) -> FormulaResult:
    daily_loss_pct = safe_div(equity - start_of_day_equity, start_of_day_equity, default=-1.0)
    rolling_drawdown_pct = safe_div(equity, rolling_peak_equity, default=0.0) - 1.0
    blockers: list[str] = []
    next_mode = "NORMAL"
    if daily_loss_pct <= -abs(daily_loss_limit):
        blockers.append("daily_loss_limit_breached")
        next_mode = "REDUCE_ONLY"
    if rolling_drawdown_pct <= -abs(rolling_drawdown_limit):
        blockers.append("rolling_drawdown_limit_breached")
        next_mode = "REDUCE_ONLY"
    if submission_unknown or reconcile_mismatch:
        blockers.append("broker_integrity_failure")
        next_mode = "LOCKDOWN"
    return result_from_blockers(
        score=1.0 if not blockers else 0.0,
        confidence=1.0,
        blockers=blockers,
        evidence={"daily_loss_pct": daily_loss_pct, "rolling_drawdown_pct": rolling_drawdown_pct, "next_mode": next_mode},
    )


__all__ = ["evaluate_kill_switch"]
