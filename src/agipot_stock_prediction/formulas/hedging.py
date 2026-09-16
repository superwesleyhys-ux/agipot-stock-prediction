from __future__ import annotations

from ._validation import validated_inputs

from typing import Mapping

from .contracts import FormulaResult, clamp, result_from_blockers, safe_div


@validated_inputs
def evaluate_hedging(
    *,
    portfolio_weights: Mapping[str, float],
    betas: Mapping[str, float],
    account_equity: float,
    target_beta: float,
    hedge_beta: float,
    hedge_instrument: str,
    approved_hedge_list: tuple[str, ...],
    hedge_spread_bps: float,
    hedge_median_adv: float,
    max_hedge_weight: float = 0.20,
    max_hedge_spread_bps: float = 10.0,
    min_hedge_adv: float = 25_000_000.0,
) -> FormulaResult:
    if account_equity <= 0 or max_hedge_weight < 0 or hedge_spread_bps < 0 or hedge_median_adv < 0:
        raise ValueError("hedging equity must be positive and limits/data nonnegative")
    portfolio_beta = sum(portfolio_weights.get(symbol, 0.0) * betas.get(symbol, 1.0) for symbol in portfolio_weights)
    excess_beta = max(0.0, portfolio_beta - target_beta)
    hedge_notional = safe_div(excess_beta * account_equity, abs(hedge_beta), default=0.0) if excess_beta > 0 else 0.0
    hedge_weight = safe_div(hedge_notional, account_equity, default=0.0)
    blockers: list[str] = []
    if not portfolio_weights or any(symbol not in betas for symbol in portfolio_weights):
        blockers.append("missing_portfolio_or_beta_data")
    if excess_beta > 0 and abs(hedge_beta) <= 1e-12:
        blockers.append("hedge_beta_cannot_be_zero")
    if excess_beta > 0:
        if hedge_weight > max_hedge_weight:
            blockers.append("hedge_weight_above_max")
        if hedge_instrument not in approved_hedge_list:
            blockers.append("hedge_instrument_not_approved")
        if hedge_spread_bps > max_hedge_spread_bps:
            blockers.append("hedge_spread_above_max")
        if hedge_median_adv < min_hedge_adv:
            blockers.append("hedge_adv_below_min")
    return result_from_blockers(
        score=1.0 - clamp(excess_beta, 0.0, 1.0) if blockers else 1.0,
        confidence=1.0,
        blockers=blockers,
        evidence={
            "portfolio_beta": portfolio_beta,
            "target_beta": target_beta,
            "excess_beta": excess_beta,
            "hedge_notional": hedge_notional,
            "hedge_weight": hedge_weight,
            "hedge_priority": ("reduce_gross", "raise_cash", "sell_high_beta", "approved_hedge_etf"),
            "options_enabled": False,
        },
    )


__all__ = ["evaluate_hedging"]
