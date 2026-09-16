from __future__ import annotations

from ._validation import validated_inputs

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, clamp, result_from_blockers


@validated_inputs
def evaluate_intrinsic_value(
    *,
    price: float,
    normalized_owner_earnings_per_share: float,
    risk_free_rate: float,
    equity_risk_premium: float,
    historical_oe_growth: float,
    analyst_growth: float,
    reinvestment_rate: float,
    roic: float,
    max_allowed_growth: float = 0.08,
    terminal_growth: float = 0.02,
    leverage_penalty: float = 0.0,
    cyclicality_penalty: float = 0.0,
    data_uncertainty_penalty: float = 0.0,
    years: int = 10,
    min_margin_of_safety: float = 0.30,
) -> FormulaResult:
    if years < 1:
        raise ValueError("intrinsic_value.years must be positive")
    if min_margin_of_safety <= 0:
        raise ValueError("intrinsic_value.min_margin_of_safety must be positive")
    if terminal_growth <= -1:
        raise ValueError("intrinsic_value.terminal_growth must exceed -1")
    px = finite_number(price, "intrinsic_value.price")
    oe = finite_number(normalized_owner_earnings_per_share, "intrinsic_value.normalized_owner_earnings_per_share")
    growth = min(
        finite_number(historical_oe_growth, "intrinsic_value.historical_oe_growth"),
        finite_number(analyst_growth, "intrinsic_value.analyst_growth"),
        finite_number(reinvestment_rate, "intrinsic_value.reinvestment_rate") * finite_number(roic, "intrinsic_value.roic"),
        finite_number(max_allowed_growth, "intrinsic_value.max_allowed_growth"),
    )
    discount_rate = (
        finite_number(risk_free_rate, "intrinsic_value.risk_free_rate")
        + finite_number(equity_risk_premium, "intrinsic_value.equity_risk_premium")
        + finite_number(leverage_penalty, "intrinsic_value.leverage_penalty")
        + finite_number(cyclicality_penalty, "intrinsic_value.cyclicality_penalty")
        + finite_number(data_uncertainty_penalty, "intrinsic_value.data_uncertainty_penalty")
    )
    blockers: list[str] = []
    if growth <= -1 or discount_rate <= -1:
        blockers.append("growth_or_discount_rate_outside_domain")
    if px <= 0 or oe <= 0:
        blockers.append("price_or_owner_earnings_not_positive")
    if discount_rate <= terminal_growth:
        blockers.append("discount_rate_not_above_terminal_growth")
    intrinsic = 0.0
    terminal_value = 0.0
    if not blockers:
        for year in range(1, years + 1):
            future_oe = oe * (1.0 + growth) ** year
            intrinsic += future_oe / (1.0 + discount_rate) ** year
        terminal_oe = oe * (1.0 + growth) ** years * (1.0 + terminal_growth)
        terminal_value = terminal_oe / (discount_rate - terminal_growth)
        intrinsic += terminal_value / (1.0 + discount_rate) ** years
    margin_of_safety = intrinsic / px - 1.0 if px > 0 else -1.0
    if margin_of_safety < min_margin_of_safety:
        blockers.append("margin_of_safety_below_min")
    if intrinsic <= px:
        blockers.append("intrinsic_value_not_above_price")
    return result_from_blockers(
        score=clamp(margin_of_safety / max(min_margin_of_safety, 1e-12), 0.0, 1.0),
        confidence=clamp(1.0 - data_uncertainty_penalty, 0.0, 1.0),
        blockers=blockers,
        evidence={
            "intrinsic_value_low": intrinsic,
            "terminal_value": terminal_value,
            "margin_of_safety": margin_of_safety,
            "growth": growth,
            "discount_rate": discount_rate,
        },
    )


__all__ = ["evaluate_intrinsic_value"]
