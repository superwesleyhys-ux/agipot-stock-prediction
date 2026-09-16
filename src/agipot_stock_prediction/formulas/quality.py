from __future__ import annotations

from ._validation import validated_inputs

from typing import Sequence

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, clamp, max_drawdown, result_from_blockers, safe_div, sigmoid


def _bounded_positive(value: float, scale: float = 1.0) -> float:
    return sigmoid(finite_number(value, "quality.component") / max(scale, 1e-12))


@validated_inputs
def evaluate_quality(
    *,
    roic: float,
    gross_margin: float,
    operating_margin: float,
    owner_earnings_growth_history: Sequence[float],
    owner_earnings_history: Sequence[float],
    net_debt_to_ebitda: float,
    interest_coverage: float,
    current_ratio: float,
    fcf: float,
    net_income: float,
    roic_minus_wacc: float,
    buyback_yield_when_undervalued: float,
    shares_outstanding_current: float,
    shares_outstanding_3y_ago: float,
    min_quality_score: float = 0.55,
) -> FormulaResult:
    growth = tuple(float(item) for item in owner_earnings_growth_history)
    profitability = (
        _bounded_positive(roic, 0.15)
        + _bounded_positive(gross_margin, 0.50)
        + _bounded_positive(operating_margin, 0.25)
    ) / 3.0
    growth_vol = max(growth) - min(growth) if growth else 1.0
    earnings_stability = clamp(1.0 - growth_vol - max_drawdown(owner_earnings_history), 0.0, 1.0)
    balance_sheet_safety = (
        clamp(1.0 - safe_div(net_debt_to_ebitda, 4.0, default=1.0), 0.0, 1.0)
        + clamp(safe_div(interest_coverage, 10.0, default=0.0), 0.0, 1.0)
        + clamp(safe_div(current_ratio, 2.0, default=0.0), 0.0, 1.0)
    ) / 3.0
    fcf_conversion = clamp(safe_div(fcf, max(abs(net_income), 1e-12), default=0.0), 0.0, 1.0)
    dilution_penalty = max(0.0, safe_div(shares_outstanding_current, shares_outstanding_3y_ago, default=1.0) - 1.0)
    capital_allocation = clamp(0.5 + roic_minus_wacc * 2.0 + buyback_yield_when_undervalued - dilution_penalty, 0.0, 1.0)
    dilution_discipline = clamp(1.0 - dilution_penalty, 0.0, 1.0)
    score = (
        0.25 * profitability
        + 0.20 * earnings_stability
        + 0.20 * balance_sheet_safety
        + 0.15 * fcf_conversion
        + 0.10 * capital_allocation
        + 0.10 * dilution_discipline
    )
    blockers = [] if score >= min_quality_score else ["quality_below_threshold"]
    if not owner_earnings_history or not growth:
        blockers.append("missing_owner_earnings_history")
    if shares_outstanding_current <= 0 or shares_outstanding_3y_ago <= 0:
        blockers.append("shares_outstanding_not_positive")
    return result_from_blockers(
        score=score,
        confidence=clamp((len(owner_earnings_history) + len(growth)) / 10.0, 0.25, 1.0),
        blockers=blockers,
        evidence={
            "profitability": profitability,
            "earnings_stability": earnings_stability,
            "balance_sheet_safety": balance_sheet_safety,
            "fcf_conversion": fcf_conversion,
            "capital_allocation": capital_allocation,
            "dilution_discipline": dilution_discipline,
            "dilution_penalty": dilution_penalty,
        },
    )


__all__ = ["evaluate_quality"]
