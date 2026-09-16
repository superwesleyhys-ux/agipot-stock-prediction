from __future__ import annotations

from ._validation import validated_inputs

from typing import Sequence

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, clamp, median, result_from_blockers


@validated_inputs
def evaluate_owner_earnings(
    *,
    net_income: float,
    depreciation: float,
    amortization: float,
    other_non_cash_charges: float,
    total_capex: float,
    normalized_working_capital_need: float = 0.0,
    stock_compensation_adjustment: float = 0.0,
    maintenance_ratio: float | None = None,
    owner_earnings_history: Sequence[float] = (),
) -> FormulaResult:
    capex = max(0.0, finite_number(total_capex, "owner_earnings.total_capex"))
    dep = max(0.0, finite_number(depreciation, "owner_earnings.depreciation"))
    if maintenance_ratio is None:
        ratio = clamp(dep / max(capex, 1e-12), 0.3, 1.0) if capex > 0 else 1.0
    else:
        ratio = clamp(maintenance_ratio, 0.0, 1.0)
    maintenance_capex = capex * ratio
    owner_earnings = (
        finite_number(net_income, "owner_earnings.net_income")
        + dep
        + finite_number(amortization, "owner_earnings.amortization")
        + finite_number(other_non_cash_charges, "owner_earnings.other_non_cash_charges")
        - maintenance_capex
        - finite_number(normalized_working_capital_need, "owner_earnings.normalized_working_capital_need")
        - finite_number(stock_compensation_adjustment, "owner_earnings.stock_compensation_adjustment")
    )
    history = tuple(float(item) for item in owner_earnings_history)
    if history:
        ttm = owner_earnings
        med3 = median(history[-3:])
        med5 = median(history[-5:])
        normalized = 0.50 * ttm + 0.30 * med3 + 0.20 * med5
    else:
        normalized = owner_earnings
    blockers = [] if owner_earnings > 0 and normalized > 0 else ["owner_earnings_not_positive"]
    confidence = clamp((len(history) + 1) / 5.0, 0.25, 1.0)
    scale = max(abs(net_income), abs(owner_earnings), 1.0)
    return result_from_blockers(
        score=clamp(normalized / scale, 0.0, 1.0),
        confidence=confidence,
        blockers=blockers,
        evidence={
            "owner_earnings": owner_earnings,
            "normalized_owner_earnings": normalized,
            "maintenance_capex": maintenance_capex,
            "maintenance_ratio": ratio,
        },
    )


__all__ = ["evaluate_owner_earnings"]
