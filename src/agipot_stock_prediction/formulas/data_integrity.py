from __future__ import annotations

from ._validation import validated_inputs

import math
from typing import Sequence

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, clamp, result_from_blockers


@validated_inputs
def evaluate_data_integrity(
    *,
    point_in_time: bool,
    quote_age_seconds: float,
    max_quote_age_seconds: float,
    prices: Sequence[float],
    quantities: Sequence[float],
    account_id_match: bool,
    agentic_account: bool,
) -> FormulaResult:
    blockers: list[str] = []
    checks: dict[str, bool] = {
        "point_in_time": bool(point_in_time),
        "account_id_match": bool(account_id_match),
        "agentic_account": bool(agentic_account),
    }
    quote_age = finite_number(quote_age_seconds, "data_integrity.quote_age_seconds")
    max_age = finite_number(max_quote_age_seconds, "data_integrity.max_quote_age_seconds")
    checks["quote_age_nonnegative"] = quote_age >= 0 and max_age >= 0
    checks["quote_fresh"] = quote_age <= max_age
    checks["prices_present"] = bool(prices)
    checks["quantities_present"] = bool(quantities)
    checks["matching_data_lengths"] = len(prices) == len(quantities)
    checks["all_prices_finite"] = all(math.isfinite(value) and value > 0 for value in prices)
    checks["all_quantities_finite"] = all(math.isfinite(value) and value >= 0 for value in quantities)
    for name, passed in checks.items():
        if not passed:
            blockers.append(name)
    score = sum(1 for passed in checks.values() if passed) / len(checks)
    return result_from_blockers(
        score=score,
        confidence=clamp(score, 0.0, 1.0),
        blockers=blockers,
        evidence={"checks": checks, "quote_age_seconds": quote_age, "max_quote_age_seconds": max_age},
    )


__all__ = ["evaluate_data_integrity"]
