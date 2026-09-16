from __future__ import annotations

from ._validation import validated_inputs

import statistics
from typing import Sequence

from .contracts import FormulaResult, clamp, result_from_blockers, safe_div, sigmoid


@validated_inputs
def evaluate_trend_confirmation(
    *,
    price_history: Sequence[float],
    min_trend_adj: float = 0.0,
    vol_floor: float = 0.10,
    ma_window: int = 200,
) -> FormulaResult:
    if ma_window < 1 or vol_floor <= 0:
        raise ValueError("trend moving-average window and volatility floor must be positive")
    prices = tuple(float(item) for item in price_history)
    if any(price <= 0 for price in prices):
        raise ValueError("trend prices must be positive")
    blockers: list[str] = []
    if len(prices) < 121:
        blockers.append("insufficient_price_history")
    current = prices[-1] if prices else 0.0
    r20 = safe_div(current, prices[-21], default=1.0) - 1.0 if len(prices) >= 21 else 0.0
    r60 = safe_div(current, prices[-61], default=1.0) - 1.0 if len(prices) >= 61 else 0.0
    r120 = safe_div(current, prices[-121], default=1.0) - 1.0 if len(prices) >= 121 else 0.0
    returns = [safe_div(prices[idx], prices[idx - 1], default=1.0) - 1.0 for idx in range(1, len(prices))]
    vol60 = statistics.pstdev(returns[-60:]) * (252.0**0.5) if len(returns) >= 2 else vol_floor
    trend = 0.25 * r20 + 0.35 * r60 + 0.40 * r120
    trend_adj = trend / max(vol60, vol_floor)
    ma_values = prices[-ma_window:] if len(prices) >= ma_window else prices
    ma = statistics.mean(ma_values) if ma_values else 0.0
    if trend_adj < min_trend_adj:
        blockers.append("trend_adj_below_threshold")
    if current <= ma:
        blockers.append("price_below_moving_average")
    return result_from_blockers(
        score=sigmoid(trend_adj),
        confidence=clamp(len(prices) / ma_window, 0.0, 1.0),
        blockers=blockers,
        evidence={"r20": r20, "r60": r60, "r120": r120, "vol60": vol60, "trend_adj": trend_adj, "moving_average": ma},
    )


__all__ = ["evaluate_trend_confirmation"]
