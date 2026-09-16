from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import finite_number


@dataclass(frozen=True)
class TransactionCostEstimate:
    notional: float
    commission: float
    spread_cost_bps: float
    impact_bps: float
    total_cost: float
    total_cost_bps: float


def estimate_transaction_cost(
    *,
    notional: float,
    spread_bps: float,
    median_daily_dollar_volume: float,
    commission: float = 0.0,
    spread_capture_ratio: float = 0.5,
    impact_coefficient: float = 10.0,
) -> TransactionCostEstimate:
    order_notional = finite_number(notional, "cost.notional")
    spread = finite_number(spread_bps, "cost.spread_bps")
    adv = finite_number(median_daily_dollar_volume, "cost.median_daily_dollar_volume")
    fixed_commission = finite_number(commission, "cost.commission")
    capture = finite_number(spread_capture_ratio, "cost.spread_capture_ratio")
    impact = finite_number(impact_coefficient, "cost.impact_coefficient")
    if order_notional <= 0:
        raise ValueError("cost.notional must be positive")
    if adv <= 0:
        raise ValueError("cost.median_daily_dollar_volume must be positive")
    if spread < 0 or fixed_commission < 0 or capture < 0 or impact < 0:
        raise ValueError("cost inputs cannot be negative")
    spread_cost_bps = spread * capture
    impact_bps = impact * math.sqrt(order_notional / adv)
    variable_cost = order_notional * (spread_cost_bps + impact_bps) / 10_000.0
    total_cost = fixed_commission + variable_cost
    return TransactionCostEstimate(
        notional=order_notional,
        commission=fixed_commission,
        spread_cost_bps=spread_cost_bps,
        impact_bps=impact_bps,
        total_cost=total_cost,
        total_cost_bps=total_cost / order_notional * 10_000.0,
    )


def apply_cost_to_return(*, gross_return: float, cost_bps: float) -> float:
    gross = finite_number(gross_return, "cost.gross_return")
    cost = finite_number(cost_bps, "cost.cost_bps")
    if cost < 0:
        raise ValueError("cost.cost_bps cannot be negative")
    return gross - cost / 10_000.0


__all__ = ["TransactionCostEstimate", "apply_cost_to_return", "estimate_transaction_cost"]
