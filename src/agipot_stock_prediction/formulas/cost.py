from __future__ import annotations

from ._validation import validated_inputs

from agipot_stock_prediction.research.cost_model import estimate_transaction_cost

from .contracts import FormulaResult, clamp, result_from_blockers


@validated_inputs
def evaluate_cost_gate(
    *,
    expected_return: float,
    order_notional: float,
    spread_bps: float,
    median_daily_dollar_volume: float,
    commission: float = 0.0,
    spread_capture_ratio: float = 0.5,
    impact_coefficient: float = 10.0,
    min_net_expected_return: float = 0.005,
    min_edge_to_cost: float = 2.0,
) -> FormulaResult:
    estimate = estimate_transaction_cost(
        notional=order_notional,
        spread_bps=spread_bps,
        median_daily_dollar_volume=median_daily_dollar_volume,
        commission=commission,
        spread_capture_ratio=spread_capture_ratio,
        impact_coefficient=impact_coefficient,
    )
    cost_return = estimate.total_cost / estimate.notional
    net_expected_return = expected_return - cost_return
    edge_to_cost = expected_return / max(cost_return, 1e-12)
    blockers: list[str] = []
    if net_expected_return < min_net_expected_return:
        blockers.append("net_expected_return_below_min")
    if edge_to_cost < min_edge_to_cost:
        blockers.append("edge_to_cost_below_min")
    return result_from_blockers(
        score=clamp(net_expected_return / max(min_net_expected_return, 1e-12), 0.0, 1.0),
        confidence=1.0,
        blockers=blockers,
        evidence={
            "expected_return": expected_return,
            "cost_return": cost_return,
            "net_expected_return": net_expected_return,
            "edge_to_cost": clamp(edge_to_cost, 0.0, 1_000_000.0),
            "transaction_cost": estimate.__dict__,
        },
    )


__all__ = ["evaluate_cost_gate"]
