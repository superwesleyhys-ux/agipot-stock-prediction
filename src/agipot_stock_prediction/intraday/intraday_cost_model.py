from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class IntradayCostConfig:
    spread_capture_ratio: float = 1.0
    impact_coefficient: float = 2.0
    fee_bps: float = 0.1
    min_spread_bps: float = 1.0
    min_impact_bps: float = 1.0
    min_edge_bps: float = 1.0
    min_edge_to_cost_ratio: float = 3.0


@dataclass(frozen=True)
class CostDecision:
    allowed: bool
    blockers: tuple[str, ...]
    spread_cost_bps: float
    impact_bps: float
    borrow_cost_bps: float
    total_cost_bps: float
    expected_edge_bps: float
    edge_to_cost_ratio: float


def compute_intraday_cost(
    *,
    predicted_return_bps: float,
    spread_bps: float,
    order_notional: float,
    dollar_volume_window: float,
    borrow_fee_annualized: float = 0.0,
    holding_days_equivalent: float = 1.0 / 252.0,
    config: IntradayCostConfig | None = None,
) -> CostDecision:
    cfg = config or IntradayCostConfig()
    values = {
        "predicted_return_bps": predicted_return_bps, "spread_bps": spread_bps,
        "order_notional": order_notional, "dollar_volume_window": dollar_volume_window,
        "borrow_fee_annualized": borrow_fee_annualized,
        "holding_days_equivalent": holding_days_equivalent,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg)},
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name != "predicted_return_bps" and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if order_notional <= 0 or dollar_volume_window <= 0:
        raise ValueError("order_notional and dollar_volume_window must be positive")
    blockers: list[str] = []
    if spread_bps <= 0:
        blockers.append("spread_zero_for_live_conclusion_blocked")
    if cfg.impact_coefficient <= 0:
        blockers.append("impact_zero_for_live_conclusion_blocked")
    effective_spread = max(spread_bps, cfg.min_spread_bps)
    spread_cost_bps = effective_spread * cfg.spread_capture_ratio
    impact_bps = max(cfg.min_impact_bps, cfg.impact_coefficient * math.sqrt(max(order_notional, 0.0) / max(dollar_volume_window, 1e-9)))
    borrow_cost_bps = max(0.0, borrow_fee_annualized) / 252.0 * holding_days_equivalent * 10_000.0
    total_cost_bps = spread_cost_bps + impact_bps + max(0.0, cfg.fee_bps) + borrow_cost_bps
    expected_edge_bps = predicted_return_bps - total_cost_bps
    edge_to_cost_ratio = predicted_return_bps / total_cost_bps if total_cost_bps > 0 else 0.0
    if expected_edge_bps < cfg.min_edge_bps:
        blockers.append("expected_edge_bps_below_min")
    if edge_to_cost_ratio < cfg.min_edge_to_cost_ratio:
        blockers.append("edge_to_cost_ratio_below_min")
    numeric = (spread_cost_bps, impact_bps, borrow_cost_bps, total_cost_bps, expected_edge_bps, edge_to_cost_ratio)
    if any(not math.isfinite(value) for value in numeric):
        blockers.append("non_finite_cost_metric")
    return CostDecision(
        allowed=not blockers,
        blockers=tuple(blockers),
        spread_cost_bps=spread_cost_bps,
        impact_bps=impact_bps,
        borrow_cost_bps=borrow_cost_bps,
        total_cost_bps=total_cost_bps,
        expected_edge_bps=expected_edge_bps if math.isfinite(expected_edge_bps) else 0.0,
        edge_to_cost_ratio=edge_to_cost_ratio if math.isfinite(edge_to_cost_ratio) else 0.0,
    )
