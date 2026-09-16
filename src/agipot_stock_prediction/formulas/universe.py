from __future__ import annotations

from ._validation import validated_inputs

from typing import Sequence

from .contracts import FormulaResult, result_from_blockers
from .liquidity import evaluate_liquidity


@validated_inputs
def evaluate_universe(
    *,
    asset_type: str,
    allowed_asset_types: Sequence[str],
    sector: str,
    circle_of_competence: Sequence[str],
    business_model_known: bool,
    data_coverage: float,
    min_data_coverage: float,
    close: float,
    volume_history: Sequence[float],
    bid: float,
    ask: float,
    order_notional: float = 0.0,
) -> FormulaResult:
    blockers: list[str] = []
    allowed_assets = {item.upper() for item in allowed_asset_types}
    allowed_sectors = {item.upper() for item in circle_of_competence}
    if asset_type.upper() not in allowed_assets:
        blockers.append("asset_type_outside_policy")
    if sector.upper() not in allowed_sectors:
        blockers.append("outside_circle_of_competence")
    if not business_model_known:
        blockers.append("business_model_unknown")
    if data_coverage < min_data_coverage:
        blockers.append("data_coverage_below_min")
    liquidity = evaluate_liquidity(
        close=close,
        volume_history=volume_history,
        bid=bid,
        ask=ask,
        order_notional=order_notional,
    )
    blockers.extend(f"liquidity:{item}" for item in liquidity.blockers)
    checks = 4 + 1
    passed = checks - len(blockers)
    score = max(0.0, passed / checks) * liquidity.score
    return result_from_blockers(
        score=score,
        confidence=min(1.0, data_coverage) * liquidity.confidence,
        blockers=blockers,
        warnings=liquidity.warnings,
        evidence={
            "asset_type": asset_type,
            "sector": sector,
            "data_coverage": data_coverage,
            "liquidity": liquidity.canonical_payload(),
        },
    )


__all__ = ["evaluate_universe"]
