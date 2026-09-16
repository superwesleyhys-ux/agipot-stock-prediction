from __future__ import annotations

from ._validation import validated_inputs

import math
import statistics
from typing import Sequence

from .contracts import FormulaResult, clamp, result_from_blockers, sigmoid


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    lx = tuple(float(item) for item in left)
    ry = tuple(float(item) for item in right)
    lmean = statistics.mean(lx)
    rmean = statistics.mean(ry)
    cov = sum((a - lmean) * (b - rmean) for a, b in zip(lx, ry))
    lvar = sum((a - lmean) ** 2 for a in lx)
    rvar = sum((b - rmean) ** 2 for b in ry)
    if lvar <= 1e-12 or rvar <= 1e-12:
        return 0.0
    return cov / math.sqrt(lvar * rvar)


@validated_inputs
def evaluate_feedback(
    *,
    current_reliability: float,
    signal_values: Sequence[float],
    future_net_returns: Sequence[float],
    skill_drawdown_penalty: float,
    turnover_cost: float,
    rho: float = 0.80,
    a: float = 1.5,
    b: float = 1.0,
    c: float = 2.0,
    d: float = 1.0,
    min_observations: int = 10,
) -> FormulaResult:
    if min_observations < 2:
        raise ValueError("feedback.min_observations must be at least 2")
    if not 0 <= rho <= 1:
        raise ValueError("feedback.rho must be in [0, 1]")
    signals = tuple(float(item) for item in signal_values)
    returns = tuple(float(item) for item in future_net_returns)
    information_coefficient = _corr(signals, returns)
    if signals and len(signals) == len(returns):
        hit_rate = sum(1 for signal, ret in zip(signals, returns) if (signal >= 0) == (ret >= 0)) / len(signals)
    else:
        hit_rate = 0.0
    new_sample = sigmoid(a * information_coefficient + b * hit_rate - c * skill_drawdown_penalty - d * turnover_cost)
    updated = clamp(rho * current_reliability + (1.0 - rho) * new_sample, 0.0, 1.0)
    blockers = [] if len(signals) >= min_observations and len(signals) == len(returns) else ["insufficient_feedback_observations"]
    return result_from_blockers(
        score=updated,
        confidence=clamp(len(signals) / min_observations, 0.0, 1.0),
        blockers=blockers,
        evidence={"information_coefficient": information_coefficient, "hit_rate": hit_rate, "updated_reliability": updated},
    )


__all__ = ["evaluate_feedback"]
