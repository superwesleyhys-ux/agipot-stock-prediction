from __future__ import annotations

from ._validation import validated_inputs

import math
from typing import Mapping

from .contracts import FormulaResult, clamp, result_from_blockers
from agipot_stock_prediction.research.contracts import finite_number


def _softmax_capped(reliabilities: Mapping[str, float], *, temperature: float, max_weight: float) -> dict[str, float]:
    """Water-fill a softmax onto the simplex while preserving the hard cap.

    Saturated skills stay fixed as the remaining probability is redistributed.
    Recomputing shifted logits also handles very small positive temperatures.
    """
    temp = finite_number(temperature, "ensemble.temperature")
    cap = finite_number(max_weight, "ensemble.max_skill_weight")
    if temp <= 0:
        raise ValueError("ensemble.temperature must be positive")
    if not 0 < cap <= 1:
        raise ValueError("ensemble.max_skill_weight must be in (0, 1]")
    if not reliabilities:
        return {}
    if len(reliabilities) * cap < 1.0:
        raise ValueError("ensemble cap is infeasible: number of skills * max_skill_weight must be >= 1")
    pending = {key: clamp(value, 0.0, 1.0) for key, value in reliabilities.items()}
    weights: dict[str, float] = {}
    remaining = 1.0
    while pending:
        largest = max(pending.values())
        exp_values = {key: math.exp((value - largest) / temp) for key, value in pending.items()}
        total = sum(exp_values.values())
        proposed = {key: remaining * value / total for key, value in exp_values.items()}
        saturated = [key for key, weight in proposed.items() if weight > cap]
        if not saturated:
            weights.update(proposed)
            break
        for key in saturated:
            weights[key] = cap
            del pending[key]
        remaining = max(0.0, 1.0 - sum(weights.values()))
    return weights


@validated_inputs
def evaluate_ensemble(
    *,
    skill_scores: Mapping[str, float],
    skill_reliabilities: Mapping[str, float],
    pass_threshold: float = 0.70,
    temperature: float = 0.25,
    max_skill_weight: float = 0.40,
) -> FormulaResult:
    blockers: list[str] = []
    if not 0 <= pass_threshold <= 1:
        raise ValueError("ensemble.pass_threshold must be in [0, 1]")
    if not skill_scores:
        blockers.append("no_skill_scores")
    if any(key not in skill_reliabilities for key in skill_scores):
        blockers.append("missing_skill_reliabilities")
    reliabilities = {key: skill_reliabilities.get(key, 0.0) for key in skill_scores}
    weights = _softmax_capped(reliabilities, temperature=temperature, max_weight=max_skill_weight)
    score = sum(clamp(skill_scores[key], 0.0, 1.0) * weights.get(key, 0.0) for key in skill_scores)
    confidence = sum(clamp(reliabilities[key], 0.0, 1.0) * weights.get(key, 0.0) for key in reliabilities)
    if score < pass_threshold:
        blockers.append("ensemble_score_below_threshold")
    return result_from_blockers(
        score=score,
        confidence=confidence,
        blockers=blockers,
        evidence={"weights": weights, "skill_scores": dict(skill_scores), "skill_reliabilities": reliabilities},
    )


__all__ = ["evaluate_ensemble"]
