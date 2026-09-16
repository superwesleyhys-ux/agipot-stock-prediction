from __future__ import annotations

import math
import statistics


def log_returns(closes: list[float]) -> list[float]:
    values: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        if not math.isfinite(previous) or not math.isfinite(current) or previous <= 0 or current <= 0:
            raise ValueError("close must be finite and positive")
        values.append(math.log(current) - math.log(previous))
    return values


def annualized_volatility(closes: list[float], lookback: int) -> float:
    daily = log_returns(closes[-(lookback + 1) :])
    if len(daily) < max(5, lookback // 2):
        return float("nan")
    return statistics.pstdev(daily) * math.sqrt(252)


def cap_and_redistribute(
    raw_weights: dict[str, float],
    cap: float,
    gross_target: float,
) -> dict[str, float]:
    if not math.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("cap must be finite and in (0, 1]")
    if not math.isfinite(gross_target) or not 0 <= gross_target <= 1:
        raise ValueError("gross_target must be finite and in [0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in raw_weights.values()):
        raise ValueError("raw weights must be finite and non-negative")
    if not raw_weights:
        return {}
    remaining = dict(raw_weights)
    output = {symbol: 0.0 for symbol in raw_weights}
    budget = gross_target
    for _ in range(len(raw_weights) + 1):
        if not remaining or budget <= 1e-12:
            break
        denominator = sum(max(value, 0.0) for value in remaining.values())
        if denominator <= 0:
            break
        capped: list[str] = []
        allocations: dict[str, float] = {}
        for symbol, value in remaining.items():
            allocation = budget * max(value, 0.0) / denominator
            if allocation >= cap:
                output[symbol] = cap
                budget -= cap
                capped.append(symbol)
            else:
                allocations[symbol] = allocation
        for symbol in capped:
            remaining.pop(symbol, None)
        if not capped:
            for symbol, allocation in allocations.items():
                output[symbol] = allocation
            break
    return {symbol: round(weight, 8) for symbol, weight in output.items() if weight > 1e-8}
