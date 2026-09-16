from __future__ import annotations

import math

from .contracts import RawSkillOutput, SkillContext, SkillVersion

from .factor_utils import clamp, feature_value, rounded
from .math_utils import cap_and_redistribute


class ConvictionAllocatorSkill:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        gross_target = float(context.binding.config.get("gross_target", 0.60))
        cap = float(context.binding.config.get("max_symbol_weight", 0.10))
        score_floor = float(context.binding.config.get("score_floor", 0.0))
        volatility_adjusted = bool(context.binding.config.get("volatility_adjusted", True))
        min_volatility = float(context.binding.config.get("min_volatility", 0.08))
        if not math.isfinite(gross_target) or not 0 < gross_target <= 1:
            raise ValueError("gross_target must be in (0, 1]")
        if not math.isfinite(cap) or not 0 < cap <= 0.25:
            raise ValueError("max_symbol_weight must be in (0, 0.25]")
        if not math.isfinite(score_floor):
            raise ValueError("score_floor must be finite")
        if not math.isfinite(min_volatility) or not 0.01 <= min_volatility <= 1:
            raise ValueError("min_volatility must be in [0.01, 1]")

        regime_scalar = clamp(float(context.state.get("regime", {}).get("risk_scalar", 1.0)))
        effective_gross = gross_target * regime_scalar
        if effective_gross <= 1e-12:
            return RawSkillOutput(
                result_type="target_portfolio",
                payload={
                    "weights": {},
                    "cash_weight": 1.0,
                    "gross_target": gross_target,
                    "effective_gross_target": 0.0,
                    "max_symbol_weight": cap,
                },
                confidence=0.0,
                reasons=("regime risk scalar set gross target to zero",),
            )

        features_by_symbol = context.state.get("features", {})
        raw: dict[str, float] = {}
        for symbol, signal in context.state.get("signals", {}).items():
            if not signal.get("eligible"):
                continue
            score = float(signal.get("score", 0.0))
            confidence = clamp(float(signal.get("confidence", 0.0)))
            edge = max(0.0, score - score_floor)
            if edge <= 0 or confidence <= 0:
                continue
            raw_value = edge * confidence
            if volatility_adjusted:
                volatility = feature_value(
                    dict(signal),
                    ("annualized_volatility",),
                    feature_value(dict(features_by_symbol.get(symbol, {})), ("annualized_volatility",), min_volatility),
                )
                raw_value /= max(min_volatility, float(volatility or min_volatility))
            if math.isfinite(raw_value) and raw_value > 0:
                raw[str(symbol)] = raw_value

        weights = cap_and_redistribute(raw, cap, effective_gross)
        invested = sum(weights.values())
        return RawSkillOutput(
            result_type="target_portfolio",
            payload={
                "weights": weights,
                "cash_weight": round(max(0.0, 1.0 - invested), 8),
                "gross_target": gross_target,
                "effective_gross_target": rounded(effective_gross),
                "max_symbol_weight": cap,
                "raw_conviction": {symbol: rounded(value) for symbol, value in sorted(raw.items())},
            },
            confidence=1.0 if weights else 0.0,
            reasons=() if weights else ("no eligible conviction signals to allocate",),
        )
