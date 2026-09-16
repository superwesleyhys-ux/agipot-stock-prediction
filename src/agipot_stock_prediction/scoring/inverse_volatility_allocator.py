from __future__ import annotations

import math

from .contracts import RawSkillOutput, SkillContext, SkillVersion

from .math_utils import cap_and_redistribute


class InverseVolatilityAllocator:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        gross_target = float(context.binding.config.get("gross_target", 0.70))
        cap = float(context.binding.config.get("max_symbol_weight", 0.05))
        if not math.isfinite(gross_target) or not 0 < gross_target <= 1:
            raise ValueError("gross_target must be in (0, 1]")
        if not math.isfinite(cap) or not 0 < cap <= 0.25:
            raise ValueError("max_symbol_weight must be in (0, 0.25]")
        raw: dict[str, float] = {}
        for symbol, signal in context.state.get("signals", {}).items():
            if not signal.get("eligible"):
                continue
            score = float(signal.get("score", 0.0))
            volatility = float(signal.get("annualized_volatility", 0.0))
            if score > 0 and volatility > 0 and math.isfinite(score) and math.isfinite(volatility):
                raw[symbol] = score / volatility
        weights = cap_and_redistribute(raw, cap, gross_target)
        invested = sum(weights.values())
        return RawSkillOutput(
            result_type="target_portfolio",
            payload={
                "weights": weights,
                "cash_weight": round(max(0.0, 1.0 - invested), 8),
                "gross_target": gross_target,
                "max_symbol_weight": cap,
            },
            confidence=1.0 if weights else 0.0,
            reasons=() if weights else ("no eligible positive sleeves to allocate",),
        )
