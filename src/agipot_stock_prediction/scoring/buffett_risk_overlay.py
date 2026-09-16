from __future__ import annotations

import math

from .contracts import RawSkillOutput, SkillContext, SkillVersion
from .factor_utils import coalesce

from .factor_utils import clamp, feature_value, rounded


class BuffettRiskOverlaySkill:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        min_quality = float(context.binding.config.get("min_quality_proxy", 0.55))
        min_margin = float(context.binding.config.get("min_margin_of_safety_proxy", 0.40))
        max_average_volatility = float(context.binding.config.get("max_average_annualized_volatility", 0.55))
        min_multiplier = float(context.binding.config.get("min_multiplier", 0.25))
        if not all(math.isfinite(value) for value in (min_quality, min_margin, max_average_volatility, min_multiplier)):
            raise ValueError("Buffett risk overlay thresholds must be finite")
        if not 0 <= min_multiplier <= 1:
            raise ValueError("min_multiplier must be in [0, 1]")

        weights = {symbol: float(value) for symbol, value in context.state.get("weights", {}).items()}
        features_by_symbol = context.state.get("features", {})
        checks: list[dict[str, object]] = []
        if not weights:
            checks.append(
                {
                    "rule_id": "buffett_overlay_no_weights",
                    "disposition": "PASS",
                    "immutable": False,
                    "reason": "no target weights require Buffett overlay scaling",
                }
            )
            return RawSkillOutput(
                result_type="risk_overlay",
                payload={"checks": checks, "blocked": False, "multiplier": 1.0},
                confidence=1.0,
            )

        gross = sum(max(value, 0.0) for value in weights.values())
        denominator = gross if gross > 0 else 1.0

        def weighted_feature(name: str, default: float) -> float:
            total = 0.0
            for symbol, weight in weights.items():
                total += max(weight, 0.0) * float(
                    coalesce(feature_value(dict(features_by_symbol.get(symbol, {})), (name,), default), default)
                )
            return total / denominator

        average_quality = weighted_feature("quality_proxy", 0.5)
        average_margin = weighted_feature("margin_of_safety_proxy", 0.5)
        average_volatility = weighted_feature("annualized_volatility", 0.35)
        multiplier = 1.0

        def add(rule_id: str, passed: bool, reason: str, *, scale: float = 1.0) -> None:
            nonlocal multiplier
            if not passed:
                multiplier *= clamp(scale, min_multiplier, 1.0)
            checks.append(
                {
                    "rule_id": rule_id,
                    "disposition": "PASS" if passed else "SCALE",
                    "immutable": False,
                    "reason": reason,
                }
            )

        add(
            "buffett_quality_floor",
            average_quality >= min_quality,
            f"weighted average quality proxy {average_quality:.4f} must meet {min_quality:.4f}",
            scale=(average_quality / min_quality if min_quality > 0 else 1.0),
        )
        add(
            "buffett_margin_of_safety_floor",
            average_margin >= min_margin,
            f"weighted average margin-of-safety proxy {average_margin:.4f} must meet {min_margin:.4f}",
            scale=(average_margin / min_margin if min_margin > 0 else 1.0),
        )
        add(
            "buffett_average_volatility_ceiling",
            average_volatility <= max_average_volatility,
            f"weighted average volatility {average_volatility:.4f} must not exceed {max_average_volatility:.4f}",
            scale=(max_average_volatility / average_volatility if average_volatility > 0 else 1.0),
        )

        multiplier = clamp(multiplier, 0.0, 1.0)
        return RawSkillOutput(
            result_type="risk_overlay",
            payload={
                "checks": checks,
                "blocked": False,
                "multiplier": rounded(multiplier),
                "average_quality_proxy": rounded(average_quality),
                "average_margin_of_safety_proxy": rounded(average_margin),
                "average_annualized_volatility": rounded(average_volatility),
            },
            confidence=1.0,
            reasons=tuple(str(check["reason"]) for check in checks if check["disposition"] == "SCALE"),
        )
