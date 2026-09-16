from __future__ import annotations

import math

from .contracts import RawSkillOutput, SkillContext, SkillVersion
from .factor_utils import coalesce

from .factor_utils import clamp, feature_value, normalize_higher, rounded, weighted_average


class MomentumLeaderAlphaSkill:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        min_return_60 = float(context.binding.config.get("min_return_60", 0.02))
        min_return_120 = float(context.binding.config.get("min_return_120", 0.04))
        min_score = float(context.binding.config.get("min_score", 0.50))
        max_volatility = float(context.binding.config.get("max_annualized_volatility", 0.80))
        max_drawdown = float(context.binding.config.get("max_drawdown_120", 0.35))
        minimum_dollar_volume = float(
            context.binding.config.get("minimum_median_daily_dollar_volume", 20_000_000.0)
        )
        if not all(math.isfinite(value) for value in (min_return_60, min_return_120, min_score, max_volatility, max_drawdown, minimum_dollar_volume)):
            raise ValueError("momentum thresholds must be finite")
        features_by_symbol = context.state.get("features", {})
        rejected = context.state.get("rejected", {})
        signals: dict[str, dict[str, object]] = {}

        for symbol in context.state.get("universe", ()):
            features = dict(features_by_symbol.get(symbol, {}))
            reasons = list(rejected.get(symbol, ()))
            return_20 = coalesce(feature_value(features, ("return_20",), 0.0), 0.0)
            return_60 = coalesce(feature_value(features, ("return_60",), 0.0), 0.0)
            return_120 = coalesce(feature_value(features, ("return_120",), 0.0), 0.0)
            volatility = coalesce(feature_value(features, ("annualized_volatility",), 0.40), 0.40)
            drawdown = coalesce(feature_value(features, ("drawdown_120",), 0.0), 0.0)
            dollar_volume = coalesce(feature_value(features, ("median_daily_dollar_volume",), 0.0), 0.0)
            trend_consistency = coalesce(feature_value(features, ("trend_consistency",), 0.5), 0.5)
            quality = coalesce(feature_value(features, ("quality_proxy",), 0.5), 0.5)
            volatility_quality = coalesce(feature_value(features, ("volatility_quality",), 0.5), 0.5)
            drawdown_quality = coalesce(feature_value(features, ("drawdown_quality",), 0.5), 0.5)
            liquidity_quality = coalesce(feature_value(features, ("liquidity_quality",), 0.5), 0.5)

            momentum_20 = coalesce(normalize_higher(return_20, -0.03, 0.08, default=0.5), 0.5)
            momentum_60 = coalesce(normalize_higher(return_60, 0.00, 0.18, default=0.5), 0.5)
            momentum_120 = coalesce(normalize_higher(return_120, 0.02, 0.30, default=0.5), 0.5)
            overextension_penalty = clamp(max(0.0, return_20 - 0.16) / 0.16)
            score = coalesce(weighted_average(
                (
                    (momentum_120, 0.27),
                    (momentum_60, 0.25),
                    (momentum_20, 0.12),
                    (trend_consistency, 0.14),
                    (volatility_quality, 0.07),
                    (drawdown_quality, 0.07),
                    (liquidity_quality, 0.04),
                    (quality, 0.04),
                )
            ), 0.0)
            score = clamp(score - overextension_penalty * 0.12)

            if return_60 < min_return_60:
                reasons.append("60-day momentum below threshold")
            if return_120 < min_return_120:
                reasons.append("120-day momentum below threshold")
            if return_20 < -0.06:
                reasons.append("short-term momentum reversal")
            if volatility > max_volatility:
                reasons.append("annualized volatility above momentum sleeve limit")
            if drawdown > max_drawdown:
                reasons.append("120-day drawdown above momentum sleeve limit")
            if dollar_volume < minimum_dollar_volume:
                reasons.append("median daily dollar volume below momentum threshold")
            if score < min_score:
                reasons.append("momentum leader score below threshold")

            unique_reasons = list(dict.fromkeys(reasons))
            eligible = not unique_reasons
            confidence = clamp((0.72 + trend_consistency * 0.18 + quality * 0.10) if eligible else 0.0)
            signals[str(symbol)] = {
                "score": rounded(score),
                "confidence": rounded(confidence),
                "eligible": eligible,
                "blocking_reasons": unique_reasons,
                "source": "momentum_leader_v1",
                "return_20": rounded(return_20),
                "return_60": rounded(return_60),
                "return_120": rounded(return_120),
                "trend_consistency": rounded(trend_consistency),
                "annualized_volatility": rounded(volatility),
                "drawdown_120": rounded(drawdown),
                "median_daily_dollar_volume": rounded(dollar_volume),
                "quality_proxy": rounded(quality),
            }
        eligible_count = sum(1 for signal in signals.values() if signal["eligible"])
        return RawSkillOutput(
            result_type="alpha_signals",
            payload={"signals": signals},
            confidence=eligible_count / len(signals) if signals else 0.0,
            reasons=tuple(
                f"{symbol}: {reason}"
                for symbol, signal in sorted(signals.items())
                for reason in signal["blocking_reasons"]
            ),
        )
