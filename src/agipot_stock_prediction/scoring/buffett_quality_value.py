from __future__ import annotations

import math

from .contracts import RawSkillOutput, SkillContext, SkillVersion
from .factor_utils import coalesce

from .factor_utils import clamp, feature_value, rounded, weighted_average


class BuffettQualityValueAlphaSkill:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        min_quality = float(context.binding.config.get("min_quality_proxy", 0.55))
        min_margin = float(context.binding.config.get("min_margin_of_safety_proxy", 0.40))
        min_score = float(context.binding.config.get("min_score", 0.52))
        max_volatility = float(context.binding.config.get("max_annualized_volatility", 0.65))
        max_drawdown = float(context.binding.config.get("max_drawdown_120", 0.45))
        minimum_dollar_volume = float(
            context.binding.config.get("minimum_median_daily_dollar_volume", 10_000_000.0)
        )
        if not all(math.isfinite(value) for value in (min_quality, min_margin, min_score, max_volatility, max_drawdown, minimum_dollar_volume)):
            raise ValueError("Buffett alpha thresholds must be finite")
        features_by_symbol = context.state.get("features", {})
        rejected = context.state.get("rejected", {})
        signals: dict[str, dict[str, object]] = {}

        for symbol in context.state.get("universe", ()):
            features = dict(features_by_symbol.get(symbol, {}))
            reasons = list(rejected.get(symbol, ()))
            quality = coalesce(feature_value(features, ("quality_proxy",), 0.0), 0.0)
            business_quality = coalesce(feature_value(features, ("business_quality_proxy",), quality), quality)
            financial_strength = coalesce(feature_value(features, ("financial_strength_proxy",), quality), quality)
            moat = coalesce(feature_value(features, ("moat_proxy",), quality), quality)
            valuation = coalesce(feature_value(features, ("valuation_proxy",), 0.5), 0.5)
            margin = coalesce(feature_value(features, ("margin_of_safety_proxy",), valuation), valuation)
            trend_consistency = coalesce(feature_value(features, ("trend_consistency",), 0.5), 0.5)
            volatility = coalesce(feature_value(features, ("annualized_volatility",), 0.35), 0.35)
            drawdown = coalesce(feature_value(features, ("drawdown_120",), 0.0), 0.0)
            dollar_volume = coalesce(feature_value(features, ("median_daily_dollar_volume",), 0.0), 0.0)
            coverage = coalesce(feature_value(features, ("external_fundamental_coverage",), 0.0), 0.0)

            score = coalesce(weighted_average(
                (
                    (quality, 0.26),
                    (business_quality, 0.16),
                    (financial_strength, 0.14),
                    (moat, 0.12),
                    (margin, 0.18),
                    (valuation, 0.08),
                    (trend_consistency, 0.06),
                )
            ), 0.0)
            score = clamp(score)

            if quality < min_quality:
                reasons.append("quality proxy below Buffett floor")
            if margin < min_margin:
                reasons.append("margin-of-safety proxy below floor")
            if volatility > max_volatility:
                reasons.append("annualized volatility above Buffett sleeve limit")
            if drawdown > max_drawdown:
                reasons.append("120-day drawdown above Buffett sleeve limit")
            if dollar_volume < minimum_dollar_volume:
                reasons.append("median daily dollar volume below Buffett sleeve threshold")
            if score < min_score:
                reasons.append("Buffett quality-value score below threshold")

            unique_reasons = list(dict.fromkeys(reasons))
            eligible = not unique_reasons
            confidence = clamp((0.58 + coverage * 0.32 + trend_consistency * 0.10) if eligible else 0.0)
            signals[str(symbol)] = {
                "score": rounded(score),
                "confidence": rounded(confidence),
                "eligible": eligible,
                "blocking_reasons": unique_reasons,
                "source": "buffett_quality_value_v1",
                "quality_proxy": rounded(quality),
                "business_quality_proxy": rounded(business_quality),
                "financial_strength_proxy": rounded(financial_strength),
                "moat_proxy": rounded(moat),
                "valuation_proxy": rounded(valuation),
                "margin_of_safety_proxy": rounded(margin),
                "trend_consistency": rounded(trend_consistency),
                "annualized_volatility": rounded(volatility),
                "drawdown_120": rounded(drawdown),
                "median_daily_dollar_volume": rounded(dollar_volume),
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
