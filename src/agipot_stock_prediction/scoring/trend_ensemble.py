from __future__ import annotations

import math
import statistics

from .contracts import RawSkillOutput, SkillContext, SkillVersion


class TrendEnsembleSkill:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        lookbacks = tuple(int(value) for value in context.binding.config.get("lookbacks", (20, 60, 120)))
        minimum_dollar_volume = float(
            context.binding.config.get("minimum_median_daily_dollar_volume", 50_000_000.0)
        )
        if minimum_dollar_volume < 0 or not math.isfinite(minimum_dollar_volume):
            raise ValueError("minimum dollar volume must be finite and non-negative")
        features_by_symbol = context.state.get("features", {})
        rejected = context.state.get("rejected", {})
        signals: dict[str, dict[str, object]] = {}
        for symbol in context.state.get("universe", ()):
            features = dict(features_by_symbol.get(symbol, {}))
            reasons = list(rejected.get(symbol, ()))
            vol = features.get("annualized_volatility")
            dollar_volume = float(features.get("median_daily_dollar_volume", 0.0))
            if vol is None or not math.isfinite(float(vol)) or float(vol) <= 0:
                reasons.append("volatility estimate unavailable")
            if dollar_volume < minimum_dollar_volume:
                reasons.append("median daily dollar volume below threshold")
            components: list[float] = []
            if vol is not None and math.isfinite(float(vol)) and float(vol) > 0:
                for lookback in lookbacks:
                    raw_return = features.get(f"return_{lookback}")
                    if raw_return is None:
                        reasons.append(f"missing {lookback}-day lookback")
                        continue
                    normalized = max(
                        -2.0,
                        min(
                            2.0,
                            float(raw_return)
                            / (float(vol) * math.sqrt(lookback / 252)),
                        ),
                    )
                    components.append(normalized)
            score = statistics.fmean(components) if components else None
            if score is None or not math.isfinite(score):
                reasons.append("signal score unavailable")
                score = 0.0
            unique_reasons = list(dict.fromkeys(reasons))
            eligible = not unique_reasons and score > 0
            signals[symbol] = {
                "score": float(score),
                "confidence": 1.0 if not unique_reasons else 0.0,
                "horizon_days": max(lookbacks),
                "annualized_volatility": float(vol or 0.0),
                "median_daily_dollar_volume": dollar_volume,
                "latest_close": float(features.get("latest_close", 0.0)),
                "history_days": int(features.get("history_days", 0)),
                "eligible": eligible,
                "blocking_reasons": unique_reasons,
            }
        eligible_count = sum(1 for signal in signals.values() if signal["eligible"])
        confidence = eligible_count / len(signals) if signals else 0.0
        return RawSkillOutput(
            result_type="alpha_signals",
            payload={"signals": signals},
            confidence=confidence,
            reasons=tuple(
                f"{symbol}: {reason}"
                for symbol, signal in sorted(signals.items())
                for reason in signal["blocking_reasons"]
            ),
        )
