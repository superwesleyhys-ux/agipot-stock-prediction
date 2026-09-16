from __future__ import annotations

import math
import statistics
from zoneinfo import ZoneInfo

from .contracts import RawSkillOutput, SkillContext, SkillVersion

from .math_utils import annualized_volatility


class TrendFeatureSkill:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        lookbacks = tuple(int(value) for value in context.binding.config.get("lookbacks", (20, 60, 120)))
        skip_recent = int(context.binding.config.get("skip_recent_days", 5))
        vol_lookback = int(context.binding.config.get("volatility_lookback", 20))
        if not lookbacks or any(value <= 1 or value > 2520 for value in lookbacks):
            raise ValueError("lookbacks must be in [2, 2520]")
        if not 0 <= skip_recent <= 60:
            raise ValueError("skip_recent_days must be in [0, 60]")
        as_of_date = context.snapshot.as_of.astimezone(ZoneInfo("America/New_York")).date()
        output: dict[str, dict[str, float]] = {}
        blockers: dict[str, list[str]] = {}
        universe = tuple(context.state.get("universe") or context.snapshot.universe)
        for symbol in universe:
            completed = [
                bar
                for bar in sorted(
                    context.snapshot.histories.get(symbol, ()), key=lambda bar: bar.timestamp
                )
                if bar.timestamp.astimezone(ZoneInfo("America/New_York")).date() < as_of_date
            ]
            endpoint = len(completed) - 1 - skip_recent
            reasons: list[str] = []
            if endpoint < 0:
                feature_bars = []
                reasons.append("skip window exceeds completed history")
            else:
                feature_bars = completed[: endpoint + 1]
            closes = [bar.close for bar in feature_bars]
            vol = annualized_volatility(closes, vol_lookback) if closes else float("nan")
            dollar_values = [bar.close * bar.volume for bar in feature_bars[-20:]]
            dollar_volume = statistics.median(dollar_values) if dollar_values else 0.0
            metrics: dict[str, float] = {
                "annualized_volatility": vol,
                "median_daily_dollar_volume": dollar_volume,
                "latest_close": feature_bars[-1].close if feature_bars else float("nan"),
                "history_days": float(len(completed)),
            }
            feature_endpoint = len(closes) - 1
            for lookback in lookbacks:
                start = feature_endpoint - lookback
                if start < 0:
                    reasons.append(f"missing {lookback}-day lookback")
                    continue
                metrics[f"return_{lookback}"] = closes[feature_endpoint] / closes[start] - 1
            if not math.isfinite(vol) or vol <= 0:
                reasons.append("volatility estimate unavailable")
            # Raw NaN values cannot cross a Skill boundary. Remove unavailable
            # metrics and preserve the reason instead.
            output[symbol] = {
                name: float(value)
                for name, value in metrics.items()
                if math.isfinite(value)
            }
            if reasons:
                blockers[symbol] = list(dict.fromkeys(reasons))
        confidence = (
            sum(1 for symbol in universe if symbol not in blockers) / len(universe)
            if universe
            else 0.0
        )
        return RawSkillOutput(
            result_type="feature_observations",
            payload={"features": output, "blocking_reasons": blockers},
            confidence=confidence,
            reasons=tuple(
                f"{symbol}: {reason}"
                for symbol, reasons in sorted(blockers.items())
                for reason in reasons
            ),
        )
