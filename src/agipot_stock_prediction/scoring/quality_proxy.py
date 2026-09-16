from __future__ import annotations

import math
from zoneinfo import ZoneInfo

from .contracts import RawSkillOutput, SkillContext, SkillVersion
from .factor_utils import coalesce

from .factor_utils import (
    clamp,
    feature_value,
    finite_float,
    normalize_higher,
    normalize_lower,
    rounded,
    weighted_average,
)


_EXTERNAL_QUALITY_KEYS = (
    "roic",
    "roe",
    "gross_margin",
    "operating_margin",
    "fcf_margin",
    "fcf_conversion",
    "gross_margin_stability",
    "earnings_stability",
    "revenue_stability",
    "debt_to_equity",
    "net_debt_to_fcf",
    "net_debt_to_ebitda",
    "interest_coverage",
    "share_dilution",
)


def _completed_closes(context: SkillContext, symbol: str) -> list[float]:
    as_of_date = context.snapshot.as_of.astimezone(ZoneInfo("America/New_York")).date()
    bars = sorted(context.snapshot.histories.get(symbol, ()), key=lambda bar: bar.timestamp)
    return [
        bar.close
        for bar in bars
        if bar.timestamp.astimezone(ZoneInfo("America/New_York")).date() < as_of_date
    ]


def _max_drawdown(closes: list[float]) -> float | None:
    if len(closes) < 2:
        return None
    peak = closes[0]
    worst = 0.0
    for close in closes:
        peak = max(peak, close)
        if peak > 0:
            worst = min(worst, close / peak - 1.0)
    return abs(worst)


def _positive_return_ratio(features: dict[str, float]) -> float:
    values = [
        finite_float(features.get(name))
        for name in ("return_20", "return_60", "return_120")
    ]
    available = [value for value in values if value is not None]
    if not available:
        return 0.5
    return sum(1 for value in available if value > 0) / len(available)


def _coverage_ratio(features: dict[str, float]) -> float:
    present = sum(1 for key in _EXTERNAL_QUALITY_KEYS if finite_float(features.get(key)) is not None)
    return present / len(_EXTERNAL_QUALITY_KEYS)


def _normalized_higher(features: dict[str, float], names: tuple[str, ...], low: float, high: float) -> float | None:
    return normalize_higher(feature_value(features, names), low, high)


def _normalized_lower(features: dict[str, float], names: tuple[str, ...], good: float, bad: float) -> float | None:
    return normalize_lower(feature_value(features, names), good, bad)


class QualityProxySkill:
    """Creates Buffett-style quality, moat, value, and safety proxies.

    The skill prefers licensed point-in-time fundamentals when the snapshot
    supplies them through external_features. When they are absent, it falls
    back to price-derived stability, drawdown, volatility, and liquidity
    proxies so the runtime graph remains testable and deterministic.
    """

    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        drawdown_lookback = int(context.binding.config.get("drawdown_lookback", 120))
        if not 20 <= drawdown_lookback <= 756:
            raise ValueError("drawdown_lookback must be in [20, 756]")
        features_by_symbol = context.state.get("features", {})
        output: dict[str, dict[str, float]] = {}
        confidence_values: list[float] = []
        blockers: dict[str, list[str]] = {}

        for symbol in context.state.get("universe", ()):
            features = {
                str(name): float(value)
                for name, value in dict(features_by_symbol.get(symbol, {})).items()
                if finite_float(value) is not None
            }
            closes = _completed_closes(context, str(symbol))
            drawdown = _max_drawdown(closes[-drawdown_lookback:])
            volatility = feature_value(features, ("annualized_volatility",), 0.35)
            dollar_volume = feature_value(features, ("median_daily_dollar_volume",), 0.0)
            trend_consistency = _positive_return_ratio(features)
            volatility_quality = coalesce(normalize_lower(volatility, 0.12, 0.60, default=0.5), 0.5)
            drawdown_quality = coalesce(normalize_lower(drawdown, 0.05, 0.40, default=0.5), 0.5)
            liquidity_quality = coalesce(normalize_higher(dollar_volume, 20_000_000.0, 250_000_000.0, default=0.5), 0.5)

            profitability = weighted_average(
                (
                    (_normalized_higher(features, ("roic",), 0.08, 0.25), 0.35),
                    (_normalized_higher(features, ("roe",), 0.10, 0.30), 0.15),
                    (_normalized_higher(features, ("fcf_margin",), 0.04, 0.18), 0.20),
                    (_normalized_higher(features, ("operating_margin",), 0.08, 0.25), 0.15),
                    (_normalized_higher(features, ("fcf_conversion",), 0.50, 1.00), 0.15),
                )
            )
            stability = weighted_average(
                (
                    (_normalized_higher(features, ("gross_margin_stability",), 0.40, 0.90), 0.30),
                    (_normalized_higher(features, ("earnings_stability",), 0.40, 0.90), 0.35),
                    (_normalized_higher(features, ("revenue_stability",), 0.40, 0.90), 0.20),
                    (volatility_quality, 0.15),
                )
            )
            financial_strength = weighted_average(
                (
                    (_normalized_lower(features, ("debt_to_equity",), 0.20, 2.00), 0.30),
                    (_normalized_lower(features, ("net_debt_to_fcf",), 0.00, 5.00), 0.25),
                    (_normalized_lower(features, ("net_debt_to_ebitda",), 0.00, 4.00), 0.20),
                    (_normalized_higher(features, ("interest_coverage",), 3.00, 15.00), 0.15),
                    (_normalized_lower(features, ("share_dilution",), 0.00, 0.08), 0.10),
                )
            )
            moat_proxy = weighted_average(
                (
                    (_normalized_higher(features, ("gross_margin",), 0.20, 0.55), 0.25),
                    (_normalized_higher(features, ("gross_margin_stability",), 0.40, 0.90), 0.25),
                    (_normalized_higher(features, ("roic",), 0.08, 0.25), 0.30),
                    (_normalized_higher(features, ("earnings_stability",), 0.40, 0.90), 0.20),
                )
            )
            price_quality = coalesce(weighted_average(
                (
                    (trend_consistency, 0.35),
                    (volatility_quality, 0.30),
                    (drawdown_quality, 0.25),
                    (liquidity_quality, 0.10),
                )
            ), 0.5)
            external_quality = weighted_average(
                (
                    (profitability, 0.38),
                    (stability, 0.27),
                    (financial_strength, 0.22),
                    (moat_proxy, 0.13),
                )
            )
            coverage = _coverage_ratio(dict(context.snapshot.external_features.get(symbol, {})))
            quality_proxy = coalesce(weighted_average(((external_quality, 0.65), (price_quality, 0.35)))
                if external_quality is not None
                else price_quality, 0.5)

            value_proxy = weighted_average(
                (
                    (_normalized_higher(features, ("owner_earnings_yield",), 0.03, 0.12), 0.30),
                    (_normalized_higher(features, ("fcf_yield",), 0.03, 0.10), 0.25),
                    (_normalized_higher(features, ("earnings_yield",), 0.03, 0.10), 0.15),
                    (_normalized_lower(features, ("price_to_fcf",), 12.0, 35.0), 0.15),
                    (_normalized_lower(features, ("pe_ratio",), 12.0, 35.0), 0.10),
                    (_normalized_lower(features, ("ev_to_ebit",), 10.0, 25.0), 0.05),
                )
            )
            if value_proxy is None:
                value_proxy = 0.50
            explicit_margin = weighted_average(
                (
                    (_normalized_higher(features, ("margin_of_safety",), 0.00, 0.35), 0.55),
                    (_normalized_higher(features, ("intrinsic_value_gap",), 0.00, 0.35), 0.45),
                )
            )
            margin_of_safety_proxy = coalesce(explicit_margin
                if explicit_margin is not None
                else weighted_average(((value_proxy, 0.60), (drawdown_quality, 0.40))), 0.5)

            if len(closes) < 60:
                blockers.setdefault(str(symbol), []).append("quality proxy needs at least 60 completed bars")
            output[str(symbol)] = {
                "quality_proxy": rounded(clamp(quality_proxy)),
                "business_quality_proxy": rounded(clamp(profitability if profitability is not None else quality_proxy)),
                "financial_strength_proxy": rounded(clamp(financial_strength if financial_strength is not None else quality_proxy)),
                "moat_proxy": rounded(clamp(moat_proxy if moat_proxy is not None else quality_proxy)),
                "valuation_proxy": rounded(clamp(value_proxy)),
                "margin_of_safety_proxy": rounded(clamp(margin_of_safety_proxy)),
                "trend_consistency": rounded(clamp(trend_consistency)),
                "volatility_quality": rounded(clamp(volatility_quality)),
                "drawdown_quality": rounded(clamp(drawdown_quality)),
                "liquidity_quality": rounded(clamp(liquidity_quality)),
                "drawdown_120": rounded(drawdown if drawdown is not None and math.isfinite(drawdown) else 0.0),
                "external_fundamental_coverage": rounded(clamp(coverage)),
            }
            confidence_values.append(clamp(0.60 + coverage * 0.40))

        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
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
