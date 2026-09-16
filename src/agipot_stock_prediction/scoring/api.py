"""Validated one-stock interface to AGIPOT's offline scoring graph."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from .buffett_quality_value import BuffettQualityValueAlphaSkill
from .contracts import HistoryBar, MarketDataSnapshot, SkillContext, SkillVersion
from .momentum_leader import MomentumLeaderAlphaSkill
from .quality_proxy import QualityProxySkill
from .trend_ensemble import TrendEnsembleSkill
from .trend_features import TrendFeatureSkill

NEW_YORK = ZoneInfo("America/New_York")
FUNDAMENTAL_KEYS = frozenset({
    "roic", "roe", "gross_margin", "operating_margin", "fcf_margin",
    "fcf_conversion", "gross_margin_stability", "earnings_stability",
    "revenue_stability", "debt_to_equity", "net_debt_to_fcf",
    "net_debt_to_ebitda", "interest_coverage", "share_dilution",
    "owner_earnings_yield", "fcf_yield", "earnings_yield", "price_to_fcf",
    "pe_ratio", "ev_to_ebit", "margin_of_safety", "intrinsic_value_gap",
})

SCORE_DEFINITIONS = {
    "trend": {
        "range": [-2.0, 2.0],
        "meaning": "Mean of clipped 20/60/120-session returns divided by annualized volatility times sqrt(lookback/252).",
        "eligibility": "All lookbacks available, positive volatility, median daily dollar volume >= 50 million, and score > 0.",
    },
    "momentum": {
        "range": [0.0, 1.0],
        "meaning": "Weighted normalized momentum (64%), trend consistency (14%), volatility/drawdown/liquidity quality (18%), and quality proxy (4%); subtract up to 0.12 for 20-session overextension.",
        "eligibility": "60/120-session returns >= 2%/4%, 20-session return >= -6%, volatility <= 80%, drawdown <= 35%, median dollar volume >= 20 million, score >= 0.50, and complete inputs.",
    },
    "value": {
        "range": [0.0, 1.0],
        "meaning": "Weighted quality (26%), business quality (16%), financial strength (14%), moat (12%), margin of safety (18%), valuation (8%), and trend consistency (6%). Missing fundamentals use explicitly labeled price proxies.",
        "eligibility": "Quality >= 0.55, margin-of-safety proxy >= 0.40, volatility <= 65%, drawdown <= 45%, median dollar volume >= 10 million, score >= 0.52, and complete price inputs.",
    },
}


def _timestamp(value: datetime | date | str, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field}: expected an ISO date or datetime") from exc
    else:
        raise ValueError(f"{field}: expected an ISO date or datetime")
    # Daily date strings and naive datetimes refer to the New York session date.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NEW_YORK)
    return parsed


def _numeric(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}: expected a finite number, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field}: expected a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field}: expected a finite number")
    return numeric


def score_stock(
    history: Iterable[Mapping[str, Any] | HistoryBar],
    *,
    symbol: str,
    as_of: datetime | date | str,
    fundamentals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run feature extraction → quality proxies → three independent scores.

    ``history`` contains one daily row per New York calendar date, each with
    ``timestamp`` (or ``date``), positive ``close``, and non-negative ``volume``
    (default 0). ISO date strings and naive timestamps are interpreted in New
    York. Callers must supply consistently adjusted prices and volume. Rows on
    or after the New York ``as_of`` date are excluded; the latest five completed
    rows are additionally skipped for trend features, as in the source model.

    ``fundamentals`` is an optional flat mapping using ``FUNDAMENTAL_KEYS``.
    Percentages and yields are fractions (0.20 means 20%); valuation and debt
    multiples are ratios. The caller must supply values known at ``as_of``.
    Unknown keys are rejected so fundamentals cannot overwrite price features.

    Invalid/non-finite input raises ``ValueError``. Insufficient valid history
    returns diagnostic scores with ``eligible=False`` and blocking reasons.
    Scores are research factors; confidence is a heuristic, not a probability
    of profit. This function performs no network calls and places no orders.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    cutoff = _timestamp(as_of, field="as_of")
    cutoff_date = cutoff.astimezone(NEW_YORK).date()
    bars: list[HistoryBar] = []
    observed_dates: set[date] = set()
    for index, row in enumerate(history):
        if isinstance(row, HistoryBar):
            bar = row
        else:
            if not isinstance(row, Mapping):
                raise ValueError(f"history[{index}] must be a mapping or HistoryBar")
            if "close" not in row:
                raise ValueError(f"history[{index}].close is required")
            timestamp = _timestamp(row.get("timestamp", row.get("date")), field=f"history[{index}].timestamp")
            bar = HistoryBar(
                timestamp=timestamp,
                close=_numeric(row["close"], field=f"history[{index}].close"),
                volume=_numeric(row.get("volume", 0), field=f"history[{index}].volume"),
            )
        session_date = bar.timestamp.astimezone(NEW_YORK).date()
        if session_date in observed_dates:
            raise ValueError(f"history contains multiple rows for New York date {session_date}")
        observed_dates.add(session_date)
        bars.append(bar)

    external: dict[str, float] = {}
    if fundamentals is not None:
        if not isinstance(fundamentals, Mapping):
            raise ValueError("fundamentals must be a flat mapping")
        for key, value in fundamentals.items():
            if key not in FUNDAMENTAL_KEYS:
                raise ValueError(f"unsupported fundamental key: {key}")
            external[key] = _numeric(value, field=f"fundamentals.{key}")

    completed = tuple(sorted(
        (bar for bar in bars if bar.timestamp.astimezone(NEW_YORK).date() < cutoff_date),
        key=lambda bar: bar.timestamp,
    ))
    snapshot = MarketDataSnapshot(
        as_of=cutoff,
        universe=(symbol,),
        histories={symbol: completed},
        external_features={symbol: external},
    )
    state: dict[str, Any] = {"universe": [symbol], "features": {}, "rejected": {}}
    context = SkillContext(snapshot=snapshot, state=state)
    version = SkillVersion()
    raw_features = TrendFeatureSkill(version).execute(context)
    features = dict(raw_features.payload["features"][symbol])
    features.update(external)
    blockers = list(raw_features.payload["blocking_reasons"].get(symbol, []))
    state["features"][symbol] = features
    quality = QualityProxySkill(version).execute(context)
    features.update(quality.payload["features"][symbol])
    blockers.extend(quality.payload["blocking_reasons"].get(symbol, []))
    state["rejected"][symbol] = list(dict.fromkeys(blockers))

    scores = {}
    for name, component in (
        ("trend", TrendEnsembleSkill),
        ("momentum", MomentumLeaderAlphaSkill),
        ("value", BuffettQualityValueAlphaSkill),
    ):
        signal = dict(component(version).execute(context).payload["signals"][symbol])
        if name == "trend" and signal["score"] <= 0:
            signal["blocking_reasons"] = list(dict.fromkeys([
                *signal["blocking_reasons"], "trend score must be positive",
            ]))
        scores[name] = signal

    all_blockers = list(dict.fromkeys([
        *state["rejected"][symbol],
        *(f"{name}: {reason}" for name, signal in scores.items() for reason in signal["blocking_reasons"]),
    ]))
    return {
        "symbol": symbol,
        "as_of": cutoff.isoformat(),
        "features": features,
        "scores": scores,
        "blocking_reasons": all_blockers,
        "metadata": {
            "input_rows": len(bars),
            "completed_rows": len(completed),
            "ignored_same_day_or_future_rows": len(bars) - len(completed),
            "skip_recent_completed_rows": 5,
            "minimum_rows_for_all_trend_lookbacks": 126,
            "calendar": "New York dates; lookbacks count supplied daily observations, not elapsed calendar days",
            "price_features_use_adjustment": "caller supplied",
            "fundamental_keys_supplied": sorted(external),
            "fundamentals_point_in_time": "caller responsibility; availability timestamps are not inferred",
            "external_fundamental_coverage": features["external_fundamental_coverage"],
            "uses_price_quality_proxies": features["external_fundamental_coverage"] < 1.0,
            "confidence_meaning": "heuristic completeness/eligibility score; not a calibrated probability or forecast return",
            "score_definitions": {name: dict(value) for name, value in SCORE_DEFINITIONS.items()},
        },
    }
