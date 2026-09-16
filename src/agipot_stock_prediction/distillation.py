"""Offline single-stock research from caller-supplied, point-in-time inputs.

No provider clients, account state, persistence, or order execution are included.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .calendar import is_us_equity_session, us_equity_session_bounds


ET = ZoneInfo("America/New_York")
DISTILLATION_VERSION = "specific-stock-distillation-oss-v1"
MIN_DAILY_POINTS = 252
MIN_FULL_HISTORY_YEARS = 8.0
SHORT_WEIGHT = 0.25
MEDIUM_WEIGHT = 0.35
LONG_WEIGHT = 0.40
ROUND_TRIP_COST_BPS = 15.0


def _num(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(clean) if clean else None


def _stdev(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def _percent_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in {None, 0}:
        return None
    return (current / float(previous) - 1.0) * 100.0


def _safe_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _safe_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        number = float(value)
        parsed = datetime.fromtimestamp(number / 1000 if number > 10_000_000_000 else number, tz=timezone.utc)
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def normalize_specific_stock_symbol(value: str) -> str:
    symbol = "".join(str(value or "").strip().upper().split())
    if symbol.endswith(".US"):
        symbol = symbol[:-3]
    if not symbol or len(symbol) > 12 or any(not (char.isalnum() or char in {"-", "."}) for char in symbol):
        raise ValueError("invalid_symbol")
    return symbol


def _latest_relevant_session(now: datetime) -> tuple[date, datetime, datetime, bool]:
    current = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    local_now = current.astimezone(ET)
    candidate = local_now.date()
    bounds = us_equity_session_bounds(candidate)
    if bounds is not None:
        opened, closed = bounds
        if local_now >= opened:
            session_end = min(local_now, closed)
            return candidate, opened, session_end, local_now < closed
    candidate -= timedelta(days=1)
    while not is_us_equity_session(candidate):
        candidate -= timedelta(days=1)
    opened, closed = us_equity_session_bounds(candidate) or (
        datetime.combine(candidate, time(9, 30), tzinfo=ET),
        datetime.combine(candidate, time(16, 0), tzinfo=ET),
    )
    return candidate, opened, closed, False


def _normalize_daily_rows(rows: Any) -> list[dict[str, Any]]:
    normalized: dict[date, dict[str, Any]] = {}
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        day = _safe_date(raw.get("date") or raw.get("timestamp"))
        close = _num(raw.get("close"))
        if day is None or close is None or close <= 0:
            continue
        open_price = _num(raw.get("open")) or close
        high = _num(raw.get("high")) or max(open_price, close)
        low = _num(raw.get("low")) or min(open_price, close)
        if min(open_price, high, low) <= 0 or high < max(open_price, close) or low > min(open_price, close):
            continue
        adjusted = _num(raw.get("adjusted_close") if raw.get("adjusted_close") is not None else raw.get("adj_close")) or close
        adjustment_factor = adjusted / close
        normalized[day] = {
            "date": day,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": adjusted,
            "adjusted_open": open_price * adjustment_factor,
            "adjusted_high": high * adjustment_factor,
            "adjusted_low": low * adjustment_factor,
            "volume": _num(raw.get("volume")),
        }
    return [normalized[day] for day in sorted(normalized)]


def _completed_daily_rows(rows: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Keep only NYSE sessions whose scheduled close is at or before ``now``.

    Daily dates denote exchange session dates. This also rejects future dates,
    weekends and holidays, even when the caller has supplied a complete-looking
    OHLC row. Data-vendor revision/availability timestamps are not inferable.
    """
    observed_at = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    local_day = observed_at.astimezone(ET).date()
    completed = []
    for row in rows:
        if row["date"] > local_day:
            continue
        bounds = us_equity_session_bounds(row["date"])
        if bounds is not None and bounds[1] <= observed_at:
            completed.append(row)
    return completed


def _normalize_intraday_rows(
    rows: Any,
    *,
    session_start: datetime,
    session_end: datetime,
) -> list[dict[str, Any]]:
    start_utc = session_start.astimezone(timezone.utc)
    end_utc = session_end.astimezone(timezone.utc)
    normalized: list[dict[str, Any]] = []
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        stamp = _safe_datetime(raw.get("timestamp") or raw.get("datetime") or raw.get("date"))
        close = _num(raw.get("close"))
        if stamp is None or close is None or close <= 0 or not (start_utc <= stamp <= end_utc):
            continue
        open_price = _num(raw.get("open")) or close
        high = _num(raw.get("high")) or max(open_price, close)
        low = _num(raw.get("low")) or min(open_price, close)
        if min(open_price, high, low) <= 0 or high < max(open_price, close) or low > min(open_price, close):
            continue
        normalized.append(
            {
                "timestamp": stamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": _num(raw.get("volume")),
                "quality_status": str(raw.get("quality_status") or "raw"),
                "quality_flags": list(raw.get("quality_flags") or []),
            }
        )
    normalized.sort(key=lambda item: item["timestamp"])
    return normalized


def _sma(values: list[float], length: int, end: int | None = None) -> float | None:
    stop = len(values) if end is None else end + 1
    start = stop - length
    return statistics.fmean(values[start:stop]) if start >= 0 and stop <= len(values) else None


def _rsi(values: list[float], length: int = 14, end: int | None = None) -> float | None:
    stop = len(values) - 1 if end is None else end
    start = stop - length
    if start < 0 or stop >= len(values):
        return None
    changes = [values[index] - values[index - 1] for index in range(start + 1, stop + 1)]
    gains = statistics.fmean(max(change, 0.0) for change in changes)
    losses = statistics.fmean(max(-change, 0.0) for change in changes)
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + gains / losses))


def _atr(rows: list[dict[str, Any]], length: int = 14) -> float | None:
    if len(rows) < length + 1:
        return None
    ranges = []
    for index in range(len(rows) - length, len(rows)):
        row = rows[index]
        previous_close = rows[index - 1]["close"]
        ranges.append(max(row["high"] - row["low"], abs(row["high"] - previous_close), abs(row["low"] - previous_close)))
    return statistics.fmean(ranges)


def _returns(values: list[float]) -> list[float | None]:
    result: list[float | None] = [None]
    for previous, current in zip(values, values[1:]):
        result.append(current / previous - 1.0 if previous else None)
    return result


def _realized_volatility(values: list[float], length: int, end: int | None = None) -> float | None:
    stop = len(values) if end is None else end + 1
    start = stop - length - 1
    if start < 0:
        return None
    sample = [math.log(values[index] / values[index - 1]) for index in range(start + 1, stop) if values[index - 1] > 0]
    deviation = _stdev(sample)
    return deviation * math.sqrt(252) * 100 if deviation is not None else None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else None


def _score_label(score: float | None) -> str:
    if score is None:
        return "DATA_BLOCKED"
    if score >= 35:
        return "BULLISH"
    if score <= -35:
        return "BEARISH"
    return "NEUTRAL"


def _stats(values: list[float], *, cost_bps: float = 0.0) -> dict[str, Any]:
    net = [value - cost_bps / 10_000 for value in values]
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value < 0]
    return {
        "sample_size": len(net),
        "mean_return_percent": _rounded((_mean(net) or 0.0) * 100, 3) if net else None,
        "median_return_percent": _rounded((_median(net) or 0.0) * 100, 3) if net else None,
        "win_rate_percent": _rounded(len(wins) / len(net) * 100, 2) if net else None,
        "payoff_ratio": _rounded(abs((_mean(wins) or 0.0) / (_mean(losses) or 1.0)), 3) if wins and losses else None,
        "round_trip_cost_bps": cost_bps,
    }


def _current_regime_edge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [row["adjusted_close"] for row in rows]
    if len(closes) < 260:
        return {
            "status": "BLOCKED",
            "reason": "INSUFFICIENT_HISTORY_FOR_OOS",
            "oos_pass": False,
            "horizons": {},
        }
    split = int(len(closes) * 0.70)
    # Fit regime thresholds on training observations only. Test-period
    # volatility must not change which training examples fall in a regime.
    rolling_vols = [
        value
        for index in range(20, split)
        if (value := _realized_volatility(closes, 20, index)) is not None
    ]
    low_vol = _quantile(rolling_vols, 0.33)
    high_vol = _quantile(rolling_vols, 0.67)
    if low_vol is None or high_vol is None:
        return {"status": "BLOCKED", "reason": "TRAINING_VOLATILITY_UNAVAILABLE", "oos_pass": False, "horizons": {}}

    def regime_at(index: int) -> tuple[bool, bool, str, str] | None:
        ma50 = _sma(closes, 50, index)
        ma200 = _sma(closes, 200, index)
        rsi = _rsi(closes, 14, index)
        volatility = _realized_volatility(closes, 20, index)
        if ma50 is None or ma200 is None or rsi is None or volatility is None:
            return None
        rsi_bucket = "low" if rsi < 40 else "high" if rsi > 60 else "mid"
        vol_bucket = "low" if volatility <= low_vol else "high" if volatility >= high_vol else "mid"
        return closes[index] > ma50, closes[index] > ma200, rsi_bucket, vol_bucket

    current = regime_at(len(closes) - 1)
    if current is None:
        return {"status": "BLOCKED", "reason": "CURRENT_REGIME_UNAVAILABLE", "oos_pass": False, "horizons": {}}
    horizons: dict[str, Any] = {}
    passes = []
    for horizon in (5, 20):
        train: list[float] = []
        test: list[float] = []
        purged = 0
        for index in range(200, len(closes) - horizon):
            # A training label must be fully realized before the test boundary.
            if index < split <= index + horizon:
                purged += 1
                continue
            if regime_at(index) != current:
                continue
            value = closes[index + horizon] / closes[index] - 1.0
            (train if index < split else test).append(value)
        train_stats = _stats(train, cost_bps=ROUND_TRIP_COST_BPS)
        test_stats = _stats(test, cost_bps=ROUND_TRIP_COST_BPS)
        passed = bool(
            train_stats["sample_size"] >= 25
            and (train_stats["mean_return_percent"] or 0) > 0
            and test_stats["sample_size"] >= 25
            and (test_stats["mean_return_percent"] or 0) > 0
            and (test_stats["win_rate_percent"] or 0) >= 50
        )
        passes.append(passed)
        horizons[f"{horizon}d"] = {
            "train": train_stats,
            "out_of_sample": test_stats,
            "passed": passed,
            "purged_cross_boundary_labels": purged,
        }
    return {
        "status": "RESEARCH_ONLY" if any(passes) else "BLOCKED",
        "reason": "BASIC_HOLDOUT_PASS_FULL_OVERFIT_GATES_NOT_RUN" if any(passes) else "NO_STABLE_TRAIN_AND_OOS_EDGE_AFTER_COST",
        "regime": {
            "above_sma50": current[0],
            "above_sma200": current[1],
            "rsi_bucket": current[2],
            "volatility_bucket": current[3],
        },
        "oos_pass": any(passes),
        "simons_stat_veto": True,
        "pbo_proxy": None,
        "feature_stability": None,
        "validation_tier": "BASIC_CHRONOLOGICAL_HOLDOUT_ONLY",
        "horizons": horizons,
        "split_policy": "first_70_percent_train_last_30_percent_oos_purge_cross_boundary_labels",
        "volatility_threshold_fit": {
            "scope": "training_only",
            "sample_size": len(rolling_vols),
            "low": low_vol,
            "high": high_vol,
            "last_training_index": split - 1,
        },
        "future_leakage_policy": "training_only_thresholds_forward_returns_used_only_as_labels",
        "limitations": [
            "Current-regime conditional holdout statistics are exploratory, not an independently validated forecast.",
            "Overlapping forward-return labels are dependent; no significance correction or walk-forward validation is performed.",
            "Volatility thresholds are fit across the training fold, not refit at each historical training observation.",
            "The caller must supply corporate-action adjustments available as of the requested time.",
        ],
    }


def _pattern_card(name: str, values: list[float], occurrences: int, interpretation: str) -> dict[str, Any]:
    result = _stats(values)
    return {"name": name, "occurrences": occurrences, **result, "interpretation": interpretation}


def _historical_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [row["adjusted_close"] for row in rows]
    daily_returns = [value for value in _returns(closes) if value is not None]
    autocorrelation = _correlation(daily_returns[1:], daily_returns[:-1]) if len(daily_returns) >= 3 else None
    gap_results: list[float] = []
    gap_continuations = 0
    breakout_results: list[float] = []
    volume_breakout_results: list[float] = []
    recovery_results: list[float] = []
    peak = closes[0] if closes else 0.0
    previous_drawdown = 0.0
    for index in range(20, len(rows)):
        row = rows[index]
        previous = rows[index - 1]
        gap = row["adjusted_open"] / previous["adjusted_close"] - 1.0 if previous["adjusted_close"] else 0.0
        if abs(gap) >= 0.01 and row["adjusted_open"]:
            intraday = row["adjusted_close"] / row["adjusted_open"] - 1.0
            gap_results.append(intraday)
            if intraday * gap > 0:
                gap_continuations += 1
        if index + 5 < len(rows) and closes[index] > max(closes[index - 20:index]):
            forward = closes[index + 5] / closes[index] - 1.0
            breakout_results.append(forward)
            recent_volumes = [value for value in (item["volume"] for item in rows[index - 20:index]) if value is not None]
            avg_volume = _mean(recent_volumes)
            if row["volume"] is not None and avg_volume and row["volume"] >= avg_volume * 1.5:
                volume_breakout_results.append(forward)
        peak = max(peak, closes[index])
        drawdown = closes[index] / peak - 1.0 if peak else 0.0
        if previous_drawdown > -0.20 >= drawdown and index + 60 < len(rows):
            recovery_results.append(closes[index + 60] / closes[index] - 1.0)
        previous_drawdown = drawdown
    gap_rate = gap_continuations / len(gap_results) * 100 if gap_results else None
    breakout_mean = (_mean(breakout_results) or 0.0) * 100 if breakout_results else None
    if breakout_mean is not None and breakout_mean > 2 and len(breakout_results) >= 30:
        dominant = "breakout_follow_through"
    elif gap_rate is not None and gap_rate < 45 and len(gap_results) >= 30:
        dominant = "opening_gap_mean_reversion"
    elif autocorrelation is not None and autocorrelation > 0.08:
        dominant = "short_horizon_trend_persistence"
    elif autocorrelation is not None and autocorrelation < -0.08:
        dominant = "short_horizon_mean_reversion"
    else:
        dominant = "mixed_regime_dependent"
    return {
        "dominant_behavior": dominant,
        "daily_return_autocorrelation": _rounded(autocorrelation, 4),
        "gap_continuation_rate_percent": _rounded(gap_rate, 2),
        "patterns": [
            _pattern_card("gap_at_least_1pct_intraday_follow_through", gap_results, len(gap_results), "开盘缺口后同日延续率需结合 gap_continuation_rate 阅读"),
            _pattern_card("20d_close_breakout_next_5d", breakout_results, len(breakout_results), "20 日收盘新高后的 5 日表现"),
            _pattern_card("volume_confirmed_breakout_next_5d", volume_breakout_results, len(volume_breakout_results), "放量 1.5 倍确认突破后的 5 日表现"),
            _pattern_card("first_20pct_drawdown_next_60d", recovery_results, len(recovery_results), "首次跌破前高 20% 后的 60 日恢复表现"),
        ],
    }


def _trend_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [row["adjusted_close"] for row in rows]
    current = closes[-1]
    ma20 = _sma(closes, 20)
    ma50 = _sma(closes, 50)
    ma200 = _sma(closes, 200)
    ret20 = _percent_change(current, closes[-21] if len(closes) > 20 else None)
    ret60 = _percent_change(current, closes[-61] if len(closes) > 60 else None)
    rsi14 = _rsi(closes)
    components = [
        {"name": "close_vs_sma20", "points": 15 if ma20 is not None and current > ma20 else -15, "evidence": {"close": _rounded(current), "sma20": _rounded(ma20)}},
        {"name": "sma20_vs_sma50", "points": 15 if ma20 is not None and ma50 is not None and ma20 > ma50 else -15, "evidence": {"sma20": _rounded(ma20), "sma50": _rounded(ma50)}},
        {"name": "sma50_vs_sma200", "points": 20 if ma50 is not None and ma200 is not None and ma50 > ma200 else -20, "evidence": {"sma50": _rounded(ma50), "sma200": _rounded(ma200)}},
        {"name": "momentum_20d", "points": _rounded(_clamp((ret20 or 0.0) * 1.25, -20, 20), 2), "evidence": {"return_percent": _rounded(ret20, 2)}},
        {"name": "momentum_60d", "points": _rounded(_clamp((ret60 or 0.0) * 0.50, -15, 15), 2), "evidence": {"return_percent": _rounded(ret60, 2)}},
        {"name": "rsi_balance", "points": _rounded(_clamp(((rsi14 or 50.0) - 50.0) / 2.0, -10, 10), 2), "evidence": {"rsi14": _rounded(rsi14, 2)}},
    ]
    score = _clamp(sum(float(item["points"] or 0.0) for item in components), -100, 100)
    return {
        "direction_score": _rounded(score, 2),
        "index": _rounded((score + 100) / 2, 2),
        "label": _score_label(score),
        "components": components,
        "sma20": _rounded(ma20),
        "sma50": _rounded(ma50),
        "sma200": _rounded(ma200),
        "rsi14": _rounded(rsi14, 2),
        "return_20d_percent": _rounded(ret20, 2),
        "return_60d_percent": _rounded(ret60, 2),
    }


def _distill_medium(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    if len(rows) < MIN_DAILY_POINTS:
        return {"status": "BLOCKED", "score": None, "index": None, "reason": "INSUFFICIENT_DAILY_HISTORY", "point_count": len(rows)}
    trend = _trend_components(rows)
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows[-20:] if row["volume"] is not None]
    avg_volume20 = _mean(volumes)
    volume_ratio = rows[-1]["volume"] / avg_volume20 if rows[-1]["volume"] is not None and avg_volume20 else None
    atr14 = _atr(rows)
    pivot = max(row["high"] for row in rows[-21:-1])
    swing_low = min(row["low"] for row in rows[-10:])
    entry = pivot
    stop = entry - 1.5 * atr14 if atr14 is not None else None
    target = entry + 2.0 * atr14 if atr14 is not None else None
    current = closes[-1]
    confirmed = bool(current > pivot and (volume_ratio or 0) >= 1.2 and (trend["direction_score"] or 0) >= 35)
    edge = _current_regime_edge(rows)
    behavior = _historical_behavior(rows)
    return {
        "status": "PASS",
        "source": source,
        "actual_granularity": "1d",
        "point_count": len(rows),
        "coverage": {"from": rows[0]["date"].isoformat(), "to": rows[-1]["date"].isoformat()},
        "direction_score": trend["direction_score"],
        "index": trend["index"],
        "label": trend["label"],
        "trend": trend,
        "atr14": _rounded(atr14),
        "volume_ratio_20d": _rounded(volume_ratio, 2),
        "historical_logic": behavior,
        "statistical_edge": edge,
        "execution_plan": {
            "status": "CONFIRMED" if confirmed else "WAIT",
            "breakout_pivot": _rounded(pivot),
            "pullback_reference_sma20": trend["sma20"],
            "swing_low_10d": _rounded(swing_low),
            "entry_reference": _rounded(entry),
            "stop_reference": _rounded(stop),
            "first_target_reference": _rounded(target),
            "rule": "仅当收盘越过前 20 日高点、量比不低于 1.2 且趋势分数至少 35 时确认；否则等待。",
        },
    }


def _aggregate_period(rows: list[dict[str, Any]], period: str) -> list[dict[str, Any]]:
    buckets: OrderedDict[Any, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        day: date = row["date"]
        key = (day.isocalendar().year, day.isocalendar().week) if period == "week" else (day.year, day.month)
        buckets.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for group in buckets.values():
        result.append(
            {
                "date": group[-1]["date"],
                "open": group[0]["open"],
                "high": max(item["high"] for item in group),
                "low": min(item["low"] for item in group),
                "close": group[-1]["close"],
                "adjusted_close": group[-1]["adjusted_close"],
                "volume": sum(item["volume"] for item in group if item["volume"] is not None) or None,
            }
        )
    return result


def _period_key(day: date, period: str) -> tuple[int, int]:
    return (day.isocalendar().year, day.isocalendar().week) if period == "week" else (day.year, day.month)


def _completed_periods(rows: list[dict[str, Any]], period: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    bars = _aggregate_period(rows, period)
    if not bars:
        return [], None
    last_day = rows[-1]["date"]
    next_day = last_day + timedelta(days=1)
    while not is_us_equity_session(next_day):
        next_day += timedelta(days=1)
    if _period_key(next_day, period) == _period_key(last_day, period):
        return bars[:-1], bars[-1]
    return bars, None


def _period_trend(rows: list[dict[str, Any]], short_length: int, long_length: int, momentum_lengths: tuple[int, int]) -> dict[str, Any]:
    closes = [row["adjusted_close"] for row in rows]
    current = closes[-1]
    short_ma = _sma(closes, short_length)
    long_ma = _sma(closes, long_length)
    momentum_short = _percent_change(current, closes[-momentum_lengths[0] - 1] if len(closes) > momentum_lengths[0] else None)
    momentum_long = _percent_change(current, closes[-momentum_lengths[1] - 1] if len(closes) > momentum_lengths[1] else None)
    score = 0.0
    score += 30 if short_ma is not None and current > short_ma else -30
    score += 30 if short_ma is not None and long_ma is not None and short_ma > long_ma else -30
    score += _clamp((momentum_short or 0.0) * 0.8, -20, 20)
    score += _clamp((momentum_long or 0.0) * 0.35, -20, 20)
    score = _clamp(score, -100, 100)
    peak = max(closes)
    drawdown = _percent_change(current, peak)
    return {
        "direction_score": _rounded(score, 2),
        "index": _rounded((score + 100) / 2, 2),
        "label": _score_label(score),
        "close": _rounded(current),
        f"sma{short_length}": _rounded(short_ma),
        f"sma{long_length}": _rounded(long_ma),
        f"return_{momentum_lengths[0]}_bars_percent": _rounded(momentum_short, 2),
        f"return_{momentum_lengths[1]}_bars_percent": _rounded(momentum_long, 2),
        "drawdown_from_history_peak_percent": _rounded(drawdown, 2),
        "point_count": len(rows),
    }


def _distill_long(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    if len(rows) < MIN_DAILY_POINTS * 2:
        return {"status": "BLOCKED", "direction_score": None, "index": None, "reason": "INSUFFICIENT_LONG_HISTORY", "point_count": len(rows)}
    weekly, provisional_week = _completed_periods(rows, "week")
    monthly, provisional_month = _completed_periods(rows, "month")
    if len(weekly) < 53 or len(monthly) < 37:
        return {"status": "BLOCKED", "direction_score": None, "index": None, "reason": "INSUFFICIENT_COMPLETED_WEEKLY_OR_MONTHLY_HISTORY"}
    weekly_trend = _period_trend(weekly, 10, 40, (13, 52))
    monthly_trend = _period_trend(monthly, 10, 30, (12, 36))
    formal_direction = statistics.fmean([weekly_trend["direction_score"], monthly_trend["direction_score"]])
    period_conflict = weekly_trend["label"] != monthly_trend["label"]
    positive_months = sum(1 for previous, current in zip(monthly, monthly[1:]) if current["adjusted_close"] > previous["adjusted_close"])
    positive_rate = positive_months / (len(monthly) - 1) * 100 if len(monthly) > 1 else None
    return {
        "status": "PARTIAL" if period_conflict else "PASS",
        "source": source,
        "actual_granularity": "derived_1wk_and_1mo_from_adjusted_daily",
        "direction_score": None if period_conflict else _rounded(formal_direction, 2),
        "index": None if period_conflict else _rounded((formal_direction + 100) / 2, 2),
        "label": "PERIOD_CONFLICT" if period_conflict else _score_label(formal_direction),
        "formal_direction_score": _rounded(formal_direction, 2),
        "formal_index": _rounded((formal_direction + 100) / 2, 2),
        "weekly": weekly_trend,
        "monthly": monthly_trend,
        "completed_through": {"weekly": weekly[-1]["date"].isoformat(), "monthly": monthly[-1]["date"].isoformat()},
        "provisional": {
            "weekly": {"date": provisional_week["date"].isoformat(), "close": _rounded(provisional_week["adjusted_close"])} if provisional_week else None,
            "monthly": {"date": provisional_month["date"].isoformat(), "close": _rounded(provisional_month["adjusted_close"])} if provisional_month else None,
            "policy": "unfinished_week_or_month_is_display_only_and_excluded_from_formal_score",
        },
        "positive_month_rate_percent": _rounded(positive_rate, 2),
        "cycle_logic": (
            "周线与月线同向，长周期趋势一致。"
            if weekly_trend["label"] == monthly_trend["label"] and weekly_trend["label"] in {"BULLISH", "BEARISH"}
            else "周线与月线都在中性区，方向优势不足。"
            if weekly_trend["label"] == monthly_trend["label"] == "NEUTRAL"
            else "周线与月线分歧，按周期冲突处理，不追价。"
        ),
        "coverage": {"from": rows[0]["date"].isoformat(), "to": rows[-1]["date"].isoformat(), "years": _rounded((rows[-1]["date"] - rows[0]["date"]).days / 365.25, 2)},
    }


def _infer_intraday_granularity(rows: list[dict[str, Any]]) -> str | None:
    deltas = [
        int((current["timestamp"] - previous["timestamp"]).total_seconds())
        for previous, current in zip(rows, rows[1:])
        if current["timestamp"] > previous["timestamp"]
    ]
    median_delta = _median(deltas)
    if median_delta is None:
        return None
    if median_delta <= 75:
        return "1m"
    if median_delta <= 375:
        return "5m"
    if median_delta <= 4_500:
        return "1h"
    return f"observed_{int(median_delta)}s"


def _minute_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 20:
        return {"status": "BLOCKED", "reason": "INSUFFICIENT_1M_REFERENCE", "point_count": len(rows)}
    first = rows[0]["open"]
    last = rows[-1]["close"]
    volumes = [row["volume"] or 0.0 for row in rows]
    total_volume = sum(volumes)
    vwap = sum(row["close"] * volume for row, volume in zip(rows, volumes)) / total_volume if total_volume else None
    opening_rows = rows[: min(15, len(rows))]
    opening_high = max(row["high"] for row in opening_rows)
    opening_low = min(row["low"] for row in opening_rows)
    session_high = max(row["high"] for row in rows)
    session_low = min(row["low"] for row in rows)
    location = (last - session_low) / (session_high - session_low) if session_high > session_low else 0.5
    return_5m = _percent_change(last, rows[-6]["close"] if len(rows) > 5 else first)
    score = 0.0
    score += _clamp((_percent_change(last, first) or 0.0) * 8, -30, 30)
    score += _clamp((_percent_change(last, vwap) or 0.0) * 10, -20, 20)
    score += _clamp((return_5m or 0.0) * 12, -20, 20)
    score += (location - 0.5) * 40
    score += 15 if last > opening_high else -15 if last < opening_low else 0
    score = _clamp(score, -100, 100)
    return {
        "status": "RESEARCH_ONLY",
        "reason": "ONE_MINUTE_DATA_CANNOT_SATISFY_ONE_SECOND_REQUEST",
        "actual_granularity": _infer_intraday_granularity(rows),
        "point_count": len(rows),
        "direction_score": _rounded(score, 2),
        "index": _rounded((score + 100) / 2, 2),
        "label": _score_label(score),
        "session_return_percent": _rounded(_percent_change(last, first), 3),
        "last_vs_vwap_percent": _rounded(_percent_change(last, vwap), 3),
        "momentum_5m_percent": _rounded(return_5m, 3),
        "vwap": _rounded(vwap),
        "opening_range_15m": {"high": _rounded(opening_high), "low": _rounded(opening_low)},
        "session_range": {"high": _rounded(session_high), "low": _rounded(session_low), "last_location_percent": _rounded(location * 100, 2)},
    }


def _tick_rows(payload: Any, *, start: datetime, end: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(payload, dict):
        return [], {"raw_count": 0, "invalid_count": 0, "excluded_condition_count": 0, "duplicate_sequence_count": 0}
    timestamps = payload.get("ts")
    prices = payload.get("price")
    shares = payload.get("shares")
    sequences = payload.get("seq")
    conditions = payload.get("sl")
    markets = payload.get("mkt")
    if not isinstance(timestamps, list) or not isinstance(prices, list):
        return [], {"raw_count": 0, "invalid_count": 0, "excluded_condition_count": 0, "duplicate_sequence_count": 0}
    start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
    excluded_codes = set("HLNPTUVWZ")
    rows: list[dict[str, Any]] = []
    invalid = 0
    excluded = 0
    duplicate = 0
    seen_sequences: set[int] = set()
    for index, raw_ts in enumerate(timestamps):
        timestamp = _num(raw_ts)
        price = _num(prices[index] if index < len(prices) else None)
        if timestamp is None or price is None or price <= 0:
            invalid += 1
            continue
        ts_ms = int(timestamp if timestamp > 10_000_000_000 else timestamp * 1000)
        # The session window is half-open: the 16:00:00 boundary belongs to
        # neither the regular-session second series nor its coverage count.
        if not start_ms <= ts_ms < end_ms:
            continue
        condition = str(conditions[index] if isinstance(conditions, list) and index < len(conditions) else "").strip().upper()
        if any(code in excluded_codes for code in condition.replace("@", "")):
            excluded += 1
            continue
        sequence = int(_num(sequences[index]) or 0) if isinstance(sequences, list) and index < len(sequences) else 0
        if sequence and sequence in seen_sequences:
            duplicate += 1
            continue
        if sequence:
            seen_sequences.add(sequence)
        rows.append(
            {
                "ts_ms": ts_ms,
                "price": price,
                "size": _num(shares[index] if isinstance(shares, list) and index < len(shares) else None) or 0.0,
                "sequence": sequence,
                "condition": condition,
                "market": str(markets[index] if isinstance(markets, list) and index < len(markets) else ""),
            }
        )
    rows.sort(key=lambda item: (item["ts_ms"], item["sequence"]))
    return rows, {
        "raw_count": len(timestamps),
        "accepted_count": len(rows),
        "invalid_count": invalid,
        "excluded_condition_count": excluded,
        "duplicate_sequence_count": duplicate,
        "condition_policy": "causal_exclusion_of_out_of_sequence_extended_or_average_price_conditions",
    }


def _downsample(items: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(items) <= max_points:
        return items
    step = (len(items) - 1) / (max_points - 1)
    indexes = sorted({round(index * step) for index in range(max_points)})
    return [items[index] for index in indexes]


def _distill_ticks(
    rows: list[dict[str, Any]],
    *,
    session_start: datetime,
    session_end: datetime,
    max_points: int,
    diagnostics: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    start_ms = int(session_start.astimezone(timezone.utc).timestamp() * 1000)
    end_ms = int(session_end.astimezone(timezone.utc).timestamp() * 1000)
    start_second = start_ms // 1000
    end_second_exclusive = max(start_second + 1, math.ceil(end_ms / 1000))
    elapsed_seconds = end_second_exclusive - start_second
    if not rows:
        if diagnostics.get("pagination_complete") and not diagnostics.get("provider_error"):
            empty_reason = "HISTORICAL_TICK_WINDOW_EMPTY"
        elif diagnostics.get("provider_error"):
            empty_reason = "HISTORICAL_TICK_REQUEST_FAILED"
        else:
            empty_reason = "HISTORICAL_TICK_DATA_UNAVAILABLE"
        return {
            "status": "DATA_BLOCKED",
            "reason": empty_reason,
            "requested_granularity": "1s",
            "actual_granularity": None,
            "direction_score": None,
            "index": None,
            "seconds_processed": 0,
            "expected_seconds": elapsed_seconds,
            "diagnostics": diagnostics,
        }
    buckets: dict[int, dict[str, Any]] = {}
    up_volume = 0.0
    down_volume = 0.0
    previous_price: float | None = None
    for row in rows:
        second = row["ts_ms"] // 1000
        bucket = buckets.get(second)
        if bucket is None:
            bucket = {"second": second, "open": row["price"], "high": row["price"], "low": row["price"], "close": row["price"], "volume": 0.0, "trades": 0}
            buckets[second] = bucket
        bucket["high"] = max(bucket["high"], row["price"])
        bucket["low"] = min(bucket["low"], row["price"])
        bucket["close"] = row["price"]
        bucket["volume"] += row["size"]
        bucket["trades"] += 1
        if previous_price is not None:
            if row["price"] > previous_price:
                up_volume += row["size"]
            elif row["price"] < previous_price:
                down_volume += row["size"]
        previous_price = row["price"]
    observed_seconds = sorted(buckets)
    first_second = observed_seconds[0]
    last_second = observed_seconds[-1]
    start_delay = max(0, rows[0]["ts_ms"] - start_ms) / 1000
    end_lag = max(0, end_ms - rows[-1]["ts_ms"]) / 1000
    pagination_declared = "pagination_complete" in diagnostics
    limit_reached = bool(
        diagnostics.get("truncated")
        or (pagination_declared and not diagnostics.get("pagination_complete"))
    )
    timeline_coverage = max(0.0, min(1.0, (last_second - first_second + 1) / elapsed_seconds))
    trade_timeline_pass = start_delay <= 10 and end_lag <= 120 and timeline_coverage >= 0.98 and not limit_reached
    seconds: list[dict[str, Any]] = []
    last_close: float | None = None
    for second in range(start_second, end_second_exclusive):
        bucket = buckets.get(second)
        if bucket is not None:
            last_close = bucket["close"]
        if last_close is None:
            continue
        seconds.append(
            {
                "time": datetime.fromtimestamp(second, tz=timezone.utc).isoformat(),
                "price": _rounded(last_close),
                "volume": _rounded(bucket["volume"], 0) if bucket else 0,
                "trade_count": bucket["trades"] if bucket else 0,
                "carried": bucket is None,
            }
        )
    first = rows[0]["price"]
    last = rows[-1]["price"]
    total_volume = sum(row["size"] for row in rows)
    vwap = sum(row["price"] * row["size"] for row in rows) / total_volume if total_volume else None
    opening_cutoff = start_ms + 5 * 60 * 1000
    opening = [row for row in rows if row["ts_ms"] <= opening_cutoff] or rows[:1]
    opening_high = max(row["price"] for row in opening)
    opening_low = min(row["price"] for row in opening)
    price_30s = next((row["price"] for row in reversed(rows) if row["ts_ms"] <= rows[-1]["ts_ms"] - 30_000), first)
    price_5m = next((row["price"] for row in reversed(rows) if row["ts_ms"] <= rows[-1]["ts_ms"] - 300_000), first)
    imbalance = (up_volume - down_volume) / (up_volume + down_volume) if up_volume + down_volume else 0.0
    session_return = _percent_change(last, first) or 0.0
    vwap_distance = _percent_change(last, vwap) or 0.0
    momentum_30s = _percent_change(last, price_30s) or 0.0
    momentum_5m = _percent_change(last, price_5m) or 0.0
    score = 0.0
    score += _clamp(session_return * 8, -25, 25)
    score += _clamp(vwap_distance * 10, -20, 20)
    score += _clamp(momentum_30s * 25, -15, 15)
    score += _clamp(momentum_5m * 12, -20, 20)
    score += imbalance * 20
    score += 10 if last > opening_high else -10 if last < opening_low else 0
    score = _clamp(score, -100, 100)
    action = "WAIT"
    if trade_timeline_pass and score >= 35 and last > opening_high:
        action = "LONG_PATTERN_ONLY"
    elif trade_timeline_pass and score <= -35 and last < opening_low:
        action = "SHORT_PATTERN_ONLY"
    return {
        "status": "DATA_BLOCKED",
        "reason": "HISTORICAL_NBBO_UNAVAILABLE" if trade_timeline_pass else "TICK_TIMELINE_INCOMPLETE_AND_NBBO_UNAVAILABLE",
        "trade_timeline_status": "PASS" if trade_timeline_pass else "PARTIAL",
        "requested_granularity": "1s",
        "actual_granularity": "trade_ticks_forward_filled_to_1s",
        "source": source,
        "direction_score": None,
        "pattern_direction_score": _rounded(score, 2),
        "index": None,
        "pattern_index": _rounded((score + 100) / 2, 2),
        "label": "DATA_BLOCKED",
        "execution_gate": "BLOCKED_NO_HISTORICAL_NBBO",
        "pattern_action": action,
        "action": "NO_TRADE",
        "seconds_processed": len(seconds),
        "expected_seconds": elapsed_seconds,
        "raw_trade_count": len(rows),
        "raw_trade_seconds": len(observed_seconds),
        "carried_seconds": sum(1 for item in seconds if item["carried"]),
        "timeline_coverage_percent": _rounded(timeline_coverage * 100, 2),
        "start_delay_seconds": _rounded(start_delay, 3),
        "end_lag_seconds": _rounded(end_lag, 3),
        "session_return_percent": _rounded(session_return, 3),
        "last_vs_vwap_percent": _rounded(vwap_distance, 3),
        "momentum_30s_percent": _rounded(momentum_30s, 3),
        "momentum_5m_percent": _rounded(momentum_5m, 3),
        "up_down_volume_imbalance": _rounded(imbalance, 4),
        "vwap": _rounded(vwap),
        "opening_range_5m": {"high": _rounded(opening_high), "low": _rounded(opening_low)},
        "diagnostics": {**diagnostics, "limit_reached": limit_reached},
        "series": _downsample(seconds, max(20, min(max_points, 1200))),
        "series_policy": "all_seconds_processed_server_side_series_downsampled_for_display",
    }


def _fundamental_quality(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload or not payload.get("ok"):
        return {
            "status": "BLOCKED",
            "quality_score": None,
            "buffett_munger_veto": True,
            "reason": "FUNDAMENTALS_UNAVAILABLE",
        }
    rows = payload.get("financialRows") if isinstance(payload.get("financialRows"), list) else []
    if not rows:
        return {"status": "BLOCKED", "quality_score": None, "buffett_munger_veto": True, "reason": "FINANCIAL_ROWS_UNAVAILABLE"}
    latest = rows[0]
    comparable = rows[4] if len(rows) > 4 else None
    latest_revenue = _num(latest.get("revenue"))
    comparable_revenue = _num(comparable.get("revenue")) if comparable else None
    revenue_growth = _percent_change(latest_revenue, comparable_revenue)
    gross_margin = _num(latest.get("grossMargin"))
    fcf_values = [_num(row.get("freeCashFlow")) for row in rows]
    positive_fcf = sum(1 for value in fcf_values if value is not None and value > 0)
    fcf_consistency = positive_fcf / len(rows) * 100 if rows else None
    latest_net_income = _num(latest.get("netIncome"))
    latest_fcf = _num(latest.get("freeCashFlow"))
    score = 0.0
    score += _clamp((revenue_growth or 0.0) + 20, 0, 40)
    score += _clamp((gross_margin or 0.0) / 2, 0, 30)
    score += _clamp((fcf_consistency or 0.0) * 0.2, 0, 20)
    score += 10 if latest_net_income is not None and latest_net_income > 0 else 0
    veto = bool(
        (fcf_consistency or 0) < 50
        or latest_net_income is None
        or latest_net_income <= 0
        or latest_fcf is None
        or latest_fcf <= 0
    )
    return {
        "status": "BLOCKED" if veto else "PASS",
        "quality_score": _rounded(score, 2),
        "revenue_yoy_percent": _rounded(revenue_growth, 2),
        "gross_margin_percent": _rounded(gross_margin, 2),
        "positive_fcf_quarter_ratio_percent": _rounded(fcf_consistency, 2),
        "latest_net_income": _rounded(latest_net_income, 2),
        "latest_free_cash_flow": _rounded(latest_fcf, 2),
        "moat_score": None,
        "valuation_margin_of_safety": None,
        "balance_sheet_risk": "NOT_COLLECTED",
        "owner_earnings_quality": "BLOCKED_LATEST_FCF_NEGATIVE" if latest_fcf is not None and latest_fcf <= 0 else "PARTIAL",
        "buffett_munger_veto": veto,
        "reason": "QUALITY_GATE_FAILED" if veto else "QUALITY_GATE_PASSED",
    }


def distill_specific_stock_from_inputs(
    symbol: str,
    *,
    daily_rows: Any,
    minute_rows: Any,
    tick_payload: Any,
    fundamentals: dict[str, Any] | None,
    now: datetime,
    daily_source: str,
    minute_source: str,
    tick_source: str,
    tick_diagnostics: dict[str, Any] | None = None,
    max_points: int = 360,
) -> dict[str, Any]:
    """Compute research indicators without fetching data or generating orders.

    ``now`` is interpreted as UTC when naive. Daily OHLC dates are US exchange
    session dates and only closed sessions are analyzed. Fundamentals must be
    ordered latest-first and available at ``now``; filing availability cannot
    be derived from a reporting-period date. Minute rows must contain completed
    bars available at their supplied timestamps. Historical corporate-action
    adjustments and data revisions must also be point-in-time inputs.
    """
    now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    normalized_symbol = normalize_specific_stock_symbol(symbol)
    session_date, session_start, session_end, session_live = _latest_relevant_session(now)
    normalized_daily = _normalize_daily_rows(daily_rows)
    daily = _completed_daily_rows(normalized_daily, now)
    minutes = _normalize_intraday_rows(minute_rows, session_start=session_start, session_end=session_end)
    ticks, parsed_tick_diagnostics = _tick_rows(tick_payload, start=session_start, end=session_end)
    diagnostics = {**parsed_tick_diagnostics, **(tick_diagnostics or {})}
    short = _distill_ticks(
        ticks,
        session_start=session_start,
        session_end=session_end,
        max_points=max_points,
        diagnostics=diagnostics,
        source=tick_source,
    )
    short["minute_reference"] = _minute_reference(minutes)
    medium = _distill_medium(daily, daily_source)
    long = _distill_long(daily, daily_source)
    quality = _fundamental_quality(fundamentals)
    available: list[tuple[str, float, float]] = []
    for name, weight, horizon in (("short", SHORT_WEIGHT, short), ("medium", MEDIUM_WEIGHT, medium), ("long", LONG_WEIGHT, long)):
        score = _num(horizon.get("direction_score"))
        if horizon.get("status") == "PASS" and score is not None:
            available.append((name, weight, score))
    available_weight = sum(weight for _, weight, _ in available)
    direction = sum(weight * score for _, weight, score in available) / available_weight if available_weight else None
    complete = len(available) == 3
    history_years = (daily[-1]["date"] - daily[0]["date"]).days / 365.25 if len(daily) >= 2 else 0.0
    full_history = bool(history_years >= MIN_FULL_HISTORY_YEARS)
    data_gates = {
        "short_1s": {
            "status": short["status"],
            "requested": "每秒（开盘至当前/收盘）",
            "actual": short.get("actual_granularity"),
            "reason": short.get("reason"),
            "trade_timeline_status": short.get("trade_timeline_status", "BLOCKED"),
            "execution_gate": short.get("execution_gate", "BLOCKED"),
        },
        "medium_1d": {
            "status": medium["status"],
            "requested": "完整日线历史",
            "actual": medium.get("actual_granularity"),
            "point_count": len(daily),
            "excluded_unclosed_or_non_session_count": len(normalized_daily) - len(daily),
            "cutoff_policy": "exchange_session_close_at_or_before_now",
            "history_years": _rounded(history_years, 2),
            "full_history_gate": "PASS" if full_history else "PARTIAL",
        },
        "long_1w_1mo": {
            "status": long["status"],
            "requested": "周线与月线",
            "actual": long.get("actual_granularity"),
        },
        "fundamentals": {"status": quality["status"], "reason": quality["reason"]},
        "historical_nbbo": {"status": "BLOCKED", "reason": "PROVIDER_TICK_HISTORY_HAS_TRADES_NOT_HISTORICAL_NBBO"},
    }
    vetoes = []
    if short["status"] != "PASS":
        vetoes.append("one_second_history_gate_not_passed")
    vetoes.append("historical_nbbo_unavailable")
    if medium.get("statistical_edge", {}).get("status") != "PASS":
        vetoes.append("full_statistical_validation_not_passed")
    if quality.get("buffett_munger_veto"):
        vetoes.append("quality_value_gate_blocked")
    vetoes.extend(["macro_regime_not_collected", "supply_chain_claims_not_verified"])
    master_matrix = {
        "g0_data": "PASS" if complete and not vetoes else "BLOCKED",
        "g1_macro": "REVIEW_REQUIRED",
        "g2_quality_supply_chain": "BLOCKED" if quality["status"] != "PASS" else "REVIEW_REQUIRED",
        "g3_statistical_edge": medium.get("statistical_edge", {}).get("status", "BLOCKED"),
        "g4_execution": "BLOCKED" if short.get("execution_gate") != "PASS" else short.get("action"),
        "g5_risk": "RESEARCH_ONLY",
        "tradable": False,
        "action": "NO_TRADE_RESEARCH_ONLY",
        "vetoes": list(dict.fromkeys(vetoes)),
    }
    digest_rows = [
        [row["date"].isoformat(), row["open"], row["high"], row["low"], row["close"], row["adjusted_close"], row["volume"]]
        for row in daily
    ]
    digest = hashlib.sha256(
        json.dumps({"symbol": normalized_symbol, "daily": digest_rows}, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    for tick in ticks:
        digest.update(f"|{tick['ts_ms']}|{tick['price']}|{tick['size']}|{tick['sequence']}".encode("ascii"))
    data_hash = digest.hexdigest()
    status = "COMPLETE" if complete and full_history and not vetoes else "PARTIAL"
    return {
        "ok": bool(daily),
        "status": status,
        "symbol": normalized_symbol,
        "provider_symbol": f"{normalized_symbol}.US",
        "version": DISTILLATION_VERSION,
        "run_id": data_hash[:16],
        "data_hash": f"sha256:{data_hash}",
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "trained_through": daily[-1]["date"].isoformat() if daily else None,
        "session": {
            "date": session_date.isoformat(),
            "timezone": "America/New_York",
            "opened_at": session_start.isoformat(),
            "analyzed_through": session_end.isoformat(),
            "live": session_live,
        },
        "specific_stock_index": {
            "research_direction_score": _rounded(direction, 2),
            "research_index": _rounded((direction + 100) / 2, 2) if direction is not None else None,
            "complete_index": _rounded((direction + 100) / 2, 2) if complete and direction is not None else None,
            "label": _score_label(direction),
            "coverage_weight_percent": _rounded(available_weight * 100, 2),
            "available_horizons": [name for name, _, _ in available],
            "missing_horizons": [name for name in ("short", "medium", "long") if name not in {item[0] for item in available}],
            "formula": "short_1s=25% + medium_1d=35% + long_1w_1mo=40%; missing horizons are reweighted only for research_index and complete_index stays null",
            "tradable": False,
        },
        "data_gates": data_gates,
        "short": short,
        "medium": medium,
        "long": long,
        "fundamental_quality": quality,
        "master_matrix": master_matrix,
        "disclosures": [
            "分钟数据不会冒充秒级数据；秒级门禁未通过时短线主分数保持为空。",
            "历史 tick 只有成交，不含完整历史 NBBO，因此任何秒级结果均为研究级，不生成订单。",
            "统计边际使用前 70% 训练、后 30% 样本外验证，并扣除 15 bps 往返成本。",
            "逐点技术指标使用当时及以前的数据，分桶阈值在训练段拟合；未来收益仅作标签。",
            "仅纳入截至 now 已收盘的交易日日线，盘中当日、未来及休市日日线均排除。",
            "波动分桶阈值仅在训练段拟合，跨入测试段的训练收益标签已剔除；重叠标签不独立，当前状态条件统计仍属探索性研究。",
            "调用方须保证分钟线、财报、复权因子和数据修订在 now 时已可获得；本函数无法推断缺失的披露时间。",
            "公开信息蒸馏不构成投资建议。",
        ],
    }
