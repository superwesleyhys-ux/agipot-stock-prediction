"""Behavioral checks for the standalone scoring and allocation boundaries."""

import json
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agipot_stock_prediction.scoring import (
    BuffettRiskOverlaySkill,
    ConvictionAllocatorSkill,
    InverseVolatilityAllocator,
    MarketDataSnapshot,
    MinimumPositiveSleevesRegime,
    RawSkillOutput,
    SkillBinding,
    SkillContext,
    SkillVersion,
    score_stock,
)
from agipot_stock_prediction.scoring.math_utils import cap_and_redistribute


def daily_history(count=170, *, drift=0.0015):
    """Synthetic weekdays with variable returns; contains no vendor data."""
    day = date(2024, 1, 2)
    close = 100.0
    rows = []
    while len(rows) < count:
        if day.weekday() < 5:
            close *= math.exp(drift + 0.003 * math.sin(len(rows) * 1.7))
            rows.append({"date": day.isoformat(), "close": close, "volume": 3_000_000})
        day += timedelta(days=1)
    return rows


def test_score_graph_is_json_serializable_and_transparent():
    result = score_stock(daily_history(), symbol="example", as_of="2025-01-02")
    json.dumps(result, allow_nan=False)
    assert result["symbol"] == "EXAMPLE"
    assert set(result["scores"]) == {"trend", "momentum", "value"}
    assert result["scores"]["trend"]["eligible"]
    assert -2 <= result["scores"]["trend"]["score"] <= 2
    assert 0 <= result["scores"]["momentum"]["score"] <= 1
    assert 0 <= result["scores"]["value"]["score"] <= 1
    assert result["metadata"]["uses_price_quality_proxies"]
    assert result["metadata"]["external_fundamental_coverage"] == 0
    assert "not a calibrated probability" in result["metadata"]["confidence_meaning"]


@pytest.mark.parametrize("count", [0, 1, 20, 60, 125])
def test_insufficient_history_blocks_every_score(count):
    result = score_stock(daily_history(count), symbol="EXAMPLE", as_of="2025-01-02")
    assert result["blocking_reasons"]
    assert not any(score["eligible"] for score in result["scores"].values())
    json.dumps(result, allow_nan=False)


def test_future_and_same_day_rows_do_not_change_any_score():
    rows = daily_history()
    baseline = score_stock(rows, symbol="EXAMPLE", as_of="2025-01-02T23:59:00-05:00")
    result = score_stock(
        rows + [
            {"timestamp": "2025-01-02T00:01:00-05:00", "close": 0.01, "volume": 1},
            {"timestamp": "2025-01-10T15:00:00-05:00", "close": 1_000_000, "volume": 1},
        ],
        symbol="EXAMPLE",
        as_of="2025-01-02T23:59:00-05:00",
    )
    assert result["features"] == baseline["features"]
    assert result["scores"] == baseline["scores"]
    assert result["metadata"]["ignored_same_day_or_future_rows"] == 2


def test_new_york_calendar_cutoff_is_used_for_utc_timestamps():
    result = score_stock(
        [
            {"timestamp": "2025-01-02T03:00:00Z", "close": 100},  # Jan 1 NY
            {"timestamp": "2025-01-02T05:01:00Z", "close": 101},  # Jan 2 NY
        ],
        symbol="EXAMPLE",
        as_of="2025-01-02T20:00:00Z",
    )
    assert result["metadata"]["completed_rows"] == 1
    assert result["metadata"]["ignored_same_day_or_future_rows"] == 1


@pytest.mark.parametrize("field,value", [
    ("close", float("nan")), ("close", float("inf")), ("close", -1),
    ("close", 0), ("close", True), ("volume", float("-inf")), ("volume", -1),
])
def test_invalid_price_or_volume_is_rejected(field, value):
    row = {"date": "2024-01-02", "close": 100, "volume": 1000}
    row[field] = value
    with pytest.raises(ValueError):
        score_stock([row], symbol="EXAMPLE", as_of="2025-01-02")


def test_nonfinite_fundamentals_cannot_enter_scores():
    with pytest.raises(ValueError, match="finite"):
        score_stock([], symbol="EXAMPLE", as_of="2025-01-02", fundamentals={"roic": float("nan")})


def test_fundamentals_cannot_override_price_features():
    with pytest.raises(ValueError, match="unsupported fundamental key"):
        score_stock([], symbol="EXAMPLE", as_of="2025-01-02", fundamentals={"return_120": 1})


def test_duplicate_daily_rows_cannot_inflate_history_coverage():
    row = {"date": "2024-01-02", "close": 100}
    with pytest.raises(ValueError, match="multiple rows"):
        score_stock([row, row], symbol="EXAMPLE", as_of="2025-01-02")


def test_worst_valuation_remains_zero_instead_of_defaulting_to_neutral():
    result = score_stock(
        daily_history(), symbol="EXAMPLE", as_of="2025-01-02",
        fundamentals={"pe_ratio": 100, "margin_of_safety": -0.1},
    )
    assert result["features"]["valuation_proxy"] == 0
    assert result["features"]["margin_of_safety_proxy"] == 0
    assert result["scores"]["value"]["valuation_proxy"] == 0
    assert not result["scores"]["value"]["eligible"]


def test_falling_market_does_not_replace_zero_trend_consistency_with_neutral():
    result = score_stock(daily_history(drift=-0.003), symbol="EXAMPLE", as_of="2025-01-02")
    assert result["features"]["trend_consistency"] == 0
    assert result["scores"]["momentum"]["trend_consistency"] == 0
    assert result["scores"]["value"]["trend_consistency"] == 0
    assert not result["scores"]["trend"]["eligible"]
    assert "trend score must be positive" in result["scores"]["trend"]["blocking_reasons"]


def test_allocation_preserves_caps_and_is_independent_of_input_order():
    raw = {"A": 100, "B": 10, "C": 9, "D": 1}
    forward = cap_and_redistribute(raw, cap=0.25, gross_target=0.8)
    reverse = cap_and_redistribute(dict(reversed(list(raw.items()))), cap=0.25, gross_target=0.8)
    assert forward == reverse
    assert all(0 <= weight <= 0.25 for weight in forward.values())
    assert sum(forward.values()) == pytest.approx(0.8)


def test_small_universe_leaves_unallocated_budget_in_cash():
    weights = cap_and_redistribute({"A": 5, "B": 1}, cap=0.1, gross_target=0.7)
    assert weights == {"A": 0.1, "B": 0.1}


def _context(state, config=None):
    return SkillContext(
        snapshot=MarketDataSnapshot(as_of=datetime(2025, 1, 2, tzinfo=ZoneInfo("America/New_York"))),
        state=state,
        binding=SkillBinding(config=config or {}),
    )


def test_defensive_regime_produces_all_cash_conviction_portfolio():
    state = {"signals": {"EXAMPLE": {"eligible": True, "score": 0.8, "confidence": 0.8}}}
    regime = MinimumPositiveSleevesRegime(SkillVersion()).execute(_context(state))
    assert regime.payload["regime_id"] == "DEFENSIVE_CASH"
    state["regime"] = regime.payload
    portfolio = ConvictionAllocatorSkill(SkillVersion()).execute(_context(state))
    assert portfolio.payload["cash_weight"] == 1.0
    assert portfolio.payload["weights"] == {}


def test_inverse_volatility_allocator_only_uses_eligible_signals():
    portfolio = InverseVolatilityAllocator(SkillVersion()).execute(_context({
        "signals": {
            "A": {"eligible": True, "score": 1, "annualized_volatility": 0.2},
            "B": {"eligible": False, "score": 2, "annualized_volatility": 0.1},
        },
    }))
    assert portfolio.payload["weights"] == {"A": 0.05}
    assert portfolio.payload["cash_weight"] == 0.95


def test_risk_overlay_does_not_treat_zero_quality_as_missing():
    overlay = BuffettRiskOverlaySkill(SkillVersion()).execute(_context({
        "weights": {"A": 0.1},
        "features": {"A": {"quality_proxy": 0, "margin_of_safety_proxy": 0}},
    }))
    assert overlay.payload["average_quality_proxy"] == 0
    assert overlay.payload["average_margin_of_safety_proxy"] == 0
    assert overlay.payload["multiplier"] < 1


def test_nonfinite_component_output_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        RawSkillOutput(result_type="alpha_signals", payload={"score": float("nan")})
