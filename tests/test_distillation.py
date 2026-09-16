from __future__ import annotations

import json
import math

import pytest
from datetime import date, datetime, timedelta, timezone

from agipot_stock_prediction import distillation as distillation_service
from agipot_stock_prediction.distillation import distill_specific_stock_from_inputs


def _daily_rows(count: int = 620) -> list[dict]:
    rows = []
    day = date(2024, 1, 2)
    price = 40.0
    while len(rows) < count:
        if day.weekday() < 5:
            drift = 0.001 if len(rows) % 11 else -0.004
            price *= 1 + drift
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": price * 0.995,
                    "high": price * 1.015,
                    "low": price * 0.985,
                    "close": price,
                    "adjusted_close": price,
                    "volume": 800_000 + len(rows) * 100,
                }
            )
        day += timedelta(days=1)
    return rows


def _minute_rows(start: datetime, count: int, step_minutes: int = 1) -> list[dict]:
    return [
        {
            "timestamp": (start + timedelta(minutes=index * step_minutes)).isoformat(),
            "open": 60 + index * 0.01,
            "high": 60.05 + index * 0.01,
            "low": 59.95 + index * 0.01,
            "close": 60.02 + index * 0.01,
            "volume": 1000 + index,
        }
        for index in range(count)
    ]


def _fundamentals_with_latest_loss() -> dict:
    return {
        "ok": True,
        "financialRows": [
            {
                "period": f"202{6 - index // 4}-{((index % 4) + 1) * 3:02d}-30",
                "revenue": 100_000_000 - index * 2_000_000,
                "grossMargin": 59,
                "netIncome": -10_000_000 if index == 0 else 2_000_000,
                "freeCashFlow": -4_000_000 if index == 0 else 5_000_000,
            }
            for index in range(8)
        ],
    }


def test_one_minute_reference_never_impersonates_one_second_data() -> None:
    session_start = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
    result = distill_specific_stock_from_inputs(
        "TEST",
        daily_rows=_daily_rows(),
        minute_rows=_minute_rows(session_start, 78, step_minutes=5),
        tick_payload=None,
        fundamentals=_fundamentals_with_latest_loss(),
        now=datetime(2026, 7, 20, 1, 0, tzinfo=timezone.utc),
        daily_source="test-daily",
        minute_source="test-five-minute",
        tick_source="unavailable",
    )

    assert result["short"]["status"] == "DATA_BLOCKED"
    assert result["short"]["direction_score"] is None
    assert result["short"]["index"] is None
    assert result["short"]["minute_reference"]["status"] == "RESEARCH_ONLY"
    assert result["short"]["minute_reference"]["actual_granularity"] == "5m"
    assert result["specific_stock_index"]["complete_index"] is None
    assert "short" in result["specific_stock_index"]["missing_horizons"]


def test_trade_ticks_without_historical_nbbo_remain_blocked() -> None:
    now = datetime(2026, 7, 17, 13, 35, tzinfo=timezone.utc)
    start = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
    timestamps = [int((start + timedelta(seconds=index)).timestamp() * 1000) for index in range(301)]
    tick_payload = {
        "ts": timestamps,
        "price": [60 + index * 0.001 for index in range(301)],
        "shares": [100] * 301,
        "seq": list(range(1, 302)),
        "sl": ["@"] * 301,
        "mkt": ["Q"] * 301,
    }
    result = distill_specific_stock_from_inputs(
        "TEST",
        daily_rows=_daily_rows(),
        minute_rows=_minute_rows(start, 6),
        tick_payload=tick_payload,
        fundamentals=_fundamentals_with_latest_loss(),
        now=now,
        daily_source="test-daily",
        minute_source="test-minute",
        tick_source="test-trades-only",
    )

    assert result["short"]["trade_timeline_status"] == "PASS"
    assert result["short"]["status"] == "DATA_BLOCKED"
    assert result["short"]["direction_score"] is None
    assert result["short"]["pattern_direction_score"] is not None
    assert result["short"]["execution_gate"] == "BLOCKED_NO_HISTORICAL_NBBO"
    assert result["short"]["expected_seconds"] == 300
    assert result["short"]["seconds_processed"] == 300
    assert result["master_matrix"]["tradable"] is False
    assert result["master_matrix"]["action"] == "NO_TRADE_RESEARCH_ONLY"


def test_latest_gaap_loss_and_negative_fcf_trigger_quality_veto() -> None:
    start = datetime(2026, 7, 17, 13, 30, tzinfo=timezone.utc)
    result = distill_specific_stock_from_inputs(
        "TEST",
        daily_rows=_daily_rows(),
        minute_rows=_minute_rows(start, 30),
        tick_payload=None,
        fundamentals=_fundamentals_with_latest_loss(),
        now=datetime(2026, 7, 20, 1, 0, tzinfo=timezone.utc),
        daily_source="test-daily",
        minute_source="test-minute",
        tick_source="unavailable",
    )

    quality = result["fundamental_quality"]
    assert quality["status"] == "BLOCKED"
    assert quality["buffett_munger_veto"] is True
    assert quality["latest_net_income"] < 0
    assert quality["latest_free_cash_flow"] < 0
    assert "quality_value_gate_blocked" in result["master_matrix"]["vetoes"]


def _row(day: str, price: float = 100.0) -> dict:
    return {"date": day, "open": price, "high": price, "low": price, "close": price, "volume": 1000}


def _research(daily_rows: list[dict], now: datetime) -> dict:
    return distill_specific_stock_from_inputs(
        "TEST",
        daily_rows=daily_rows,
        minute_rows=[],
        tick_payload=None,
        fundamentals=None,
        now=now,
        daily_source="synthetic",
        minute_source="unavailable",
        tick_source="unavailable",
    )


@pytest.mark.parametrize(
    ("observed_at", "trained_through", "count"),
    [
        ("2026-07-20T13:29:00+00:00", "2026-07-17", 1),
        ("2026-07-20T13:30:00+00:00", "2026-07-17", 1),
        ("2026-07-20T19:59:59+00:00", "2026-07-17", 1),
        ("2026-07-20T20:00:00+00:00", "2026-07-20", 2),
    ],
)
def test_daily_cutoff_uses_actual_session_close(observed_at: str, trained_through: str, count: int) -> None:
    result = _research(
        [_row("2026-07-17"), _row("2026-07-18"), _row("2026-07-20"), _row("2026-07-21")],
        datetime.fromisoformat(observed_at),
    )
    assert result["trained_through"] == trained_through
    assert result["data_gates"]["medium_1d"]["point_count"] == count
    assert result["data_gates"]["medium_1d"]["excluded_unclosed_or_non_session_count"] == 4 - count


@pytest.mark.parametrize("observed_at, expected_count", [("17:59:59", 1), ("18:00:00", 2)])
def test_early_close_and_exchange_holiday(observed_at: str, expected_count: int) -> None:
    # The day after US Thanksgiving closes at 13:00 ET (18:00 UTC).
    result = _research(
        [_row("2026-11-25"), _row("2026-11-26"), _row("2026-11-27")],
        datetime.fromisoformat(f"2026-11-27T{observed_at}+00:00"),
    )
    assert result["data_gates"]["medium_1d"]["point_count"] == expected_count
    assert result["trained_through"] == ("2026-11-25" if expected_count == 1 else "2026-11-27")


def test_future_and_unclosed_prices_do_not_change_research_or_hash() -> None:
    now = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)
    historical = [row for row in _daily_rows(900) if row["date"] < "2026-07-20"]
    baseline = _research(historical, now)
    contaminated = _research(historical + [_row("2026-07-20", 1e9), _row("2026-12-01", 1e12)], now)
    for key in ("data_hash", "run_id", "trained_through", "medium", "long", "specific_stock_index"):
        assert contaminated[key] == baseline[key]


def test_naive_now_means_utc() -> None:
    naive = datetime(2026, 7, 20, 20, 0)
    rows = [_row("2026-07-17"), _row("2026-07-20")]
    assert _research(rows, naive) == _research(rows, naive.replace(tzinfo=timezone.utc))


def test_future_only_daily_input_is_blocked() -> None:
    result = _research([_row("2026-07-21")], datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc))
    assert result["ok"] is False
    assert result["trained_through"] is None
    assert result["specific_stock_index"]["research_index"] is None
    assert result["master_matrix"]["tradable"] is False


def test_test_period_volatility_cannot_change_training_thresholds() -> None:
    # Fixed-length alternatives isolate held-out data from threshold fitting.
    length = 1000
    split = int(length * 0.70)
    rows = [{"adjusted_close": 100 * math.exp(index * 0.001 + 0.01 * math.sin(index))} for index in range(length)]
    altered = [dict(row) for row in rows]
    for index in range(split, length):
        altered[index]["adjusted_close"] *= 1.8 if index % 2 else 0.5
    first = distillation_service._current_regime_edge(rows)
    second = distillation_service._current_regime_edge(altered)
    assert first["volatility_threshold_fit"] == second["volatility_threshold_fit"]
    assert first["volatility_threshold_fit"]["last_training_index"] == split - 1


def test_flat_price_history_keeps_zero_threshold_and_is_json_serializable() -> None:
    rows = _daily_rows(500)
    for row in rows:
        row.update(open=100.0, high=100.0, low=100.0, close=100.0, adjusted_close=100.0)
    result = _research(rows, datetime(2026, 7, 20, 20, 0, tzinfo=timezone.utc))
    thresholds = result["medium"]["statistical_edge"]["volatility_threshold_fit"]
    assert thresholds["low"] == thresholds["high"] == 0.0
    assert result["medium"]["statistical_edge"]["oos_pass"] is False
    json.dumps(result, allow_nan=False)


def test_training_labels_crossing_holdout_boundary_are_purged(monkeypatch) -> None:
    # Fix the regime so every eligible index matches and counts expose leakage.
    monkeypatch.setattr(distillation_service, "_realized_volatility", lambda *args: 0.1)
    monkeypatch.setattr(distillation_service, "_rsi", lambda *args: 70.0)
    length = 1000
    split = int(length * 0.70)
    rows = [{"adjusted_close": 100 * 1.001**index} for index in range(length)]
    result = distillation_service._current_regime_edge(rows)
    for horizon in (5, 20):
        stats = result["horizons"][f"{horizon}d"]
        assert stats["train"]["sample_size"] == split - horizon - 200
        assert stats["out_of_sample"]["sample_size"] == length - horizon - split
        assert stats["purged_cross_boundary_labels"] == horizon
    assert result["simons_stat_veto"] is True
    assert result["validation_tier"] == "BASIC_CHRONOLOGICAL_HOLDOUT_ONLY"
