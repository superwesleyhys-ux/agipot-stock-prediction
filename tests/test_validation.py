"""Chronology and evidence regressions; all data here is synthetic."""

from datetime import date, timedelta
import math

import pytest

from agipot_stock_prediction.validation.overfit_guard import evaluate_overfit_guard
from agipot_stock_prediction.validation.result_ranker import rank_trials
from agipot_stock_prediction.validation.walk_forward_factory import build_walk_forward_windows


def calendar_days(start, end):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def test_calendar_windows_honor_all_periods_without_partition_overlap():
    sessions = calendar_days(date(2020, 1, 1), date(2022, 12, 31))
    windows = build_walk_forward_windows(sessions, train_years=1, validation_months=3,
                                         test_months=6, step_months=3)
    assert len(windows) == 6
    first = windows[0]
    assert (first.train_start, first.train_end) == (date(2020, 1, 1), date(2020, 12, 31))
    assert (first.validation_start, first.validation_end) == (date(2021, 1, 1), date(2021, 3, 31))
    assert (first.test_start, first.test_end) == (date(2021, 4, 1), date(2021, 9, 30))
    assert windows[1].train_start == date(2020, 4, 1)
    for window in windows:
        assert window.train_end < window.validation_start <= window.validation_end < window.test_start
        assert window.test_end <= sessions[-1]


def test_incomplete_test_interval_is_omitted():
    sessions = calendar_days(date(2020, 1, 1), date(2021, 9, 29))
    assert build_walk_forward_windows(sessions, train_years=1, validation_months=3, test_months=6) == ()


def test_missing_validation_partition_is_not_replaced_by_training_endpoint():
    sessions = calendar_days(date(2020, 1, 1), date(2020, 12, 31))
    sessions += calendar_days(date(2021, 4, 1), date(2021, 9, 30))
    assert build_walk_forward_windows(sessions, train_years=1, validation_months=3, test_months=6) == ()


@pytest.mark.parametrize("configuration", [{"train_years": 0}, {"validation_months": 0},
                                          {"test_months": -1}, {"step_months": 0}, {"train_years": True}])
def test_bad_window_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        build_walk_forward_windows([], **configuration)


def test_calendar_windows_handle_leap_day_and_duplicate_sessions():
    sessions = calendar_days(date(2020, 2, 29), date(2022, 12, 31))
    result = build_walk_forward_windows(sessions + sessions[:20], train_years=1,
                                        validation_months=1, test_months=1, step_months=6)
    assert result[0].validation_start == date(2021, 2, 28)
    assert result[0].train_end == date(2021, 2, 27)
    assert result[0].validation_end == date(2021, 3, 28)


@pytest.fixture
def trial():
    return {"mandate_passed": True, "geometric_daily_return": 0.001,
            "symbols": ["SYNTH_A", "SYNTH_B"], "yearly_results": [{"year": 2020}, {"year": 2021}]}


def test_proxy_screen_reports_its_actual_evidence_level(trial):
    result = evaluate_overfit_guard([trial], all_trials_recorded=True)
    assert result.passed
    public = result.public_dict()
    assert public["validation_level"] == "heuristic_only"
    assert public["formal_pbo_computed"] is False
    assert public["formal_dsr_computed"] is False
    assert public["out_of_sample_verified"] is False
    assert "not formal PBO" in " ".join(public["limitations"])
    assert "not deflated Sharpe" in " ".join(public["limitations"])


def test_empty_trials_cannot_pass():
    result = evaluate_overfit_guard([], all_trials_recorded=True)
    assert not result.passed
    assert "no_trials" in result.blockers


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, True])
def test_nonfinite_trial_cannot_pass_overfit_screen(trial, bad):
    with pytest.raises(ValueError):
        evaluate_overfit_guard([trial | {"geometric_daily_return": bad}], all_trials_recorded=True)


@pytest.mark.parametrize("key", ["geometric_daily_return", "mandate_passed", "symbols", "yearly_results"])
def test_missing_trial_evidence_is_rejected(trial, key):
    del trial[key]
    with pytest.raises(ValueError):
        evaluate_overfit_guard([trial], all_trials_recorded=True)


def test_unrecorded_search_and_single_slice_are_blocked(trial):
    trial["symbols"] = ["SYNTH_A"]
    trial["yearly_results"] = [{"year": 2020}]
    result = evaluate_overfit_guard([trial], all_trials_recorded=False)
    assert not result.passed
    assert {"all_trials_not_recorded", "single_slice_overfit_risk"} <= set(result.blockers)


def test_zero_drawdown_beats_twenty_percent_at_equal_return():
    base = {"promotion_allowed": False, "geometric_daily_return": 0.001, "fill_rate": 1}
    flat = base | {"max_drawdown_pct": 0, "id": "zero_drawdown"}
    risky = base | {"max_drawdown_pct": 20, "id": "twenty_percent"}
    assert rank_trials([risky, flat])[0]["id"] == "zero_drawdown"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None])
def test_ranking_rejects_nonfinite_metrics(bad):
    with pytest.raises(ValueError):
        rank_trials([{"promotion_allowed": False, "geometric_daily_return": bad,
                      "max_drawdown_pct": 0, "fill_rate": 1}])


def test_ranking_does_not_make_up_missing_metrics():
    with pytest.raises(ValueError, match="required"):
        rank_trials([{"promotion_allowed": False, "geometric_daily_return": 0.1, "fill_rate": 1}])
