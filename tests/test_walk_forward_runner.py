from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from agipot_stock_prediction.validation.delayed_label_replay import AdaptiveEdgeAdapter
from agipot_stock_prediction.validation.point_in_time import TemporalContractError
from agipot_stock_prediction.validation.walk_forward_factory import WalkForwardWindow, build_walk_forward_windows
from agipot_stock_prediction.validation.walk_forward_runner import IdentityPreprocessor, TemporalSample, WalkForwardHarness


def at(day, hour=0):
    return datetime(2024, 1, day, hour, tzinfo=timezone.utc)


def samples():
    return [TemporalSample(str(day), at(day, 12), {"x": float(day), "ret_5m": 0.002}, 2,
                           at(day, 15), at(day, 12), at(day, 13)) for day in range(1, 15)]


def windows():
    return [WalkForwardWindow(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7), date(2024, 1, 8)),
            WalkForwardWindow(date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7), date(2024, 1, 8), date(2024, 1, 9))]


class ConstantModel:
    def __init__(self, value=2, fit_log=None):
        self.value = value
        self.fit_log = fit_log

    def fit(self, features, labels):
        if self.fit_log is not None:
            self.fit_log.append((features, labels))

    def predict(self, features):
        return self.value


def run(harness, rows=None, folds=None):
    return harness.run(rows or samples(), folds or windows(), final_holdout_start=at(12), holdout_end=at(14, 23))


def test_validation_selects_candidate_and_holdout_cannot_change_fitting():
    def experiment(rows):
        fits, preprocess_fits = [], []

        class Center:
            def fit(self, features):
                self.mean = sum(row["x"] for row in features) / len(features)
                preprocess_fits.append(self.mean)

            def transform(self, row):
                return {**row, "x": row["x"] - self.mean}

        harness = WalkForwardHarness({"bad": lambda: ConstantModel(10, fits), "good": lambda: ConstantModel(2, fits)},
                                     preprocessor_factory=Center, session_timezone="UTC")
        return run(harness, rows), fits, preprocess_fits

    rows = samples()
    baseline, fitted, preprocessing = experiment(rows)
    changed = [replace(row, features={"x": 99999.0}, label=10) if row.predicted_at >= at(12) else row for row in rows]
    altered, altered_fits, altered_preprocessing = experiment(changed)
    assert baseline["selection"] == altered["selection"] == {
        "candidate": "good", "criterion": "validation_mean_squared_error_only", "holdout_used": False,
    }
    assert fitted == altered_fits
    assert preprocessing == altered_preprocessing
    assert baseline["final_holdout"]["mean_squared_error"] == 0
    assert altered["final_holdout"]["mean_squared_error"] == 64
    assert preprocessing[:2] == [2.5, 3.0]


def test_train_labels_must_be_mature_and_not_overlap_next_partition():
    rows = samples()
    rows[2] = replace(rows[2], label_available_at=at(5))
    rows[3] = replace(rows[3], label_end=at(5, 1), label_available_at=at(5, 2))
    report = run(WalkForwardHarness(ConstantModel, session_timezone="UTC"), rows)
    first = report["candidates"]["default"]["folds"][0]
    assert first["training_sample_ids"] == ["1", "2"]
    assert first["preprocessor_fit_sample_ids"] == ["1", "2"]
    assert first["purged"] == {"unmatured_label": ["3"], "overlapping_label": ["4"]}
    assert all(stamp <= first["training_cutoff"] for stamp in first["training_label_available_at"])


def test_embargo_is_elapsed_time_not_sample_count():
    report = run(WalkForwardHarness(ConstantModel, session_timezone="UTC", embargo=timedelta(hours=30)))
    fold = report["candidates"]["default"]["folds"][0]
    assert fold["training_sample_ids"] == ["1", "2", "3"]
    assert fold["training_cutoff"] == at(3, 18).isoformat()
    assert report["embargo_seconds"] == 108000


def test_cross_fold_predictions_are_deduplicated_before_aggregate_metrics():
    result = run(WalkForwardHarness(ConstantModel, session_timezone="UTC"))
    candidate = result["candidates"]["default"]
    assert candidate["validation"]["sample_size"] == 3
    assert candidate["test"]["sample_size"] == 3
    assert candidate["validation"]["duplicate_predictions_removed"] == 1
    assert candidate["test"]["duplicate_predictions_removed"] == 1
    assert next(row for row in candidate["test"]["predictions"] if row["sample_id"] == "8")["fold"] == 0


@pytest.mark.parametrize("reuse_model", [True, False])
def test_each_fold_requires_fresh_models_and_preprocessors(reuse_model):
    shared = ConstantModel() if reuse_model else IdentityPreprocessor()
    harness = WalkForwardHarness((lambda: shared) if reuse_model else ConstantModel,
                                 preprocessor_factory=IdentityPreprocessor if reuse_model else lambda: shared,
                                 session_timezone="UTC")
    with pytest.raises(TemporalContractError, match="reused"):
        run(harness)


def test_development_window_cannot_include_frozen_holdout():
    contaminated = [replace(windows()[0], test_end=date(2024, 1, 12))]
    with pytest.raises(TemporalContractError, match="frozen holdout"):
        run(WalkForwardHarness(ConstantModel, session_timezone="UTC"), folds=contaminated)


def test_pending_holdout_labels_do_not_become_metrics():
    rows = samples()
    rows[-1] = replace(rows[-1], label=None, label_available_at=None)
    result = run(WalkForwardHarness(ConstantModel, session_timezone="UTC"), rows)
    assert result["final_holdout"]["sample_size"] == 2
    assert result["final_holdout"]["pending_labels"] == 1


def test_validation_labels_maturing_in_holdout_cannot_select_a_model():
    rows = [replace(row, label_available_at=at(13)) if row.sample_id in {"5", "6", "7"} else row for row in samples()]
    with pytest.raises(TemporalContractError, match="no mature validation"):
        run(WalkForwardHarness(ConstantModel, session_timezone="UTC"), rows)


def test_validation_label_overlapping_test_is_not_used_for_selection():
    rows = samples()
    rows[5] = replace(rows[5], label_end=at(7, 1), label_available_at=at(7, 2), label=999)
    report = run(WalkForwardHarness(ConstantModel, session_timezone="UTC"), rows)
    first_validation = report["candidates"]["default"]["folds"][0]["validation"]
    assert first_validation["purged_labels"] == 1
    assert first_validation["sample_size"] == 1
    assert first_validation["mean_squared_error"] == 0


def test_test_label_reaching_holdout_is_excluded_from_development_metrics():
    rows = samples()
    rows[7] = replace(rows[7], label_end=at(12), label_available_at=at(12, 1), label=999)
    report = run(WalkForwardHarness(ConstantModel, session_timezone="UTC"), rows)
    first_test = report["candidates"]["default"]["folds"][0]["test"]
    assert first_test["purged_labels"] == 1
    assert first_test["sample_size"] == 1
    assert "8" in report["final_fit"]["purged"]["overlapping_label"]


def test_adaptive_edge_model_adapter_runs_each_fold_and_final_refit():
    result = run(WalkForwardHarness(lambda: AdaptiveEdgeAdapter(min_samples=1), session_timezone="UTC"))
    assert result["final_holdout"]["sample_size"] == 3
    assert result["formal_pbo_dsr"] == "not_computed"


def test_temporal_sample_rejects_unavailable_feature_and_unmatured_label_contract():
    row = samples()[0]
    with pytest.raises(TemporalContractError, match="features were not available"):
        replace(row, feature_available_at=at(2))
    with pytest.raises(TemporalContractError, match="label_available_at"):
        replace(row, label_available_at=row.predicted_at)


def test_existing_calendar_factory_preserves_actual_sessions_around_leap_and_holiday():
    # Calendar-generated dates include leap day and exclude the US July holiday;
    # the window factory uses actual supplied sessions rather than inventing dates.
    from agipot_stock_prediction.calendar import is_us_equity_session, us_equity_session_bounds

    day, stop = date(2023, 1, 3), date(2025, 4, 30)
    sessions = []
    while day <= stop:
        if is_us_equity_session(day):
            sessions.append(day)
        day += timedelta(days=1)
    assert date(2024, 2, 29) in sessions
    assert date(2024, 7, 4) not in sessions
    assert us_equity_session_bounds(date(2024, 11, 29))[1].hour == 13
    built = build_walk_forward_windows(sessions, train_years=1, validation_months=1, test_months=1, step_months=1)
    assert built
    assert all(endpoint in sessions for window in built for endpoint in vars(window).values())
    assert all(window.train_end < window.validation_start <= window.validation_end < window.test_start for window in built)


@pytest.mark.parametrize("days", [
    [date(2024, 2, 27), date(2024, 2, 28), date(2024, 2, 29), date(2024, 3, 1), date(2024, 3, 4)],
    [date(2024, 11, 25), date(2024, 11, 27), date(2024, 11, 29), date(2024, 12, 2), date(2024, 12, 3)],
])
def test_runner_accepts_leap_day_and_half_day_endpoints_without_inventing_holiday_bars(days):
    from agipot_stock_prediction.calendar import us_equity_session_bounds

    rows = []
    for index, day in enumerate(days):
        _, close = us_equity_session_bounds(day)
        prediction = close - timedelta(hours=1)
        rows.append(TemporalSample(str(day), prediction, {"x": float(index)}, 2,
                                   close + timedelta(minutes=1), prediction, close))
    window = WalkForwardWindow(days[0], days[1], days[2], days[2], days[3], days[3])
    _, final_close = us_equity_session_bounds(days[-1])
    holdout_start = final_close.replace(hour=0, minute=0, second=0)
    report = WalkForwardHarness(ConstantModel).run(rows, [window], final_holdout_start=holdout_start,
                                                  holdout_end=final_close + timedelta(minutes=2))
    assert report["candidates"]["default"]["validation"]["sample_size"] == 1
    assert report["final_holdout"]["sample_size"] == 1
    assert report["final_fit"]["training_sample_ids"] == [str(day) for day in days[:-1]]
