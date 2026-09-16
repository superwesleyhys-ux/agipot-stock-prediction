from datetime import datetime, timedelta, timezone

import pytest

from agipot_stock_prediction.validation.delayed_label_replay import (
    AdaptiveEdgeAdapter, DelayedLabelReplayHarness, FeatureEvent, LabelEvent,
)
from agipot_stock_prediction.validation.point_in_time import TemporalContractError


START = datetime(2024, 2, 29, 14, 30, tzinfo=timezone.utc)


def at(minutes):
    return START + timedelta(minutes=minutes)


def feature(name, minute=0, **kwargs):
    return FeatureEvent(name, at(minute), kwargs.get("available_at", at(minute)),
                        kwargs.get("completed_at", at(minute)), {"ret_5m": 0.002})


class CountingModel:
    def __init__(self):
        self.updates = []
        self.predictions = 0

    def predict_one(self, features):
        self.predictions += 1
        return float(len(self.updates))

    def learn_one(self, features, label):
        self.updates.append(label)


def test_no_learning_before_label_maturity_and_evaluation_uses_saved_prediction():
    model = CountingModel()
    harness = DelayedLabelReplayHarness(model)
    harness.process(feature("a"))
    harness.process(feature("b", 1))
    assert model.updates == []
    label = LabelEvent("b", at(2), at(2), 3)
    harness.process(label)
    harness.process(LabelEvent("a", at(3), at(4), 10))
    records = {row["sample_id"]: row for row in harness.report()["records"]}
    assert records["a"]["prediction"] == 0
    assert records["a"]["squared_error"] == 100
    assert records["a"]["learned_at"] == at(4).isoformat()
    assert model.predictions == 2
    assert model.updates == [3.0, 10.0]


def test_incomplete_feature_and_immature_label_are_not_processed_early():
    harness = DelayedLabelReplayHarness(CountingModel())
    event = feature("a", completed_at=at(1))
    with pytest.raises(TemporalContractError, match="not completed"):
        harness.process(event, now=at(0))
    harness.process(event, now=at(1))
    with pytest.raises(TemporalContractError, match="not completed"):
        harness.process(LabelEvent("a", at(2), at(3), 1), now=at(2))
    assert harness.report()["learned_count"] == 0


def test_duplicates_are_idempotent_including_replaying_entire_event_log():
    model = CountingModel()
    events = [feature("a"), LabelEvent("a", at(1), at(2), 2)]
    harness = DelayedLabelReplayHarness(model)
    first = harness.replay(events + events)
    assert harness.replay(events) == first
    assert model.updates == [2.0]
    assert model.predictions == 1


def test_fresh_replays_are_deterministic_and_bulk_order_is_explicit():
    events = [LabelEvent("a", at(2), at(3), 10), feature("b", 1), feature("a"), LabelEvent("b", at(3), at(4), -3)]
    first = DelayedLabelReplayHarness(AdaptiveEdgeAdapter(min_samples=1)).replay(events)
    second = DelayedLabelReplayHarness(AdaptiveEdgeAdapter(min_samples=1)).replay(reversed(events))
    assert first == second
    assert first["learned_count"] == 2


def test_streaming_out_of_order_and_conflicting_duplicate_fail():
    harness = DelayedLabelReplayHarness(CountingModel())
    harness.process(feature("a", 2))
    with pytest.raises(TemporalContractError, match="out-of-order"):
        harness.process(feature("b", 1))
    with pytest.raises(TemporalContractError, match="conflicting"):
        harness.process(feature("a", 3))


def test_same_timestamp_features_are_predicted_before_any_labels():
    events = [feature("a"), LabelEvent("a", at(2), at(2), 99), feature("b", 2)]
    report = DelayedLabelReplayHarness(CountingModel()).replay(events)
    assert {row["sample_id"]: row["prediction"] for row in report["records"]} == {"a": 0, "b": 0}


def test_missing_label_is_pending_and_orphan_label_is_an_error():
    harness = DelayedLabelReplayHarness(CountingModel())
    assert harness.replay([feature("a")])["pending_count"] == 1
    with pytest.raises(TemporalContractError, match="no saved prediction"):
        harness.process(LabelEvent("unknown", at(2), at(2), 1))


def test_failed_model_update_poisoned_runner_cannot_silently_retry():
    class FailingModel(CountingModel):
        def learn_one(self, features, label):
            super().learn_one(features, label)
            raise RuntimeError("partial update")

    harness = DelayedLabelReplayHarness(FailingModel())
    harness.process(feature("a"))
    label = LabelEvent("a", at(1), at(1), 1)
    with pytest.raises(RuntimeError, match="partial"):
        harness.process(label)
    with pytest.raises(TemporalContractError, match="fresh model"):
        harness.process(label)
    assert harness.report()["records"][0]["status"] == "LEARN_FAILED_RESTART_REQUIRED"


def test_interrupted_partial_learning_requires_a_fresh_model():
    class InterruptedModel(CountingModel):
        def learn_one(self, features, label):
            super().learn_one(features, label)
            raise KeyboardInterrupt()

    model = InterruptedModel()
    harness = DelayedLabelReplayHarness(model)
    harness.process(feature("a"))
    label = LabelEvent("a", at(1), at(1), 1)
    with pytest.raises(KeyboardInterrupt):
        harness.process(label)
    with pytest.raises(TemporalContractError, match="fresh model"):
        harness.process(label)
    assert model.updates == [1.0]
    assert harness.report()["restart_required"] is True
    assert harness.report()["failure_stage"] == "learning"


@pytest.mark.parametrize("exception", [RuntimeError("partially mutated predictor"), KeyboardInterrupt()])
def test_failed_stateful_prediction_cannot_be_retried(exception):
    class FailingPredictor(CountingModel):
        def predict_one(self, features):
            super().predict_one(features)
            raise exception

    model = FailingPredictor()
    harness = DelayedLabelReplayHarness(model)
    with pytest.raises(type(exception)):
        harness.process(feature("a"))
    with pytest.raises(TemporalContractError, match="fresh model"):
        harness.process(feature("b", 1))
    assert model.predictions == 1
    assert harness.report()["failure_stage"] == "prediction"
