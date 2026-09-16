from datetime import datetime, timedelta, timezone

import pytest

from agipot_stock_prediction.validation.point_in_time import (
    ModelTemporalContract, PointInTimeHarness, PointInTimeRecord, TemporalContractError,
)


def at(day, hour=0):
    return datetime(2024, 2, day, hour, tzinfo=timezone.utc)


def record(name="earnings", version="v1", **kwargs):
    return PointInTimeRecord(name, version, kwargs.pop("event_time", at(1)),
                            kwargs.pop("available_at", at(3)), kwargs.pop("value", 10), **kwargs)


def test_future_financial_disclosure_does_not_change_historical_selection():
    original = record(published_at=at(3))
    future = record("next-quarter", available_at=at(20), published_at=at(20), value=999)
    assert PointInTimeHarness.select([original, future], as_of=at(10)) == (original,)


def test_revision_requires_both_disclosure_and_vendor_availability():
    original = record(published_at=at(3))
    revised = record(version="v2", available_at=at(7), published_at=at(3), revision_at=at(6), value=12)
    assert PointInTimeHarness.select([revised, original], as_of=at(6)) == (original,)
    assert PointInTimeHarness.select([original, revised], as_of=at(7)) == (revised,)


def test_delayed_old_publication_does_not_replace_newer_revision():
    old = record(available_at=at(9), published_at=at(3))
    new = record(version="v2", available_at=at(8), published_at=at(3), revision_at=at(7))
    assert PointInTimeHarness.select([new, old], as_of=at(10)) == (new,)


def test_minute_observation_is_hidden_until_completion_even_if_delivered_early():
    minute = record("bar", event_time=at(1), available_at=at(1), completed_at=at(1) + timedelta(minutes=1))
    assert PointInTimeHarness.select([minute], as_of=at(1)) == ()
    assert PointInTimeHarness.select([minute], as_of=minute.completed_at) == (minute,)


def test_event_time_itself_cannot_be_in_the_future():
    future = record(event_time=at(10), available_at=at(3))
    assert PointInTimeHarness.select([future], as_of=at(5)) == ()


def test_conflicting_versions_fail_and_identical_retries_are_idempotent():
    first = record()
    assert PointInTimeHarness.select([first, first], as_of=at(10)) == (first,)
    with pytest.raises(TemporalContractError, match="conflicting"):
        PointInTimeHarness.select([first, record(value=11)], as_of=at(10))
    with pytest.raises(TemporalContractError, match="ambiguous"):
        PointInTimeHarness.select([first, record(version="v2")], as_of=at(10))


@pytest.mark.parametrize("group", ["training_records", "training_labels", "preprocessing_records"])
def test_model_cannot_use_future_training_labels_or_preprocessing(group):
    contract = ModelTemporalContract(
        training_cutoff=at(5), trained_through=at(4), model_available_at=at(6),
        **{group: (record(available_at=at(7)),)},
    )
    result = PointInTimeHarness.validate_model(contract, as_of=at(10))
    assert not result["passed"]
    assert f"{group}:earnings:not_available_at_training_cutoff" in result["blockers"]


def test_model_training_date_alone_does_not_prove_availability():
    contract = ModelTemporalContract(at(5), at(4), at(12))
    assert PointInTimeHarness.validate_model(contract, as_of=at(10))["blockers"] == ["model_not_available_at_inference"]


def test_valid_model_contract_reports_declared_lineage_without_claiming_completeness():
    contract = ModelTemporalContract(at(5), at(4), at(6), training_records=(record(),), training_labels=(record("label"),))
    result = PointInTimeHarness.validate_model(contract, as_of=at(6))
    assert result["passed"]
    assert result["declared_input_counts"]["training_labels"] == 1
    assert result["lineage_completeness"] == "caller_declared_not_independently_verified"


def test_point_in_time_rejects_naive_datetimes():
    with pytest.raises(TemporalContractError, match="timezone"):
        record(available_at=datetime(2024, 2, 3))
