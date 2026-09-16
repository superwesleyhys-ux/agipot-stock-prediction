from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from agipot_stock_prediction.research.experiment_runner import (
    ExperimentIntegrityError,
    ExperimentRunner,
    TrialSpec,
    canonical_json,
    demo_experiment_specs,
    fingerprint,
)


@pytest.fixture
def spec():
    return demo_experiment_specs()[0].spec


def test_fingerprint_covers_every_declared_influence(spec):
    changes = [
        {"inputs": {"prices": {"version": "v2", "content": [100, 102]}}},
        {"config": {"threshold": 0.3}},
        {"code_version": "other-commit"},
        {"code_digest": fingerprint("dirty source code")},
        {"dependency_lock_digest": fingerprint("different-lock")},
        {"dependency_versions": {"python": "3.13.0"}},
        {"label_definition": {"horizon": 5}},
        {"split_protocol": {"train": [0], "test": [1]}},
        {"seed": 8},
        {"input_scope": "all_raw_records"},
    ]
    assert all(replace(spec, **change).fingerprint() != spec.fingerprint() for change in changes)
    for name in ("minute_rows", "fundamentals", "tick_rows", "model_weights"):
        inputs = {**spec.inputs, name: {"version": "v1", "content": {"value": 1}}}
        before = replace(spec, inputs=inputs)
        after = replace(spec, inputs={**inputs, name: {"version": "v1", "content": {"value": 2}}})
        assert before.fingerprint() != after.fingerprint()


def test_input_versions_and_content_are_both_part_of_identity(spec):
    source = next(iter(spec.inputs))
    changed = {**spec.inputs[source], "version": "revised-version"}
    assert replace(spec, inputs={source: changed}).fingerprint() != spec.fingerprint()


def test_canonical_mapping_order_and_equivalent_timezones():
    a = {"a": 1, "b": [2, 3]}
    assert fingerprint(a) == fingerprint({"b": [2, 3], "a": 1})
    assert fingerprint(a) != fingerprint({"a": 1, "b": [3, 2]})
    assert canonical_json(datetime(2025, 1, 1, tzinfo=timezone.utc)) == '"2025-01-01T00:00:00+00:00"'


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), {1: "ambiguous-key"}, {1, 2}, datetime(2025, 1, 1)])
def test_noncanonical_values_fail_closed(invalid):
    with pytest.raises(ValueError):
        fingerprint(invalid)


def test_declared_used_only_scope_does_not_hash_future_unused_rows(spec):
    # Selection is an explicit manifest policy, not a hidden runner behavior.
    records = [{"available_at": 1, "value": 100}, {"available_at": 2, "value": 101}]

    def manifest(rows):
        return replace(spec, inputs={"prices": {
            "version": "visible-snapshot-v1", "content": [row for row in rows if row["available_at"] <= 2],
        }}, config={"as_of": 2}, input_scope="available_at <= config.as_of; hash used records only")

    assert manifest(records).fingerprint() == manifest(records + [{"available_at": 3, "value": 999}]).fingerprint()


def test_runner_preserves_success_failure_and_explicit_retry(tmp_path):
    trials = demo_experiment_specs()
    received = []
    runner = ExperimentRunner(tmp_path, trials, on_event=received.append)
    assert runner.run(trials[0].trial_id, lambda spec: {"metric": 1, "seed": spec["seed"]})["status"] == "SUCCESS"

    def fail(_):
        raise RuntimeError("synthetic trial failure")

    assert runner.run(trials[1].trial_id, fail)["status"] == "FAILED"
    with pytest.raises(ValueError, match="retry=True"):
        runner.run(trials[1].trial_id, lambda spec: {})
    retried = runner.run(trials[1].trial_id, lambda spec: {"metric": 0}, retry=True)
    assert retried["status"] == "SUCCESS" and retried["attempt"] == 2
    report = runner.verify_integrity()
    assert report["complete"] and report["failed_attempts"] == 1
    assert report["formal_pbo"] == report["formal_dsr"] == "NOT_COMPUTED"
    assert any(event["event_type"] == "trial_failed" for event in received)
    assert any(event["event_type"] == "trial_retry" for event in received)
    assert ExperimentRunner(tmp_path).verify_integrity() == report


def test_every_trial_must_be_registered_before_execution(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec), TrialSpec("b", spec)])
    with pytest.raises(ValueError, match="not preregistered"):
        runner.run("c", lambda spec: {})
    with pytest.raises(ExperimentIntegrityError, match="terminal outcome"):
        runner.verify_integrity()
    with pytest.raises(ExperimentIntegrityError, match="cannot be replaced"):
        ExperimentRunner(tmp_path, [TrialSpec("a", spec)])


def test_abort_and_interrupt_are_retained(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("skip", spec), TrialSpec("interrupt", spec)])
    runner.abort("skip", "preregistered budget exhausted")

    def interrupted(_):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        runner.run("interrupt", interrupted)
    report = runner.verify_integrity()
    assert report["trial_states"] == {"skip": "ABORTED", "interrupt": "ABORTED"}
    assert report["aborted_attempts"] == 2


def _write_rechained(path, events):
    previous = "0" * 64
    for index, event in enumerate(events):
        event.pop("event_hash", None)
        event["sequence"], event["previous_hash"] = index, previous
        event["event_hash"] = fingerprint(event)
        previous = event["event_hash"]
    path.write_text("".join(canonical_json(event) + "\n" for event in events))


def test_missing_failed_trial_fails_accounting_even_after_rehashing(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("good", spec), TrialSpec("bad", spec)])
    runner.run("good", lambda spec: {})
    runner.run("bad", lambda spec: 1 / 0)
    events = [json.loads(line) for line in runner.log_path.read_text().splitlines()]
    _write_rechained(runner.log_path, [event for event in events if event["trial_id"] != "bad"])
    with pytest.raises(ExperimentIntegrityError, match="missing from"):
        runner.verify_integrity()


def test_missing_failed_terminal_record_cannot_look_complete(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("bad", spec)])
    runner.run("bad", lambda spec: 1 / 0)
    lines = runner.log_path.read_text().splitlines(keepends=True)
    runner.log_path.write_text("".join(lines[:-1]))
    with pytest.raises(ExperimentIntegrityError, match="terminal outcome"):
        runner.verify_integrity()


def test_hash_chain_detects_changed_event_content(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)])
    runner.run("a", lambda spec: {"metric": 1})
    runner.log_path.write_text(runner.log_path.read_text().replace('"metric":1', '"metric":9'))
    with pytest.raises(ExperimentIntegrityError, match="hash-chain"):
        runner.verify_integrity()


def test_external_head_checkpoint_detects_valid_prefix_rollback(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)])
    runner.run("a", lambda spec: 1 / 0)
    prefix = runner.log_path.read_text()
    runner.run("a", lambda spec: {"metric": 1}, retry=True)
    checkpoint = runner.verify_integrity()
    runner.log_path.write_text(prefix)
    with pytest.raises(ExperimentIntegrityError, match="external checkpoint"):
        runner.verify_integrity(expected_head_hash=checkpoint["head_hash"])


def test_callback_failure_does_not_erase_local_outcome(tmp_path, spec):
    def unavailable(_):
        raise ConnectionError("tracker offline")

    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)], on_event=unavailable)
    assert runner.run("a", lambda spec: {"metric": 1})["status"] == "SUCCESS"
    assert runner.verify_integrity()["integration_errors"] == 4


def test_worker_cannot_mutate_registered_manifest(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)])
    before = runner.plan_path.read_bytes()

    def work(snapshot):
        snapshot["config"]["threshold"] = 999
        return {"metric": 1}

    runner.run("a", work)
    assert runner.plan_path.read_bytes() == before


def test_model_artifacts_are_copied_hashed_and_verified(tmp_path, spec):
    original = tmp_path / "model.json"
    original.write_text('{"coefficient": 2}')
    runner = ExperimentRunner(tmp_path / "runs", [TrialSpec("a", spec)])
    outcome = runner.run("a", lambda spec: {"artifact_paths": {"model": str(original)}, "metrics": {"mae": 0.1}})
    artifact = outcome["result"]["artifacts"]["model"]
    original.write_text("original file can change after capture")
    assert runner.verify_integrity()["complete"]
    (runner.directory / artifact["path"]).write_text("tampered")
    with pytest.raises(ExperimentIntegrityError, match="artifact"):
        runner.verify_integrity()


def test_nonfinite_worker_metrics_are_recorded_as_failure(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)])
    assert runner.run("a", lambda spec: {"metric": float("nan")})["status"] == "FAILED"
    assert runner.verify_integrity()["failed_attempts"] == 1


@pytest.mark.parametrize("record", ["[]\n", "null\n", '{"event_type":"trial_succeeded"}\n'])
def test_malformed_records_report_integrity_errors(tmp_path, spec, record):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)])
    runner.log_path.write_text(record)
    with pytest.raises(ExperimentIntegrityError):
        runner.verify_integrity()


def test_started_trial_must_reference_registered_specification(tmp_path, spec):
    runner = ExperimentRunner(tmp_path, [TrialSpec("a", spec)])
    runner.run("a", lambda spec: {})
    events = [json.loads(line) for line in runner.log_path.read_text().splitlines()]
    events[2]["payload"]["spec_fingerprint"] = fingerprint("different-specification")
    _write_rechained(runner.log_path, events)
    with pytest.raises(ExperimentIntegrityError, match="invalid start"):
        runner.verify_integrity()
