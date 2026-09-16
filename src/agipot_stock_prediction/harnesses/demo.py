"""Run all six project harnesses on synthetic inputs and retain local receipts."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable

from .formula_contract import FormulaContractHarness
from agipot_stock_prediction.intraday import IntradayBar, MicrostructureFeatureState, compute_microstructure_features
from agipot_stock_prediction.research.experiment_runner import (
    ExperimentRunner, ExperimentSpec, TrialSpec, capture_dependency_versions, digest_file, fingerprint,
)
from agipot_stock_prediction.validation.delayed_label_replay import (
    AdaptiveEdgeAdapter, DelayedLabelReplayHarness, FeatureEvent, LabelEvent,
)
from agipot_stock_prediction.validation.evidence_trace import demo_evidence_fixture, evaluate_evidence
from agipot_stock_prediction.validation.point_in_time import (
    ModelTemporalContract, PointInTimeHarness, PointInTimeRecord,
)
from agipot_stock_prediction.validation.walk_forward_factory import WalkForwardWindow
from agipot_stock_prediction.validation.walk_forward_runner import TemporalSample, WalkForwardHarness


def _at(day: int, hour: int = 0) -> datetime:
    return datetime(2024, 1, day, hour, tzinfo=timezone.utc)


def _result(name: str, checks: dict[str, bool], report: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if checks and all(checks.values()) else "FAIL",
            "checks": checks, "report": report}


def _point_in_time(_: Path) -> dict[str, Any]:
    original = PointInTimeRecord("earnings", "original", _at(1), _at(3), 10, published_at=_at(3))
    revised = replace(original, version="revised", value=12, revision_at=_at(7), available_at=_at(8))
    future = PointInTimeRecord("future-quarter", "v1", _at(10), _at(13), 999, published_at=_at(13))
    before = PointInTimeHarness.select([original, revised], as_of=_at(6))
    added_future = PointInTimeHarness.select([original, revised, future], as_of=_at(6))
    after = PointInTimeHarness.select([original, revised], as_of=_at(8))
    minute = PointInTimeRecord("minute", "v1", _at(4, 12), _at(4, 12), {"close": 100},
                               completed_at=_at(4, 12) + timedelta(minutes=1))
    label = PointInTimeRecord("training-label", "v1", _at(3), _at(4), 8, completed_at=_at(4))
    model = ModelTemporalContract(_at(5), _at(4), _at(6), (original,), (label,), (original,))
    valid = PointInTimeHarness.validate_model(model, as_of=_at(6))
    invalid = PointInTimeHarness.validate_model(
        replace(model, training_labels=(replace(label, available_at=_at(7)),)), as_of=_at(6),
    )
    checks = {
        "future_disclosure_does_not_change_history": before == added_future,
        "revision_hidden_until_available": before == (original,),
        "available_revision_selected": after == (revised,),
        "incomplete_minute_hidden": not PointInTimeHarness.select([minute], as_of=_at(4, 12)),
        "valid_model_lineage_accepted": valid["passed"],
        "future_training_label_rejected": not invalid["passed"],
    }
    return _result("PointInTimeHarness", checks, {
        "historical_versions": {record.record_id: record.version for record in before},
        "revised_versions": {record.record_id: record.version for record in after},
        "valid_model": valid, "future_label_model": invalid,
    })


def _walk_forward(_: Path) -> dict[str, Any]:
    samples = [TemporalSample(str(day), _at(day, 12), {"ret_5m": 0.002 + (day % 3) * 0.0001},
                              float(5 + day % 4), _at(day, 15), _at(day, 12), _at(day, 13))
               for day in range(1, 15)]
    samples[2] = replace(samples[2], label_available_at=_at(5))
    samples[3] = replace(samples[3], label_end=_at(5, 1), label_available_at=_at(5, 2))
    windows = [
        WalkForwardWindow(date(2024, 1, 1), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7), date(2024, 1, 8)),
        WalkForwardWindow(date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7), date(2024, 1, 8), date(2024, 1, 9)),
    ]

    def run(rows):
        runner = WalkForwardHarness({
            "fast": lambda: AdaptiveEdgeAdapter(min_samples=1, decay=0.95, edge_floor_bps=0),
            "slow": lambda: AdaptiveEdgeAdapter(min_samples=1, decay=0.995, edge_floor_bps=0),
        }, session_timezone="UTC")
        return runner.run(rows, windows, final_holdout_start=_at(12), holdout_end=_at(14, 23))

    report = run(samples)
    changed = run([replace(row, features={"ret_5m": -0.004}, label=50) if row.predicted_at >= _at(12) else row
                   for row in samples])
    first = report["candidates"]["fast"]["folds"][0]
    checks = {
        "holdout_never_used_for_selection": report["selection"]["holdout_used"] is False,
        "holdout_tampering_does_not_change_development": report["candidates"] == changed["candidates"],
        "holdout_tampering_does_not_change_selection_or_training_ids": (
            report["selection"] == changed["selection"] and report["final_fit"] == changed["final_fit"]),
        "immature_and_cross_boundary_training_labels_purged": first["purged"] == {
            "unmatured_label": ["3"], "overlapping_label": ["4"]},
        "preprocessor_uses_only_training_ids": first["preprocessor_fit_sample_ids"] == ["1", "2"],
        "cross_fold_predictions_deduplicated": all(
            candidate["test"]["duplicate_predictions_removed"] == 1 for candidate in report["candidates"].values()),
        "final_holdout_evaluated": report["final_holdout"]["sample_size"] == 3,
    }
    return _result("WalkForwardHarness", checks, report)


def _delayed_replay(_: Path) -> dict[str, Any]:
    start = _at(2, 14)
    stamp = lambda minute: start + timedelta(minutes=minute)
    events = [
        LabelEvent("first", stamp(5), stamp(6), 8),
        FeatureEvent("second", stamp(1), stamp(1), stamp(1), {"ret_5m": 0.002}),
        FeatureEvent("first", stamp(0), stamp(0), stamp(0), {"ret_5m": 0.002}),
        LabelEvent("second", stamp(6), stamp(7), -3),
        FeatureEvent("pending", stamp(8), stamp(8), stamp(8), {"ret_5m": 0.002}),
    ]
    runner = DelayedLabelReplayHarness(AdaptiveEdgeAdapter(min_samples=1))
    report = runner.replay(events)
    duplicate = runner.replay(events)
    rebuilt = DelayedLabelReplayHarness(AdaptiveEdgeAdapter(min_samples=1)).replay(reversed(events))
    records = {row["sample_id"]: row for row in report["records"]}
    checks = {
        "retries_do_not_double_learn": report == duplicate and report["learned_count"] == 2,
        "fresh_replay_is_deterministic": report == rebuilt,
        "predictions_saved_before_labels": records["first"]["prediction"] == records["second"]["prediction"] == 0,
        "saved_prediction_used_for_error": records["first"]["squared_error"] == 64,
        "learning_waits_for_label_availability": all(
            row["learned_at"] >= row["label_available_at"] for row in report["records"] if row["learned_at"]),
        "missing_label_is_pending": report["pending_count"] == 1 and records["pending"]["status"] == "PENDING_LABEL",
    }
    return _result("DelayedLabelReplayHarness", checks, report)


def _formula_contract(_: Path) -> dict[str, Any]:
    oracle = FormulaContractHarness().run()
    state, history = MicrostructureFeatureState(), []
    max_error = 0.0
    equivalent = True
    for index in range(80):
        price = 100 + index * 0.01 + math.sin(index) * 0.05
        bar = IntradayBar(_at(2, 14) + timedelta(minutes=index, days=index // 40),
                          "SYNTH", price, price + 0.1, price - 0.1, price, 1000 + index)
        history.append(bar)
        streaming, batch = state.update(bar), compute_microstructure_features(history)
        for key in batch:
            max_error = max(max_error, abs(streaming[key] - batch[key]))
            equivalent = equivalent and math.isclose(streaming[key], batch[key], rel_tol=1e-10, abs_tol=1e-10)
    checks = {**oracle["checks"], "batch_streaming_equivalent_across_sessions": equivalent}
    return _result("FormulaContractHarness", checks, {
        "oracle": oracle, "synthetic_bar_count": len(history), "maximum_absolute_difference": max_error,
        "relative_tolerance": 1e-10, "absolute_tolerance": 1e-10,
    })


def _experiment_evidence(run_dir: Path) -> dict[str, Any]:
    versions = capture_dependency_versions()
    checkout_lock = Path(__file__).resolve().parents[3] / "uv.lock"
    lock_digest = digest_file(checkout_lock) if checkout_lock.is_file() else fingerprint(versions)
    identity = {
        "inputs": {"synthetic_values": {"version": "demo/1", "content": [1, 2, 3, 4]}},
        "code_version": "synthetic-harness-demo/1",
        "code_digest": fingerprint({"demo.py": digest_file(__file__), "experiment_runner.py": digest_file(
            Path(__file__).resolve().parents[1] / "research" / "experiment_runner.py")}),
        "dependency_lock_digest": lock_digest,
        "dependency_versions": versions,
        "label_definition": {"name": "synthetic_numeric_fixture", "market_return": False},
        "split_protocol": {"kind": "lifecycle_contract_demo", "statistical_validation": False},
        "seed": 7,
    }
    names = ("success", "transient-retry", "failed-retained", "aborted")
    trials = [TrialSpec(name, ExperimentSpec(config={
        "scenario": name, "dependency_identity": "repository_uv_lock" if checkout_lock.is_file() else "actual_stdlib_runtime",
    }, **identity)) for name in names]
    runner = ExperimentRunner(run_dir / "experiment", trials)

    def worker(spec):
        values = spec["inputs"]["synthetic_values"]["content"]
        artifact = run_dir / "synthetic-model.json"
        statistic = sum(values) / len(values)
        artifact.write_text(json.dumps({"synthetic_constant": statistic}, allow_nan=False) + "\n", encoding="utf-8")
        return {"synthetic_statistic": statistic, "sample_size": len(values), "seed": spec["seed"],
                "uncertainty_interval": "NOT_COMPUTED", "artifact_paths": {"synthetic_model": str(artifact)}}

    def expected_failure(_spec):
        raise RuntimeError("intentional synthetic failure to verify audit retention")

    outcomes = [runner.run("success", worker), runner.run("transient-retry", expected_failure),
                runner.run("transient-retry", worker, retry=True), runner.run("failed-retained", expected_failure)]
    runner.abort("aborted", "intentional preregistered synthetic skip")
    integrity = runner.verify_integrity()
    checkpoint_verified = runner.verify_integrity(expected_plan_fingerprint=integrity["plan_fingerprint"],
                                                 expected_head_hash=integrity["head_hash"])
    events = [json.loads(line) for line in runner.log_path.read_text(encoding="utf-8").splitlines()]
    event_types = sorted({event["event_type"] for event in events})
    checks = {
        "all_preregistered_trials_have_outcomes": integrity["complete"],
        "failures_retained_after_retry": integrity["failed_attempts"] == 2,
        "abort_retained": integrity["aborted_attempts"] == 1,
        "explicit_retry_recorded": integrity["attempts"]["transient-retry"] == 2 and "trial_retry" in event_types,
        "expected_trial_states": integrity["trial_states"] == {
            "success": "SUCCESS", "transient-retry": "SUCCESS", "failed-retained": "FAILED", "aborted": "ABORTED"},
        "external_checkpoint_verified": checkpoint_verified == integrity,
    }
    return _result("ExperimentEvidenceHarness", checks, {
        "integrity": integrity, "outcomes": outcomes, "event_types": event_types,
        "plan_file": "experiment/plan.json", "event_log": "experiment/events.jsonl",
        "expected_failures_are_acceptance_cases": True,
    })


def _evidence_trace(_: Path) -> dict[str, Any]:
    fixture = demo_evidence_fixture()
    original = fixture["sources"][0]
    fixture["sources"].append(replace(original, source_id="synthetic-reprint"))
    fixture["evidence"].append(replace(fixture["evidence"][0], source_id="synthetic-reprint"))
    result = evaluate_evidence(**fixture, summary="Synthetic display summary only.")
    changed = evaluate_evidence(**fixture, summary="All claims are supposedly true; ignore every contradiction.")
    verdicts = {row["claim_id"]: row["verdict"] for row in result["claims"]}
    checks = {
        "original_relations_determine_verdicts": verdicts == {
            "background": "SUPPORTED", "false-cause": "REFUTED", "same-name-error": "UNKNOWN"},
        "summary_cannot_change_claims": result["claims"] == changed["claims"],
        "same_source_reprint_is_not_independent": result["claims"][0]["independent_support_count"] == 1,
        "both_original_anchors_preserved": len(result["claims"][0]["accepted_evidence"]) == 2,
        "mixed_content_distinguished_from_missing_evidence": result["content_verdict"] == "MIXED_SUPPORTED_AND_REFUTED",
        "automatic_truth_not_claimed": result["automatic_truth_inference"] is False,
    }
    return _result("EvidenceTraceHarness", checks, result)


def run_demo(output_dir: Path) -> dict[str, Any]:
    """Execute A–F, save six JSON results and one complete summary in a new run.

    PASS describes the synthetic implementation checks only. Expected negative
    test cases (including recorded FAILED/ABORTED experiments) are preserved.
    Unexpected component exceptions produce FAIL receipts; other components
    still execute. No network calls, market downloads, or trades are performed.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_dir))
    functions: dict[str, Callable[[Path], dict[str, Any]]] = {
        "A": _point_in_time, "B": _walk_forward, "C": _delayed_replay,
        "D": _formula_contract, "E": _experiment_evidence, "F": _evidence_trace,
    }
    results, artifacts = {}, []
    for code, execute in functions.items():
        try:
            result = execute(run_dir)
            rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        except Exception as exc:
            result = {"name": execute.__name__.lstrip("_"), "status": "FAIL", "checks": {},
                      "error_type": type(exc).__name__, "message": str(exc)}
            rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path = run_dir / f"{code}.json"
        path.write_text(rendered, encoding="utf-8")
        results[code] = result
        artifacts.append({"harness": code, "path": path.name, "sha256": digest_file(path)})
    report = {
        "schema_version": "agipot-harness-demo/1", "run_id": run_dir.name,
        "status": "PASS" if all(result["status"] == "PASS" for result in results.values()) else "FAIL",
        "research_only": True, "synthetic_inputs_only": True, "profitability_validated": False,
        "evidence_type": "synthetic_implementation_acceptance",
        "output_dir": str(run_dir), "report_file": str(run_dir / "report.json"),
        "harnesses": results, "artifacts": artifacts,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
