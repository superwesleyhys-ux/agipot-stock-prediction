import json
from pathlib import Path

from agipot_stock_prediction.harnesses import demo
from agipot_stock_prediction.harnesses.cli import main
from agipot_stock_prediction.research.experiment_runner import ExperimentRunner, digest_file


def _strict_json(path):
    def reject(value):
        raise AssertionError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def test_demo_executes_six_real_harnesses_and_retains_complete_json_receipts(tmp_path):
    report = demo.run_demo(tmp_path)
    assert report["status"] == "PASS"
    assert report["research_only"] and report["synthetic_inputs_only"]
    assert report["profitability_validated"] is False
    assert set(report["harnesses"]) == set("ABCDEF")
    assert len(report["artifacts"]) == 6
    root = Path(report["output_dir"])
    assert root.parent == tmp_path.resolve()
    assert _strict_json(Path(report["report_file"])) == report
    for artifact in report["artifacts"]:
        path = root / artifact["path"]
        assert path.resolve().is_relative_to(root)
        assert digest_file(path) == artifact["sha256"]
        payload = _strict_json(path)
        assert payload == report["harnesses"][artifact["harness"]]
        assert payload["checks"] and all(payload["checks"].values())

    a, b, c, d, e, f = [report["harnesses"][code]["report"] for code in "ABCDEF"]
    assert not a["future_label_model"]["passed"]
    assert b["selection"]["holdout_used"] is False
    assert b["final_holdout"]["sample_size"] == 3
    assert c["learned_count"] == 2 and c["pending_count"] == 1
    assert d["synthetic_bar_count"] == 80
    assert e["integrity"]["failed_attempts"] == 2 and e["integrity"]["aborted_attempts"] == 1
    assert {"trial_failed", "trial_retry", "trial_aborted", "trial_succeeded"} <= set(e["event_types"])
    assert ExperimentRunner(root / "experiment").verify_integrity() == e["integrity"]
    assert f["content_verdict"] == "MIXED_SUPPORTED_AND_REFUTED"
    assert f["claims"][0]["independent_support_count"] == 1


def test_repeat_demo_uses_isolated_runs_without_overwriting_previous_evidence(tmp_path):
    first = demo.run_demo(tmp_path)
    receipt = Path(first["report_file"]).read_bytes()
    event_log = Path(first["output_dir"]) / "experiment" / "events.jsonl"
    events = event_log.read_bytes()
    second = demo.run_demo(tmp_path)
    assert first["run_id"] != second["run_id"]
    assert first["output_dir"] != second["output_dir"]
    assert Path(first["report_file"]).read_bytes() == receipt
    assert event_log.read_bytes() == events
    for code in "ABCDF":
        assert first["harnesses"][code] == second["harnesses"][code]
    assert first["harnesses"]["E"]["report"]["integrity"]["trial_states"] == second["harnesses"]["E"]["report"]["integrity"]["trial_states"]


def test_unexpected_harness_error_is_a_failed_receipt_and_other_harnesses_still_run(tmp_path, monkeypatch):
    def broken(_):
        raise ValueError("synthetic broken component")

    monkeypatch.setattr(demo, "_walk_forward", broken)
    report = demo.run_demo(tmp_path)
    assert report["status"] == "FAIL"
    assert report["harnesses"]["B"]["error_type"] == "ValueError"
    assert all(report["harnesses"][code]["status"] == "PASS" for code in "ACDEF")
    assert _strict_json(Path(report["report_file"])) == report


def test_harness_cli_demo_emits_json_and_success_exit_code(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "demo"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert Path(output["report_file"]).is_file()
    assert Path(output["output_dir"]).parent == tmp_path / "reports" / "harness-demo"


def test_harness_cli_demo_returns_nonzero_for_failed_acceptance(tmp_path, capsys, monkeypatch):
    original = demo._formula_contract

    def failing_check(run_dir):
        result = original(run_dir)
        result["checks"]["synthetic_regression"] = False
        result["status"] = "FAIL"
        return result

    monkeypatch.setattr(demo, "_formula_contract", failing_check)
    assert main(["--root", str(tmp_path), "demo"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["harnesses"]["D"]["checks"]["synthetic_regression"] is False
