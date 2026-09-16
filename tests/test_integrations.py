"""Optional SDK adapters: real calls on fabricated data, no external services."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import threading

import pytest

from agipot_stock_prediction.integrations import IntegrationUnavailable
from agipot_stock_prediction.integrations import _optional
from agipot_stock_prediction.integrations.data_quality import validate_frame, validate_with_gx
from agipot_stock_prediction.integrations.drift import evidently_report
from agipot_stock_prediction.integrations.optional import openapi_schema, smoke_browser, smoke_container, smoke_openapi
from agipot_stock_prediction.integrations.research import (
    bootstrap_mean_interval, evaluate_naive_forecast, import_qlib_predictions,
    qlib_risk_analysis, time_series_splits, tune_train_validation,
)
from agipot_stock_prediction.integrations.river import make_adaptive_edge_regressor
from agipot_stock_prediction.integrations.tracking import record_mlflow_run


@pytest.fixture(autouse=True)
def no_sdk_telemetry(monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("GX_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")


def dependency(module):
    return pytest.importorskip(module, reason=f"optional integration dependency {module} is not installed")


def test_missing_optional_dependency_has_explicit_error(monkeypatch):
    def unavailable(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(_optional.importlib, "import_module", unavailable)
    with pytest.raises(IntegrationUnavailable, match="Install.*scikit-learn"):
        time_series_splits(20)


def synthetic_frame(kind):
    pd = dependency("pandas")
    base = {"symbol": "SYNTH", "available_at": "2024-01-02T22:00:00Z"}
    if kind in {"daily", "minute"}:
        base.update(open=100, high=102, low=99, close=101, volume=200)
        base.update({"date": "2024-01-02"} if kind == "daily" else {"timestamp": "2024-01-02T21:00:00Z"})
    elif kind == "tick":
        base.update(timestamp="2024-01-02T21:00:00Z", seq=1, price=101, shares=20)
    else:
        base.update(period_end="2023-12-31", metric="revenue", value=1000, unit="USD",
                    published_at="2024-01-02T21:00:00Z")
    return pd.DataFrame([base])


@pytest.mark.parametrize("kind", ["daily", "minute", "tick", "fundamental"])
def test_pandera_validates_all_four_canonical_frame_types(kind):
    dependency("pandera.pandas")
    frame = synthetic_frame(kind)
    assert len(validate_frame(frame, kind=kind)) == 1
    with pytest.raises(Exception):
        validate_frame(frame.drop(columns=["available_at"]), kind=kind)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1])
def test_pandera_rejects_bad_prices(bad):
    dependency("pandera.pandas")
    frame = synthetic_frame("daily")
    frame["close"] = bad
    with pytest.raises(Exception):
        validate_frame(frame)


def test_pandera_rejects_duplicate_keys_and_naive_time():
    dependency("pandera.pandas")
    pd = dependency("pandas")
    frame = synthetic_frame("daily")
    with pytest.raises(Exception):
        validate_frame(pd.concat([frame, frame], ignore_index=True))
    frame["available_at"] = "2024-01-02T22:00:00"
    with pytest.raises(Exception):
        validate_frame(frame)


def test_gx_runs_real_batch_expectations():
    dependency("great_expectations")
    frame = synthetic_frame("daily")
    assert validate_with_gx(frame)["success"] is True
    frame["close"] = -1
    assert validate_with_gx(frame)["success"] is False


def test_sklearn_split_respects_explicit_gap():
    dependency("sklearn")
    folds = time_series_splits(30, n_splits=3, gap=2, test_size=5)
    assert len(folds) == 3
    for fold in folds:
        assert fold["validation"][0] - fold["train"][-1] == 3
        assert len(fold["validation"]) == 5


def test_sktime_runs_actual_refit_naive_baseline():
    dependency("sktime")
    result = evaluate_naive_forecast(list(range(12)), initial_window=6)
    assert not result.empty
    assert result["test_MeanAbsoluteError"].tolist() == pytest.approx([1] * 6)


def test_optuna_never_copies_or_passes_holdout_to_objective():
    dependency("optuna")
    class PoisonHoldout:
        def __deepcopy__(self, memo):
            raise AssertionError("holdout must remain inaccessible")
    observations = []
    def objective(trial, train, validation):
        observations.append((list(train), list(validation)))
        coefficient = trial.suggest_float("coefficient", 0, 1)
        train[0] = "mutation"
        return coefficient
    report = tune_train_validation([1, 2, 3, 4, PoisonHoldout()], train_end=2,
                                   validation_end=4, objective=objective, n_trials=3, seed=5)
    assert observations == [([1, 2], [3, 4])] * 3
    assert report["final_holdout_evaluated"] is False
    assert len(report["trials"]) == 3
    assert report["status"] == "completed"


def test_optuna_keeps_failed_trials():
    dependency("optuna")
    def invalid(trial, train, validation):
        raise ValueError("synthetic failure")
    report = tune_train_validation([1, 2, 3], train_end=1, validation_end=2,
                                   objective=invalid, n_trials=2)
    assert report["status"] == "all_trials_failed"
    assert [row["state"] for row in report["trials"]] == ["FAIL", "FAIL"]


def test_stationary_bootstrap_reproducible_interval():
    dependency("arch")
    returns = [0.01, -0.02, 0.03, 0.015, -0.005] * 5
    first = bootstrap_mean_interval(returns, block_size=3, reps=100, seed=4)
    assert first == bootstrap_mean_interval(returns, block_size=3, reps=100, seed=4)
    assert first["lower"] <= first["mean"] <= first["upper"]
    assert first["formal_dsr_computed"] is False


def test_qlib_export_import_preserves_identity_without_pickle(tmp_path):
    pd = dependency("pandas")
    frame = pd.DataFrame({"datetime": ["2024-01-02"], "instrument": ["SYNTH"], "score": [0.1]})
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    result = import_qlib_predictions(path)
    assert result.index.names == ["datetime", "instrument"]
    assert result["score"].tolist() == [0.1]
    with pytest.raises(ValueError, match="local CSV"):
        import_qlib_predictions("https://example.invalid/predictions.csv")


def test_qlib_actual_risk_evaluator():
    dependency("qlib.contrib.evaluate")
    report = qlib_risk_analysis([0.01, -0.02, 0.03, 0.0], annualization=252)
    assert report["framework"] == "qlib"
    assert report["return_unit"] == "decimal"
    assert report["metrics"]["mean"] == pytest.approx(0.005)
    assert report["metrics"]["annualized_return"] == pytest.approx(1.26)


def test_river_adapter_uses_real_regressor_and_separate_update():
    base = dependency("river.base")
    model = make_adaptive_edge_regressor(min_samples=1, edge_floor_bps=0)
    assert isinstance(model, base.Regressor)
    features = {"ret_5m": 0.002}
    assert model.predict_one(features) == 0
    assert model.model.samples == 0
    assert model.learn_one(features, 5) is model
    assert model.model.samples == 1
    assert model.predict_one(features) > 0
    fresh = model.clone()
    assert fresh.model.samples == 0
    with pytest.raises(ValueError):
        model.learn_one(features, math.nan)


def test_mlflow_logs_actual_local_run_and_manifest(tmp_path):
    dependency("mlflow.tracking")
    dependency("sqlalchemy")
    report = record_mlflow_run({"trial_id": "synthetic", "code_commit": "fixture"},
                              {"mae": 0.01}, output_dir=tmp_path, status="FAILED")
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=report["tracking_uri"])
    run = client.get_run(report["run_id"])
    assert run.info.status == "FAILED"
    assert run.data.metrics["mae"] == 0.01
    assert [item.path for item in client.list_artifacts(report["run_id"])] == ["experiment_evidence.json"]
    assert report["tracking_uri"].startswith("sqlite:///")


def test_tracking_does_not_accept_remote_uri():
    with pytest.raises(ValueError, match="local path"):
        record_mlflow_run({}, {}, output_dir="https://example.invalid")


def test_evidently_computes_real_drift_report(tmp_path):
    dependency("evidently")
    pd = dependency("pandas")
    reference = pd.DataFrame({"feature": [i / 10 for i in range(100)]})
    current = pd.DataFrame({"feature": [i / 10 + 100 for i in range(100)]})
    output = tmp_path / "drift.html"
    result = evidently_report(reference, current, output_html=output)
    assert output.stat().st_size > 100
    assert result["report"]["metrics"]
    assert result["prediction_accuracy_verified"] is False


OPENAPI = {"openapi": "3.0.3", "info": {"title": "Synthetic", "version": "1"},
           "paths": {"/health": {"get": {"responses": {"200": {"description": "ok",
           "content": {"application/json": {"schema": {"type": "object", "required": ["ok"],
           "properties": {"ok": {"type": "boolean"}}}}}}}}}}}


def test_schemathesis_loads_real_schema_without_service():
    dependency("schemathesis")
    schema = openapi_schema(OPENAPI)
    assert schema["/health"]["GET"].method == "get"


def test_openapi_external_references_cannot_trigger_implicit_network():
    with pytest.raises(ValueError, match="document-local"):
        openapi_schema({"$ref": "https://example.invalid/schema.json"})


def test_http_case_cannot_override_read_only_method():
    with pytest.raises(ValueError, match="override"):
        smoke_openapi(OPENAPI, base_url="http://example.invalid", path="/health", allow_network=True,
                      case_kwargs={"method": "POST"})


def test_schemathesis_calls_and_validates_explicit_local_fixture():
    dependency("schemathesis")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = smoke_openapi(OPENAPI, base_url=f"http://127.0.0.1:{server.server_port}",
                               path="/health", allow_network=True)
        assert result["status_code"] == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("operation", [
    lambda: smoke_openapi(OPENAPI, base_url="http://example.invalid", path="/health"),
    lambda: smoke_browser(url="https://example.invalid"),
    lambda: smoke_container("example:latest", command=["true"]),
])
def test_conditional_integrations_require_explicit_execution(operation):
    with pytest.raises(ValueError, match="requires|requires|requires"):
        operation()


def test_playwright_existing_browser_local_report(tmp_path):
    dependency("playwright.sync_api")
    report = tmp_path / "report.html"
    report.write_text("<!doctype html><title>Synthetic</title><p>Offline evidence fixture</p>")
    try:
        result = smoke_browser(html_path=report, expected_text="Offline evidence fixture")
    except IntegrationUnavailable as exc:
        pytest.skip(str(exc))
    assert result["title"] == "Synthetic"
    assert result["target_kind"] == "local_report"


def test_promptfoo_provider_calls_shared_evidence_harness():
    from agipot_stock_prediction.integrations.promptfoo import call_api, fixture_prompt
    response = call_api(fixture_prompt(), {}, {})
    report = json.loads(response["output"])
    assert report["content_verdict"] == "MIXED_SUPPORTED_AND_REFUTED"
    assert report["automatic_truth_inference"] is False
    assert response["cost"] == 0
    payload = json.loads(fixture_prompt())
    payload["summary"] = "Everything is fine."
    edited = json.loads(call_api(json.dumps(payload))["output"])
    assert edited["claims"] == report["claims"]
    assert edited["content_verdict"] == report["content_verdict"]


def test_promptfoo_invalid_input_is_error_not_fake_output():
    from agipot_stock_prediction.integrations.promptfoo import call_api
    response = call_api('{"claims": NaN}')
    assert "error" in response
    assert "output" not in response


def test_promptfoo_missing_evidence_remains_unknown_despite_confident_summary():
    from agipot_stock_prediction.integrations.promptfoo import call_api, fixture_prompt
    payload = json.loads(fixture_prompt())
    payload["evidence"] = []
    before = json.loads(call_api(json.dumps(payload))["output"])
    payload["summary"] = "Every relation is definitely supported."
    after = json.loads(call_api(json.dumps(payload))["output"])
    assert before["content_verdict"] == after["content_verdict"] == "UNKNOWN"
    assert [claim["verdict"] for claim in after["claims"]] == ["UNKNOWN"] * 3
    assert before["claims"] == after["claims"]
