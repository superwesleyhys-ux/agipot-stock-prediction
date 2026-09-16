import pytest
pytest.importorskip("pytest_regressions")
from agipot_stock_prediction import analyze_stock


def test_empty_public_report_contract(data_regression):
    report = analyze_stock({"symbol": "SYNTH", "as_of": "2025-12-01T21:00:00Z", "daily_rows": []})
    data_regression.check({
        "schema_version": report["schema_version"], "symbol": report["symbol"],
        "research_only": report["research_only"], "intraday": report["intraday"],
        "eligible": {key: value["eligible"] for key, value in report["scores"]["scores"].items()},
        "complete_index": report["distillation"]["specific_stock_index"]["complete_index"],
    })
