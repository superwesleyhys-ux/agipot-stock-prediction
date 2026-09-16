from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from agipot_stock_prediction import analyze_stock
from agipot_stock_prediction.cli import main


def payload():
    start = datetime(2023, 1, 2, tzinfo=timezone.utc)
    rows = []
    for i in range(700):
        day = start + timedelta(days=i)
        if day.weekday() < 5:
            price = 100 + i/10 + math.sin(i/5)
            rows.append({"date": day.date().isoformat(), "close": price, "volume": 1_000_000})
    return {"symbol": "SYNTH", "as_of": "2025-01-03T22:00:00Z", "daily_rows": rows}


def test_combined_pipeline_is_json_serializable_and_keeps_gates():
    report = analyze_stock(payload())
    json.dumps(report, allow_nan=False)
    assert report["research_only"] is True
    assert report["distillation"]["specific_stock_index"]["complete_index"] is None
    assert set(report["scores"]["scores"]) == {"trend", "momentum", "value"}
    assert report["intraday"]["status"] == "NOT_PROVIDED"


def test_missing_history_stays_blocked():
    data = payload()
    data["daily_rows"] = []
    report = analyze_stock(data)
    assert not report["distillation"]["ok"]
    assert all(not score["eligible"] for score in report["scores"]["scores"].values())


def test_cli_uses_local_json_and_output(tmp_path):
    source, output = tmp_path/"input.json", tmp_path/"report.json"
    source.write_text(json.dumps(payload()))
    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["symbol"] == "SYNTH"


def test_cli_rejects_nonfinite_json(tmp_path):
    source = tmp_path/"input.json"
    source.write_text('{"symbol": "SYNTH", "daily_rows": [NaN]}')
    with pytest.raises(SystemExit) as error:
        main(["--input", str(source)])
    assert error.value.code == 2


def test_naive_asof_requires_an_explicit_timezone():
    data = payload()
    data["as_of"] = "2025-01-03"
    with pytest.raises(ValueError, match="timezone"):
        analyze_stock(data)
