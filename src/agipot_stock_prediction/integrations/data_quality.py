"""Strict research DataFrame contracts; time visibility is a separate harness."""

from __future__ import annotations

import math
from typing import Any

from ._optional import disable_telemetry, require

FIELDS = {
    "daily": ("symbol", "date", "open", "high", "low", "close", "volume", "available_at"),
    "minute": ("symbol", "timestamp", "open", "high", "low", "close", "volume", "available_at"),
    "tick": ("symbol", "timestamp", "seq", "price", "shares", "available_at"),
    "fundamental": ("symbol", "period_end", "metric", "value", "unit", "published_at", "available_at"),
}
KEYS = {"daily": ["symbol", "date"], "minute": ["symbol", "timestamp"],
        "tick": ["symbol", "timestamp", "seq"],
        "fundamental": ["symbol", "period_end", "metric", "available_at"]}


def _kind(kind: str) -> None:
    if kind not in FIELDS:
        raise ValueError(f"kind must be one of {tuple(FIELDS)}")


def _aware_time(value: Any) -> bool:
    pd = require("pandas")
    try:
        stamp = pd.Timestamp(value)
        return not pd.isna(stamp) and stamp.tzinfo is not None
    except (TypeError, ValueError, OverflowError):
        return False


def dataframe_schema(kind: str = "daily") -> Any:
    """Return a real Pandera schema for canonical research frames, with no I/O."""
    _kind(kind)
    pa = require("pandera.pandas", "pandera[pandas]")
    columns = {"symbol": pa.Column(str, pa.Check.str_length(min_value=1))}
    checks = [pa.Check(lambda frame: len(frame) > 0, error="empty research batch")]
    for name in FIELDS[kind][1:]:
        if name in {"date", "period_end"}:
            columns[name] = pa.Column(str, pa.Check(
                lambda value: _valid_date(value), element_wise=True, error="expected ISO calendar date"))
        elif name in {"timestamp", "published_at", "available_at"}:
            columns[name] = pa.Column(None, pa.Check(_aware_time, element_wise=True,
                                                    error="timestamp must have timezone"))
        elif name in {"metric", "unit"}:
            columns[name] = pa.Column(str, pa.Check.str_length(min_value=1))
        elif name == "seq":
            columns[name] = pa.Column(int, pa.Check.ge(0), coerce=False)
        else:
            validators = [pa.Check(lambda series: series.map(math.isfinite), error="nonfinite data")]
            if name in {"open", "high", "low", "close", "price"}:
                validators.append(pa.Check.gt(0))
            elif name in {"volume", "shares"}:
                validators.append(pa.Check.ge(0))
            columns[name] = pa.Column(float, validators, coerce=True)
    if kind in {"daily", "minute"}:
        checks.extend([
            pa.Check(lambda frame: (frame.high >= frame[["open", "close", "low"]].max(axis=1)).all(), error="high below OHLC"),
            pa.Check(lambda frame: (frame.low <= frame[["open", "close", "high"]].min(axis=1)).all(), error="low above OHLC"),
        ])
    if kind == "fundamental":
        columns["revision_at"] = pa.Column(None, pa.Check(_aware_time, element_wise=True), required=False)
    return pa.DataFrameSchema(columns, checks=checks, unique=KEYS[kind], strict=False,
                              name=f"agipot_{kind}_v1")


def _valid_date(value: str) -> bool:
    from datetime import date
    try:
        return date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def validate_frame(frame: Any, *, kind: str = "daily") -> Any:
    """Validate and return a copy; Pandera SchemaErrors mean the gate failed."""
    return dataframe_schema(kind).validate(frame.copy(), lazy=True)


def validate_with_gx(frame: Any, *, kind: str = "daily") -> dict[str, Any]:
    """Run real GX batch expectations for required fields, ranges and uniqueness.

    This optional independent batch report does not replace Pandera's OHLC/type
    checks or the project's historical visibility harness.
    """
    _kind(kind)
    disable_telemetry()
    gx = require("great_expectations", "great-expectations")
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="agipot_local_frame")
    asset = source.add_dataframe_asset(name=kind)
    definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = definition.get_batch(batch_parameters={"dataframe": frame.copy()})
    expectations = [gx.expectations.ExpectTableRowCountToBeBetween(min_value=1)]
    missing = set(FIELDS[kind]) - set(frame.columns)
    for name in FIELDS[kind]:
        expectations.append(gx.expectations.ExpectColumnToExist(column=name))
        if name not in missing:
            expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=name))
    if not missing:
        expectations.append(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=KEYS[kind]))
    for name in set(frame.columns) & {"open", "high", "low", "close", "price", "volume", "shares", "value"}:
        positive = name in {"open", "high", "low", "close", "price"}
        minimum = -float.fromhex("0x1.fffffffffffffp+1023") if name == "value" else 0
        expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(
            column=name, min_value=minimum, strict_min=positive,
            max_value=float.fromhex("0x1.fffffffffffffp+1023")))
    results = [batch.validate(expectation).to_json_dict() for expectation in expectations]
    return {"framework": "great_expectations", "success": all(result["success"] for result in results),
            "kind": kind, "results": results, "point_in_time_verified": False}
