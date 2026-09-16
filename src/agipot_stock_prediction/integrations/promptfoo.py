"""Promptfoo Python provider for deterministic, original-evidence regression.

This provider calls no language model or paid API. It checks supplied atomic
annotations against the same EvidenceTraceHarness used by ordinary tests.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from typing import Any


def _datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO strings with a timezone")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return result


def _parse_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value} is forbidden")


def call_api(prompt: str, options: dict | None = None, context: dict | None = None) -> dict[str, Any]:
    """Promptfoo's synchronous Python provider entry point.

    Prompt is a JSON object with claims/sources/entities/evidence arrays and
    optional summary. Evidence offsets are half-open Unicode character offsets.
    Caller-supplied support/refutation annotations are not automatic inference.
    """
    from agipot_stock_prediction.validation.evidence_trace import (
        AtomicClaim, Entity, EvidenceSpan, SourceDocument, evaluate_evidence,
    )
    try:
        payload = json.loads(prompt, parse_constant=_parse_constant)
        if not isinstance(payload, dict) or set(payload) - {"claims", "sources", "entities", "evidence", "summary"}:
            raise ValueError("prompt must be an evidence object with only documented fields")
        specs = {"claims": (AtomicClaim, {"as_of"}), "sources": (SourceDocument, {"published_at"}),
                 "entities": (Entity, set()), "evidence": (EvidenceSpan, {"valid_from", "valid_until"})}
        converted = {}
        for name, (constructor, time_fields) in specs.items():
            records = payload.get(name)
            if not isinstance(records, list):
                raise ValueError(f"{name} must be an array")
            converted[name] = []
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"{name} must contain objects")
                arguments = dict(record)
                for field in time_fields & set(arguments):
                    arguments[field] = _datetime(arguments[field])
                converted[name].append(constructor(**arguments))
        report = evaluate_evidence(**converted, summary=payload.get("summary"))
        return {"output": json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False), "cost": 0}
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": f"evidence input invalid: {exc}"}


def fixture_prompt() -> str:
    """Serialize the project's synthetic mixed-truth fixture as a provider prompt."""
    from agipot_stock_prediction.validation.evidence_trace import demo_evidence_fixture
    payload = {key: [asdict(item) for item in values] for key, values in demo_evidence_fixture().items()}
    return json.dumps(payload, sort_keys=True, default=lambda value: value.isoformat(), ensure_ascii=False)
