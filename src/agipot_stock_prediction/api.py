"""One offline entry point for multi-horizon research and factor scores."""
from datetime import datetime, timezone
from typing import Any, Mapping

from .distillation import distill_specific_stock_from_inputs
from .intraday import AdaptiveEdgeModel, adaptive_edge_alpha, family_alpha
from .scoring import score_stock


def analyze_stock(payload: Mapping[str, Any], *, model: AdaptiveEdgeModel | None = None) -> dict[str, Any]:
    """Analyze caller-supplied data without network requests or order execution.

    A trained online model may be supplied explicitly. This function never
    fabricates labels or trains on returns that have not yet been observed.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("input must be a JSON object")
    for name in ("symbol", "as_of", "daily_rows"):
        if name not in payload:
            raise ValueError(f"missing required field: {name}")
    stamp = payload["as_of"]
    as_of = stamp if isinstance(stamp, datetime) else datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include a timezone")
    daily = payload["daily_rows"]
    minutes = payload.get("minute_rows", [])
    ticks = payload.get("tick_payload", {})
    if not isinstance(daily, list) or not isinstance(minutes, list):
        raise ValueError("daily_rows and minute_rows must be arrays")
    if not isinstance(ticks, dict):
        raise ValueError("tick_payload must be an object of ts/price/shares arrays")
    for name in ("fundamentals", "factor_fundamentals", "intraday_features"):
        if payload.get(name) is not None and not isinstance(payload[name], dict):
            raise ValueError(f"{name} must be an object")
    source = str(payload.get("data_source", "caller_supplied"))
    symbol = str(payload["symbol"])
    distillation = distill_specific_stock_from_inputs(
        symbol, daily_rows=daily, minute_rows=minutes, tick_payload=ticks,
        fundamentals=payload.get("fundamentals"), now=as_of,
        daily_source=source, minute_source=source, tick_source=source,
    )
    scores = score_stock(daily, symbol=distillation["symbol"], as_of=as_of,
                         fundamentals=payload.get("factor_fundamentals"))
    intraday: dict[str, Any] = {"status": "NOT_PROVIDED"}
    features = payload.get("intraday_features")
    if features is not None:
        intraday = {
            "status": "RESEARCH_ONLY",
            "rule_alpha": family_alpha("intraday_momentum", features),
            "online_model_status": "NOT_TRAINED" if model is None else (
                "WARMUP" if model.samples < model.min_samples else "AVAILABLE"
            ),
            "online_alpha": adaptive_edge_alpha(model, features) if model is not None else None,
            "feature_timing": "caller_must_supply_only_completed_observations",
        }
    return {
        "schema_version": "agipot-stock-prediction/1",
        "symbol": distillation["symbol"],
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
        "data_source": source,
        "research_only": True,
        "distillation": distillation,
        "scores": scores,
        "intraday": intraday,
    }
