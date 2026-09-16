"""Predict once, evaluate saved predictions at label maturity, then learn."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .point_in_time import TemporalContractError, aware_utc


def finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TemporalContractError(f"{name} must be a finite number")
    return float(value)


def frozen_features(features: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(features, Mapping) or any(not isinstance(key, str) for key in features):
        raise TemporalContractError("features must map string keys to finite numbers")
    return MappingProxyType({key: finite(value, f"features.{key}") for key, value in sorted(features.items())})


class OnlineModel(Protocol):
    def predict_one(self, features: Mapping[str, float]) -> float: ...
    def learn_one(self, features: Mapping[str, float], label: float) -> None: ...


class AdaptiveEdgeAdapter:
    """Online and batch protocol adapter; labels and predictions are in basis points."""

    def __init__(self, model=None, **model_options: Any) -> None:
        from agipot_stock_prediction.intraday import AdaptiveEdgeModel

        if model is not None and model_options:
            raise ValueError("pass a model or constructor options, not both")
        self.model = model if model is not None else AdaptiveEdgeModel(**model_options)

    def predict_one(self, features: Mapping[str, float]) -> float:
        return self.model.edge_bps(features)

    def learn_one(self, features: Mapping[str, float], label: float) -> None:
        self.model.learn(features, realized_return_bps=finite(label, "label"))

    def fit(self, features: Sequence[Mapping[str, float]], labels: Sequence[float]) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must have equal lengths")
        if self.model.samples:
            raise ValueError("batch fit requires a fresh AdaptiveEdgeModel")
        for row, label in zip(features, labels):
            self.learn_one(row, label)

    def predict(self, features: Mapping[str, float]) -> float:
        return self.predict_one(features)


@dataclass(frozen=True)
class FeatureEvent:
    sample_id: str
    event_time: datetime
    available_at: datetime
    completed_at: datetime
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise TemporalContractError("sample_id must be a nonempty string")
        for name in ("event_time", "available_at", "completed_at"):
            object.__setattr__(self, name, aware_utc(getattr(self, name), name))
        if self.completed_at < self.event_time or self.available_at < self.event_time:
            raise TemporalContractError("completion and availability cannot precede event_time")
        object.__setattr__(self, "features", frozen_features(self.features))

    @property
    def ready_at(self) -> datetime:
        return max(self.available_at, self.completed_at)


@dataclass(frozen=True)
class LabelEvent:
    sample_id: str
    event_time: datetime
    available_at: datetime
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise TemporalContractError("sample_id must be a nonempty string")
        for name in ("event_time", "available_at"):
            object.__setattr__(self, name, aware_utc(getattr(self, name), name))
        if self.available_at < self.event_time:
            raise TemporalContractError("label availability cannot precede label interval end")
        object.__setattr__(self, "value", finite(self.value, "label"))

    @property
    def ready_at(self) -> datetime:
        return self.available_at


class DelayedLabelReplayHarness:
    """An in-memory replay ledger with strict streaming order and idempotent IDs.

    Bulk replay sorts by ready time, then features before labels, then sample ID.
    Streaming process rejects unseen events older than its watermark. Identical
    retries are no-ops; conflicting retries and labels without predictions fail.
    Missing labels remain pending. No historical prediction is recomputed.
    """

    def __init__(self, model: OnlineModel) -> None:
        self.model = model
        self._watermark: datetime | None = None
        self._order_watermark: tuple[datetime, int, str] | None = None
        self._features: dict[str, FeatureEvent] = {}
        self._labels: dict[str, LabelEvent] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._failed = False
        self._failure_stage: str | None = None

    def process(self, event: FeatureEvent | LabelEvent, *, now: datetime | None = None) -> None:
        if self._failed:
            raise TemporalContractError("replay failed during a model operation; restart with a fresh model")
        if not isinstance(event, (FeatureEvent, LabelEvent)):
            raise TemporalContractError("expected FeatureEvent or LabelEvent")
        seen = self._features if isinstance(event, FeatureEvent) else self._labels
        previous = seen.get(event.sample_id)
        if previous is not None:
            if previous != event:
                raise TemporalContractError(f"conflicting duplicate {type(event).__name__}: {event.sample_id}")
            return
        at = aware_utc(now if now is not None else event.available_at, "now")
        if at < event.ready_at:
            raise TemporalContractError("event is not completed/available at now")
        order = (at, 0 if isinstance(event, FeatureEvent) else 1, event.sample_id)
        if self._order_watermark is not None and order < self._order_watermark:
            raise TemporalContractError("out-of-order streaming event; use sorted replay or restart")
        if isinstance(event, FeatureEvent):
            try:
                prediction = finite(self.model.predict_one(dict(event.features)), "prediction")
            except BaseException:
                # Stateful predictors can mutate caches/normalizers before an
                # error or interrupt. Their partially changed state is unsafe
                # to retry or reuse for later historical predictions.
                self._failed = True
                self._failure_stage = "prediction"
                raise
            self._features[event.sample_id] = event
            self._records[event.sample_id] = {
                "sample_id": event.sample_id,
                "event_time": event.event_time.isoformat(),
                "feature_available_at": event.available_at.isoformat(),
                "completed_at": event.completed_at.isoformat(),
                "predicted_at": at.isoformat(),
                "prediction": prediction,
                "label_available_at": None,
                "label": None,
                "learned_at": None,
                "squared_error": None,
                "status": "PENDING_LABEL",
            }
        else:
            if event.sample_id not in self._features:
                raise TemporalContractError(f"label has no saved prediction: {event.sample_id}")
            record = self._records[event.sample_id]
            predicted_at = datetime.fromisoformat(record["predicted_at"])
            if event.event_time < predicted_at:
                raise TemporalContractError("label interval ended before the saved prediction")
            # Calculate the error from the original prediction BEFORE updating.
            squared_error = finite((record["prediction"] - event.value) ** 2, "squared_error")
            record.update(label_available_at=event.available_at.isoformat(), label=event.value,
                          squared_error=squared_error, status="EVALUATED")
            try:
                self.model.learn_one(dict(self._features[event.sample_id].features), event.value)
            except BaseException:
                self._failed = True
                self._failure_stage = "learning"
                record["status"] = "LEARN_FAILED_RESTART_REQUIRED"
                raise
            record.update(learned_at=at.isoformat(), status="LEARNED")
            self._labels[event.sample_id] = event
        self._watermark = at
        self._order_watermark = order

    def replay(self, events: Iterable[FeatureEvent | LabelEvent]) -> dict[str, Any]:
        ordered = tuple(events)
        if any(not isinstance(event, (FeatureEvent, LabelEvent)) for event in ordered):
            raise TemporalContractError("expected FeatureEvent or LabelEvent")
        for event in sorted(ordered, key=lambda event: (
            event.ready_at, 0 if isinstance(event, FeatureEvent) else 1, event.sample_id,
        )):
            self.process(event, now=event.ready_at)
        return self.report()

    def report(self) -> dict[str, Any]:
        records = [dict(self._records[key]) for key in sorted(self._records)]
        errors = [record["squared_error"] for record in records if record["status"] == "LEARNED"]
        return {
            "research_only": True,
            "records": records,
            "predicted_count": len(records),
            "learned_count": len(errors),
            "pending_count": sum(record["status"] == "PENDING_LABEL" for record in records),
            "mean_squared_error": sum(errors) / len(errors) if errors else None,
            "watermark": self._watermark.isoformat() if self._watermark is not None else None,
            "ordering_policy": "ready_time_then_features_before_labels_then_sample_id",
            "duplicate_policy": "identical_idempotent_conflicting_rejected",
            "persistence": "in_memory_rebuild_from_event_log_with_fresh_model",
            "restart_required": self._failed,
            "failure_stage": self._failure_stage,
        }
