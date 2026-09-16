"""Availability-time contracts for observations, revisions and model artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


class TemporalContractError(ValueError):
    """An observation or model cannot be used at the requested historical time."""


def aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TemporalContractError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class PointInTimeRecord:
    """One immutable-identity version of an observation; timestamps are inclusive.

    ``record_id`` identifies the observation across revisions, not the instrument.
    ``version`` is an opaque identity, never a lexical ordering of revisions.
    Financial period dates belong in event_time; actual disclosure times belong
    in published_at/revision_at. All supplied times must be visible before use.
    """

    record_id: str
    version: str
    event_time: datetime
    available_at: datetime
    value: Any
    published_at: datetime | None = None
    revision_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise TemporalContractError("record_id must be a nonempty string")
        if not isinstance(self.version, str) or not self.version:
            raise TemporalContractError("version must be a nonempty string")
        for field in ("event_time", "available_at", "published_at", "revision_at", "completed_at"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, aware_utc(value, field))
        if self.completed_at is not None and self.completed_at < self.event_time:
            raise TemporalContractError("completed_at cannot precede event_time")
        if self.published_at is not None and self.revision_at is not None and self.revision_at < self.published_at:
            raise TemporalContractError("revision_at cannot precede published_at")
        object.__setattr__(self, "value", deepcopy(self.value))

    @property
    def visible_at(self) -> datetime:
        return max(value for value in (
            self.event_time, self.available_at, self.published_at, self.revision_at, self.completed_at,
        ) if value is not None)

    @property
    def revision_order(self) -> tuple[datetime, datetime]:
        # A delayed delivery of an old publication must not roll back a newer
        # disclosed revision. Availability breaks ties only within that version time.
        return (self.revision_at or self.published_at or self.event_time, self.available_at)


@dataclass(frozen=True)
class ModelTemporalContract:
    """The declared inputs and cutoff used to produce one model artifact.

    trained_through is the latest permitted training observation event time;
    training_cutoff is when training inputs/labels were frozen. Artifact
    availability must be no earlier than that cutoff. Explicit empty tuples
    describe models without that input class; they do not prove lineage completeness.
    """

    training_cutoff: datetime
    trained_through: datetime
    model_available_at: datetime
    training_records: tuple[PointInTimeRecord, ...] = ()
    training_labels: tuple[PointInTimeRecord, ...] = ()
    preprocessing_records: tuple[PointInTimeRecord, ...] = ()

    def __post_init__(self) -> None:
        for field in ("training_cutoff", "trained_through", "model_available_at"):
            object.__setattr__(self, field, aware_utc(getattr(self, field), field))
        for field in ("training_records", "training_labels", "preprocessing_records"):
            values = tuple(getattr(self, field))
            if any(not isinstance(value, PointInTimeRecord) for value in values):
                raise TemporalContractError(f"{field} must contain PointInTimeRecord values")
            object.__setattr__(self, field, values)


class PointInTimeHarness:
    """Select visible revisions and validate explicitly declared model lineage."""

    @staticmethod
    def select(records: Iterable[PointInTimeRecord], *, as_of: datetime) -> tuple[PointInTimeRecord, ...]:
        cutoff = aware_utc(as_of, "as_of")
        versions: dict[tuple[str, str], PointInTimeRecord] = {}
        selected: dict[str, PointInTimeRecord] = {}
        for record in records:
            if not isinstance(record, PointInTimeRecord):
                raise TemporalContractError("records must contain PointInTimeRecord values")
            key = (record.record_id, record.version)
            if key in versions and versions[key] != record:
                raise TemporalContractError(f"conflicting record version: {key}")
            versions[key] = record
            if record.visible_at > cutoff:
                continue
            previous = selected.get(record.record_id)
            if previous is not None and previous.revision_order == record.revision_order and previous.version != record.version:
                raise TemporalContractError(f"ambiguous simultaneous versions: {record.record_id}")
            if previous is None or record.revision_order > previous.revision_order:
                selected[record.record_id] = record
        return tuple(selected[key] for key in sorted(selected))

    @staticmethod
    def validate_model(contract: ModelTemporalContract, *, as_of: datetime) -> dict[str, Any]:
        cutoff = aware_utc(as_of, "as_of")
        blockers: list[str] = []
        if contract.trained_through > contract.training_cutoff:
            blockers.append("trained_through_after_training_cutoff")
        if contract.training_cutoff > contract.model_available_at:
            blockers.append("artifact_available_before_training_cutoff")
        if contract.model_available_at > cutoff:
            blockers.append("model_not_available_at_inference")
        for group in ("training_records", "training_labels", "preprocessing_records"):
            for record in getattr(contract, group):
                if record.visible_at > contract.training_cutoff:
                    blockers.append(f"{group}:{record.record_id}:not_available_at_training_cutoff")
                if group != "training_labels" and record.event_time > contract.trained_through:
                    blockers.append(f"{group}:{record.record_id}:after_trained_through")
        return {
            "passed": not blockers,
            "blockers": blockers,
            "as_of": cutoff.isoformat(),
            "training_cutoff": contract.training_cutoff.isoformat(),
            "trained_through": contract.trained_through.isoformat(),
            "model_available_at": contract.model_available_at.isoformat(),
            "declared_input_counts": {name: len(getattr(contract, name)) for name in (
                "training_records", "training_labels", "preprocessing_records",
            )},
            "lineage_completeness": "caller_declared_not_independently_verified",
            "research_only": True,
        }
