"""Auditable original-text evidence contracts and deterministic fixture checking.

This is NOT an automatic truth, entailment, extraction or entity-resolution
model. A human or independently evaluated annotator supplies stable entity IDs,
atomic relations, time bounds and support/refutation verdicts. This harness
checks their identity, original-text anchors, temporal scope and source counts.
Generated summaries are presentation data and never participate in verdicts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _time(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    version: str
    text: str
    published_at: datetime
    origin_id: str

    def __post_init__(self) -> None:
        for name in ("source_id", "version", "text", "origin_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "published_at", _time(self.published_at, "published_at"))

    @property
    def content_hash(self) -> str:
        return _hash(self.text)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    label: str

    def __post_init__(self) -> None:
        _text(self.entity_id, "entity_id")
        _text(self.label, "label")


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    subject_id: str
    predicate: str
    object_id: str
    as_of: datetime

    def __post_init__(self) -> None:
        for name in ("claim_id", "subject_id", "predicate", "object_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "as_of", _time(self.as_of, "claim.as_of"))


@dataclass(frozen=True)
class EvidenceSpan:
    source_id: str
    source_version: str
    start: int
    end: int
    quote: str
    subject_id: str
    predicate: str
    object_id: str
    verdict: str
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    timeless: bool = False

    def __post_init__(self) -> None:
        for name in ("source_id", "source_version", "quote", "subject_id", "predicate", "object_id"):
            _text(getattr(self, name), name)
        if self.verdict not in {"supports", "refutes", "unknown"}:
            raise ValueError("verdict must be supports, refutes or unknown")
        if (isinstance(self.start, bool) or isinstance(self.end, bool)
                or not isinstance(self.start, int) or not isinstance(self.end, int)
                or self.start < 0 or self.end <= self.start):
            raise ValueError("evidence offsets must be integer half-open character positions")
        if not isinstance(self.timeless, bool):
            raise ValueError("timeless must be boolean")
        for name in ("valid_from", "valid_until"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _time(getattr(self, name), name))
        if self.valid_from is not None and self.valid_until is not None and self.valid_from >= self.valid_until:
            raise ValueError("evidence validity interval must have positive duration")
        if self.timeless and (self.valid_from is not None or self.valid_until is not None):
            raise ValueError("timeless evidence cannot also have a bounded validity interval")


def _source_groups(sources: Sequence[SourceDocument]) -> dict[tuple[str, str], str]:
    """Collapse declared syndication origins OR exactly identical full texts."""
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    for source in sources:
        left, right = find("origin:" + source.origin_id), find("text:" + source.content_hash)
        parents[max(left, right)] = min(left, right)
    return {(source.source_id, source.version): find("origin:" + source.origin_id) for source in sources}


def evaluate_evidence(
    claims: Sequence[AtomicClaim],
    sources: Sequence[SourceDocument],
    entities: Sequence[Entity],
    evidence: Sequence[EvidenceSpan],
    *,
    summary: str | None = None,
) -> dict[str, Any]:
    """Check fixed annotations and report SUPPORTED/REFUTED/UNKNOWN/CONFLICTED.

    Offsets are Python Unicode character indices ``text[start:end]``. A span
    needs the exact source version and quote, an exact stable-ID relation, a
    source published by the claim's as-of time, and a validity interval
    ``[valid_from, valid_until)`` covering that time. An annotator may explicitly
    mark a relation timeless. Unspecified temporal scope is not assumed current.

    A validated anchor proves where the annotated text lives, not whether that
    text logically entails the claim or whether the publisher itself is true.
    """
    if summary is not None and not isinstance(summary, str):
        raise ValueError("summary must be a display string or None")
    entity_map = {entity.entity_id: entity for entity in entities}
    if len(entity_map) != len(entities):
        raise ValueError("entity IDs must be unique; equal labels do not merge IDs")
    source_map = {(source.source_id, source.version): source for source in sources}
    if len(source_map) != len(sources):
        raise ValueError("source ID/version pairs must be unique")
    if len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("claim IDs must be unique")
    for relation in [*claims, *evidence]:
        if relation.subject_id not in entity_map or relation.object_id not in entity_map:
            raise ValueError("all relation endpoints require registered stable entity IDs")
    results = []
    for claim in claims:
        # A later syndication record must not retroactively merge two groups
        # that were independent in the information available at this as-of.
        groups = _source_groups([source for source in sources if source.published_at <= claim.as_of])
        accepted = []
        rejected = []
        candidate_relation = (claim.subject_id, claim.predicate, claim.object_id)
        for index, span in enumerate(evidence):
            relation = (span.subject_id, span.predicate, span.object_id)
            if relation != candidate_relation:
                continue
            source = source_map.get((span.source_id, span.source_version))
            reasons = []
            if source is None:
                reasons.append("source_version_missing")
            elif span.end > len(source.text) or source.text[span.start:span.end] != span.quote:
                reasons.append("original_text_anchor_mismatch")
            if source is not None and source.published_at > claim.as_of:
                reasons.append("source_not_yet_published")
            if not span.timeless:
                if span.valid_from is None or span.valid_until is None:
                    reasons.append("temporal_scope_unverified")
                elif not span.valid_from <= claim.as_of < span.valid_until:
                    reasons.append("evidence_outside_validity_interval")
            entry = {
                "evidence_index": index,
                "source_id": span.source_id,
                "source_version": span.source_version,
                "start": span.start,
                "end": span.end,
                "quote": span.quote,
                "annotated_verdict": span.verdict,
            }
            if reasons:
                rejected.append({**entry, "reasons": reasons})
            else:
                accepted.append({
                    **entry,
                    "source_content_hash": source.content_hash,
                    "independent_source_group": groups[(span.source_id, span.source_version)],
                    "published_at": source.published_at.isoformat(),
                    "valid_from": span.valid_from.isoformat() if span.valid_from else None,
                    "valid_until": span.valid_until.isoformat() if span.valid_until else None,
                    "timeless": span.timeless,
                })
        support = {item["independent_source_group"] for item in accepted if item["annotated_verdict"] == "supports"}
        refutation = {item["independent_source_group"] for item in accepted if item["annotated_verdict"] == "refutes"}
        verdict = "CONFLICTED" if support and refutation else "SUPPORTED" if support else "REFUTED" if refutation else "UNKNOWN"
        results.append({
            "claim_id": claim.claim_id,
            "relation": {"subject_id": claim.subject_id, "predicate": claim.predicate, "object_id": claim.object_id},
            "as_of": claim.as_of.isoformat(),
            "verdict": verdict,
            "independent_support_count": len(support),
            "independent_refutation_count": len(refutation),
            "accepted_evidence": accepted,
            "rejected_evidence": rejected,
        })
    verdicts = {item["verdict"] for item in results}
    if "SUPPORTED" in verdicts and "REFUTED" in verdicts:
        content_verdict = "MIXED_SUPPORTED_AND_REFUTED"
    elif "CONFLICTED" in verdicts:
        content_verdict = "CONFLICTED_EVIDENCE"
    elif "SUPPORTED" in verdicts:
        content_verdict = "SUPPORTED_WITH_UNKNOWN" if "UNKNOWN" in verdicts else "SUPPORTED"
    elif "REFUTED" in verdicts:
        content_verdict = "REFUTED_WITH_UNKNOWN" if "UNKNOWN" in verdicts else "REFUTED"
    else:
        content_verdict = "UNKNOWN"
    return {
        "schema": "evidence-trace/1",
        "claims": results,
        "content_verdict": content_verdict,
        "display_summary": summary,
        "judgment_basis": "original_text_anchors_and_supplied_atomic_annotations",
        "automatic_truth_inference": False,
        "limitations": [
            "Stable entity IDs, relation annotations and verdicts require independently established ground truth.",
            "Origin IDs and exact text hashes deduplicate known syndication; unknown paraphrases may remain correlated.",
            "No evidence means UNKNOWN; keyword co-occurrence and display summaries do not establish a relation.",
        ],
    }


def demo_evidence_fixture() -> dict[str, Any]:
    """Synthetic fixed truth: true background, explicitly refuted causal claim."""
    published = datetime(2025, 1, 2, tzinfo=timezone.utc)
    as_of = datetime(2025, 1, 3, tzinfo=timezone.utc)
    until = datetime(2025, 2, 1, tzinfo=timezone.utc)
    text = "Aurora Solar opened a laboratory. The laboratory did not cause the robotics company's earnings decline."
    first = "Aurora Solar opened a laboratory."
    second = "The laboratory did not cause the robotics company's earnings decline."
    sources = [SourceDocument("synthetic-wire", "v1", text, published, "synthetic-origin")]
    entities = [
        Entity("company:aurora-solar", "Aurora"),
        Entity("company:aurora-robotics", "Aurora"),
        Entity("asset:laboratory", "laboratory"),
        Entity("event:earnings-decline", "earnings decline"),
    ]
    return {
        "sources": sources,
        "entities": entities,
        "claims": [
            AtomicClaim("background", "company:aurora-solar", "opened", "asset:laboratory", as_of),
            AtomicClaim("false-cause", "asset:laboratory", "caused", "event:earnings-decline", as_of),
            AtomicClaim("same-name-error", "company:aurora-robotics", "opened", "asset:laboratory", as_of),
        ],
        "evidence": [
            EvidenceSpan("synthetic-wire", "v1", 0, len(first), first, "company:aurora-solar", "opened", "asset:laboratory", "supports", published, until),
            EvidenceSpan("synthetic-wire", "v1", text.index(second), len(text), second, "asset:laboratory", "caused", "event:earnings-decline", "refutes", published, until),
        ],
    }


__all__ = ["SourceDocument", "Entity", "AtomicClaim", "EvidenceSpan", "evaluate_evidence", "demo_evidence_fixture"]
