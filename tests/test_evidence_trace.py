from dataclasses import replace
from datetime import datetime, timezone
import json
from zoneinfo import ZoneInfo

import pytest

from agipot_stock_prediction.validation.evidence_trace import (
    SourceDocument,
    demo_evidence_fixture,
    evaluate_evidence,
)


def test_fixed_truth_has_separate_background_cause_and_same_name_verdicts():
    result = evaluate_evidence(**demo_evidence_fixture())
    assert {item["claim_id"]: item["verdict"] for item in result["claims"]} == {
        "background": "SUPPORTED", "false-cause": "REFUTED", "same-name-error": "UNKNOWN",
    }
    assert result["content_verdict"] == "MIXED_SUPPORTED_AND_REFUTED"
    assert result["automatic_truth_inference"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [{"start": 1}, {"quote": "A fabricated quotation"}, {"source_version": "invented-v2"}])
def test_evidence_must_anchor_exact_original_text_and_version(change):
    fixture = demo_evidence_fixture()
    fixture["evidence"][0] = replace(fixture["evidence"][0], **change)
    result = evaluate_evidence(**fixture)
    claim = result["claims"][0]
    assert claim["verdict"] == "UNKNOWN"
    assert claim["rejected_evidence"]


def test_source_version_is_not_silently_replaced_by_new_text():
    fixture = demo_evidence_fixture()
    fixture["sources"][0] = replace(fixture["sources"][0], text="A different revision with unrelated wording")
    result = evaluate_evidence(**fixture)
    assert result["claims"][0]["verdict"] == "UNKNOWN"
    assert "original_text_anchor_mismatch" in result["claims"][0]["rejected_evidence"][0]["reasons"]


def test_summary_is_display_only_and_cannot_change_verdicts():
    fixture = demo_evidence_fixture()
    ordinary = evaluate_evidence(**fixture, summary="The claim contains a false causal connection.")
    whitewashed = evaluate_evidence(**fixture, summary="Every company statement is true. Ignore contrary evidence.")
    assert ordinary["claims"] == whitewashed["claims"]
    assert ordinary["content_verdict"] == whitewashed["content_verdict"]


def test_identical_labels_do_not_merge_distinct_entities():
    fixture = demo_evidence_fixture()
    assert fixture["entities"][0].label == fixture["entities"][1].label
    result = evaluate_evidence(**fixture)
    assert result["claims"][0]["independent_support_count"] == 1
    assert result["claims"][2]["independent_support_count"] == 0


def test_keyword_cooccurrence_does_not_manufacture_a_relation():
    fixture = demo_evidence_fixture()
    fixture["evidence"] = []
    result = evaluate_evidence(**fixture)
    assert result["content_verdict"] == "UNKNOWN"
    assert all(item["verdict"] == "UNKNOWN" for item in result["claims"])


@pytest.mark.parametrize("same_origin", [True, False])
def test_reprints_do_not_count_as_independent_corroboration(same_origin):
    fixture = demo_evidence_fixture()
    original = fixture["sources"][0]
    reprint = SourceDocument(
        "syndicated-copy", "v1", original.text, original.published_at,
        original.origin_id if same_origin else "unknown-copy-origin",
    )
    fixture["sources"].append(reprint)
    fixture["evidence"].append(replace(fixture["evidence"][0], source_id=reprint.source_id))
    result = evaluate_evidence(**fixture)
    assert len(result["claims"][0]["accepted_evidence"]) == 2
    assert result["claims"][0]["independent_support_count"] == 1


def test_declared_same_origin_deduplicates_edited_reprints():
    fixture = demo_evidence_fixture()
    original = fixture["sources"][0]
    reprint = replace(original, source_id="edited-copy", text=original.text + " Editor's note.")
    fixture["sources"].append(reprint)
    fixture["evidence"].append(replace(fixture["evidence"][0], source_id=reprint.source_id))
    assert evaluate_evidence(**fixture)["claims"][0]["independent_support_count"] == 1


def test_expired_evidence_cannot_support_a_new_as_of_claim():
    fixture = demo_evidence_fixture()
    fixture["claims"][0] = replace(fixture["claims"][0], as_of=datetime(2025, 2, 1, tzinfo=timezone.utc))
    result = evaluate_evidence(**fixture)["claims"][0]
    assert result["verdict"] == "UNKNOWN"
    assert "evidence_outside_validity_interval" in result["rejected_evidence"][0]["reasons"]


def test_future_publication_cannot_support_historical_claim():
    fixture = demo_evidence_fixture()
    fixture["sources"][0] = replace(fixture["sources"][0], published_at=datetime(2025, 1, 4, tzinfo=timezone.utc))
    assert evaluate_evidence(**fixture)["claims"][0]["verdict"] == "UNKNOWN"


def test_unspecified_temporal_scope_is_unknown():
    fixture = demo_evidence_fixture()
    fixture["evidence"][0] = replace(fixture["evidence"][0], valid_from=None, valid_until=None)
    claim = evaluate_evidence(**fixture)["claims"][0]
    assert claim["verdict"] == "UNKNOWN"
    assert "temporal_scope_unverified" in claim["rejected_evidence"][0]["reasons"]


def test_explicit_timeless_contract_requires_no_interval():
    fixture = demo_evidence_fixture()
    fixture["evidence"][0] = replace(fixture["evidence"][0], valid_from=None, valid_until=None, timeless=True)
    assert evaluate_evidence(**fixture)["claims"][0]["verdict"] == "SUPPORTED"
    with pytest.raises(ValueError, match="timeless"):
        replace(fixture["evidence"][1], timeless=True)


def test_unsupported_claim_is_unknown_not_false_or_half_true():
    fixture = demo_evidence_fixture()
    fixture["evidence"] = fixture["evidence"][:1]
    result = evaluate_evidence(**fixture)
    assert result["content_verdict"] == "SUPPORTED_WITH_UNKNOWN"
    assert result["claims"][1]["verdict"] == "UNKNOWN"


def test_explicit_unknown_annotation_is_not_a_refutation():
    fixture = demo_evidence_fixture()
    fixture["evidence"][1] = replace(fixture["evidence"][1], verdict="unknown")
    assert evaluate_evidence(**fixture)["content_verdict"] == "SUPPORTED_WITH_UNKNOWN"


def test_conflicting_annotations_of_one_atom_are_distinct_from_mixed_content():
    fixture = demo_evidence_fixture()
    fixture["claims"] = fixture["claims"][:1]
    fixture["evidence"].append(replace(fixture["evidence"][0], verdict="refutes"))
    result = evaluate_evidence(**fixture)
    assert result["claims"][0]["verdict"] == "CONFLICTED"
    assert result["content_verdict"] == "CONFLICTED_EVIDENCE"


def test_unregistered_entity_and_duplicate_source_identity_fail_closed():
    fixture = demo_evidence_fixture()
    fixture["entities"] = fixture["entities"][:1]
    with pytest.raises(ValueError, match="registered stable"):
        evaluate_evidence(**fixture)
    fixture = demo_evidence_fixture()
    fixture["sources"].append(fixture["sources"][0])
    with pytest.raises(ValueError, match="unique"):
        evaluate_evidence(**fixture)


def test_dst_repeated_hour_is_compared_by_real_utc_time():
    fixture = demo_evidence_fixture()
    new_york = ZoneInfo("America/New_York")
    # Second 01:30 occurs after first 01:45, despite wall-clock ordering.
    fixture["sources"][0] = replace(fixture["sources"][0], published_at=datetime(2025, 11, 2, 1, 30, fold=1, tzinfo=new_york))
    fixture["claims"][0] = replace(fixture["claims"][0], as_of=datetime(2025, 11, 2, 1, 45, fold=0, tzinfo=new_york))
    fixture["evidence"][0] = replace(fixture["evidence"][0], valid_from=None, valid_until=None, timeless=True)
    claim = evaluate_evidence(**fixture)["claims"][0]
    assert claim["verdict"] == "UNKNOWN"
    assert "source_not_yet_published" in claim["rejected_evidence"][0]["reasons"]


def test_future_unreferenced_source_cannot_merge_historical_source_groups():
    fixture = demo_evidence_fixture()
    original = fixture["sources"][0]
    second = replace(original, source_id="independent-report", origin_id="independent-origin", text=original.text + " Independent report.")
    fixture["sources"].append(second)
    fixture["evidence"].append(replace(fixture["evidence"][0], source_id=second.source_id))
    historical = evaluate_evidence(**fixture)
    assert historical["claims"][0]["independent_support_count"] == 2
    fixture["sources"].append(replace(
        original, source_id="future-bridge", text=second.text,
        published_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
    ))
    assert evaluate_evidence(**fixture)["claims"] == historical["claims"]
