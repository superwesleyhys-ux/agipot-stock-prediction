# Temporal research harnesses (A / B / C)

These harnesses run locally with the Python standard library and the existing
AGIPOT model. They enforce declared timestamp contracts and execution order.
Synthetic acceptance tests prove those software properties, not predictive
profitability, calibrated uncertainty, or independently verified data provenance.
They do not change `scoring` or `distillation` cutoff semantics.

## A. Point-in-time visibility

```python
from datetime import datetime, timezone
from agipot_stock_prediction.validation.point_in_time import (
    PointInTimeHarness, PointInTimeRecord, ModelTemporalContract,
)

def utc(day):
    return datetime(2024, 1, day, tzinfo=timezone.utc)

versions = [
    PointInTimeRecord("quarter-earnings", "original", utc(1), utc(3), 10,
                      published_at=utc(3)),
    PointInTimeRecord("quarter-earnings", "revision", utc(1), utc(8), 12,
                      published_at=utc(3), revision_at=utc(7)),
]
assert PointInTimeHarness.select(versions, as_of=utc(6))[0].value == 10
assert PointInTimeHarness.select(versions, as_of=utc(8))[0].value == 12

artifact = ModelTemporalContract(
    training_cutoff=utc(5), trained_through=utc(4), model_available_at=utc(6),
    training_records=(versions[0],), preprocessing_records=(versions[0],),
)
audit = PointInTimeHarness.validate_model(artifact, as_of=utc(6))
assert audit["passed"]
```

`record_id` identifies one observation across revisions, not all observations of
an instrument. `version` is an opaque, nonempty string. Every timestamp must be
timezone-aware; internally it is converted to UTC. The first visible instant is
the maximum of `event_time`, `available_at`, `published_at`, `revision_at` and
`completed_at`, ignoring absent optional fields. Equality at the boundary is
allowed. A minute observation delivered before it is complete remains hidden.

For visible versions, the latest revision/publication time wins; vendor
availability breaks ties. A late delivery of an older publication cannot undo a
newer disclosed revision. If publication/revision timestamps are absent,
availability breaks ties between versions of the same event. Identical retries
are accepted, conflicting identities and ambiguous simultaneous versions fail.

`validate_model` checks artifact availability at inference, training cutoff
ordering, every declared training label's visibility, and preprocessing inputs'
visibility at the training cutoff. `trained_through` alone is insufficient.
Training observations and preprocessing inputs must also fall on or before
`trained_through`; forward labels may end later but must mature before training.
The returned `passed` checks only the supplied lineage. Empty input lists do not
prove that a model used no undisclosed data; external lineage completeness and
truthful provider timestamps require separate evidence.

## B. Fold-local fitting and a frozen final holdout

```python
from datetime import date, datetime, timedelta, timezone
from agipot_stock_prediction.validation.delayed_label_replay import AdaptiveEdgeAdapter
from agipot_stock_prediction.validation.walk_forward_factory import WalkForwardWindow
from agipot_stock_prediction.validation.walk_forward_runner import TemporalSample, WalkForwardHarness

def at(day, hour=0):
    return datetime(2024, 1, day, hour, tzinfo=timezone.utc)

samples = [
    TemporalSample(str(day), at(day, 12), {"ret_5m": 0.002}, 8.0,
                   at(day, 15), at(day, 12), at(day, 13))
    for day in range(1, 15)
]
window = WalkForwardWindow(date(2024, 1, 1), date(2024, 1, 4),
                           date(2024, 1, 5), date(2024, 1, 6),
                           date(2024, 1, 7), date(2024, 1, 8))
runner = WalkForwardHarness(
    {"fast": lambda: AdaptiveEdgeAdapter(min_samples=1, decay=0.95),
     "slow": lambda: AdaptiveEdgeAdapter(min_samples=1, decay=0.995)},
    session_timezone="UTC",
)
report = runner.run(samples, [window], final_holdout_start=at(12),
                    holdout_end=at(14, 23))
assert report["selection"]["holdout_used"] is False
```

`build_walk_forward_windows` can generate the supplied windows from actual
session dates. The runner additionally performs actual fitting. Each candidate
gets a **new model and preprocessor for every fold and final refit**; returning
the same object from a factory is rejected. Model protocol:
`fit(sequence_of_feature_mappings, sequence_of_labels)` and
`predict(one_feature_mapping) -> finite float`. Preprocessor protocol:
`fit(sequence_of_feature_mappings)` and
`transform(one_feature_mapping) -> finite_feature_mapping`. `transform` must
only apply its fitted state, and `predict` must not train. Factories must create
untrained objects and must not independently read evaluation data. The default
preprocessor copies the input without learning parameters.

`TemporalSample` requires a unique sample ID, prediction time, numeric features,
label information interval `[label_start, label_end]`, and actual
`label_available_at`. The interval must start at or after prediction; labels
cannot be available before their interval ends. Missing labels use `None` for
both label and availability. `feature_available_at`, if supplied, cannot exceed
prediction time. Precomputed features still require independent point-in-time
construction: no runner can infer whether an already supplied feature used the
future internally.

Date windows are inclusive dates in `session_timezone` (default New York),
converted to half-open timestamp ranges. The underlying actual bar timestamps
retain half-day/holiday/leap-day semantics; the runner does not fabricate bars
or change exchange calendars. Train, validation, and test intervals must be
disjoint **within each fold**, and all development windows must end before the
final holdout. Different rolling folds may overlap.

Training includes only labels available by the end of that fold's training
period. Any label information interval reaching the next partition is purged.
Validation labels must mature before that fold's test boundary, and validation
label intervals reaching test are purged from metrics. Test labels reaching the
final holdout are likewise excluded. Predictions are retained with an explicit
`label_status`, so excluded labels cannot silently increase the sample count.
Optional `embargo=timedelta(...)` moves the protected boundary backwards by
elapsed time and caps training availability at that boundary. This is a
one-sided separation before evaluation, **not a number-of-rows gap**; forward-only
training never uses post-test observations, so no post-test exclusion is needed.

Candidate selection uses only deduplicated validation MSE with labels mature
strictly before its fold's test boundary and the final holdout. A candidate without mature validation labels
fails. Ties are broken by candidate name. Fold test scores cannot select the
winner. The winner is frozen, then refit from scratch on mature, non-overlapping
pre-holdout samples; final-holdout values never enter this fit. Those final labels
are read only for evaluation after selection. `evaluation_as_of` defaults to
`holdout_end`; labels maturing later remain pending. The holdout must contain at
least one prediction observation.

Repeated sample predictions across folds are deduplicated **before aggregate
metrics**, retaining the earliest chronological fold per candidate and partition.
Validation and test metrics remain separate. Reports include excluded IDs,
training cutoffs, preprocessing fit IDs, label availability, sample sizes and
duplicate counts. MSE units are squared label units; `AdaptiveEdgeAdapter` uses
basis points. Fold diagnostics do not replace an untouched final evaluation;
external repeated tuning on a published holdout can still invalidate it. Formal
PBO/DSR and uncertainty intervals remain explicitly uncomputed.

## C. Delayed-label event replay

```python
from datetime import datetime, timedelta, timezone
from agipot_stock_prediction.validation.delayed_label_replay import (
    AdaptiveEdgeAdapter, DelayedLabelReplayHarness, FeatureEvent, LabelEvent,
)

t = datetime(2024, 2, 29, 14, 30, tzinfo=timezone.utc)
events = [
    FeatureEvent("sample-1", t, t, t, {"ret_5m": 0.002}),
    LabelEvent("sample-1", t + timedelta(minutes=5),
               t + timedelta(minutes=6), 8.0),
]
runner = DelayedLabelReplayHarness(AdaptiveEdgeAdapter(min_samples=1))
report = runner.replay(events)
assert report["learned_count"] == 1
assert runner.replay(events) == report  # exact retries do not train twice
```

The sequence is feature completion/availability → save one prediction → label
interval completion/availability → evaluate that **saved prediction** → learn.
`LabelEvent.event_time` is the label interval's end; `available_at` is the later
time the label can actually be consumed. It cannot precede the saved prediction.
`FeatureEvent` snapshots numeric inputs so later mutation of the caller's mapping
does not alter the pending training sample. The ledger records sample ID,
prediction time/value, label availability/value, learned time and squared error.

Bulk `replay` buffers and sorts by ready time (maximum of feature availability and
completion), then all features before labels, then sample ID. This ensures a label
at time T cannot affect other feature predictions at the same T. Streaming
`process(event, now=...)` requires that order and rejects an unseen event behind
the watermark. Early/incomplete events fail without advancing the model. Exact
retries are idempotent even behind the watermark; conflicting duplicates fail.
Orphan labels fail, and observations without labels remain pending. Restarting
with a fresh model and the same log yields the same result for deterministic
models. A failed/partially applied model update poisons that runner; it cannot
silently retry and double-learn.

The replay ledger is in memory; durable event logs/checkpoints and trusted
timestamp provenance are separate responsibilities. `AdaptiveEdgeAdapter` also
supports the batch protocol for B; it rejects batch fitting an already trained
wrapped model. A River-like model can implement the same `predict_one` /
`learn_one` protocol without adding a mandatory dependency.

## Acceptance checks

```bash
python -m pytest -q tests/test_point_in_time.py tests/test_walk_forward_runner.py tests/test_delayed_label_replay.py
```

Tests use synthetic values and cover future financial disclosures, delayed
revisions, incomplete minute observations, unavailable models/labels/preprocessing
inputs, holdout tampering, fresh fold state, maturity/purge/elapsed-time embargo,
overlapping fold deduplication, leap-day/holiday/half-day endpoints, saved original
predictions, duplicate/out-of-order/missing events, and model-update failures.
