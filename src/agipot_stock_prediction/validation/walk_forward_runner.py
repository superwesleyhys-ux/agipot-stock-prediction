"""Fold-local fitting, mature-label purge and validation-only model selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from .delayed_label_replay import finite, frozen_features
from .point_in_time import TemporalContractError, aware_utc
from .walk_forward_factory import WalkForwardWindow


class BatchModel(Protocol):
    def fit(self, features: Sequence[Mapping[str, float]], labels: Sequence[float]) -> None: ...
    def predict(self, features: Mapping[str, float]) -> float: ...


class Preprocessor(Protocol):
    def fit(self, features: Sequence[Mapping[str, float]]) -> None: ...
    def transform(self, features: Mapping[str, float]) -> Mapping[str, float]: ...


class IdentityPreprocessor:
    def fit(self, features: Sequence[Mapping[str, float]]) -> None:
        pass

    def transform(self, features: Mapping[str, float]) -> Mapping[str, float]:
        return dict(features)


@dataclass(frozen=True)
class TemporalSample:
    """One forecast and its forward-label information interval, in explicit units.

    features must already be point-in-time observations as of predicted_at.
    The harness controls subsequent fitting; it cannot audit arbitrary feature
    code that has already consumed future observations.
    """

    sample_id: str
    predicted_at: datetime
    features: Mapping[str, float]
    label: float | None
    label_available_at: datetime | None
    label_start: datetime
    label_end: datetime
    feature_available_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise TemporalContractError("sample_id must be a nonempty string")
        for name in ("predicted_at", "label_start", "label_end", "label_available_at", "feature_available_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, aware_utc(value, name))
        if not self.predicted_at <= self.label_start <= self.label_end:
            raise TemporalContractError("forward label interval must start at/after prediction and have ordered endpoints")
        if self.label_available_at is not None and self.label_available_at < self.label_end:
            raise TemporalContractError("label_available_at cannot precede label_end")
        if self.feature_available_at is not None and self.feature_available_at > self.predicted_at:
            raise TemporalContractError("features were not available at predicted_at")
        if (self.label is None) != (self.label_available_at is None):
            raise TemporalContractError("label and label_available_at must both be present or both be absent")
        if self.label is not None:
            object.__setattr__(self, "label", finite(self.label, "label"))
        object.__setattr__(self, "features", frozen_features(self.features))


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated = [record for record in records if record["label"] is not None]
    errors = [finite((record["prediction"] - record["label"]) ** 2, "squared_error") for record in evaluated]
    return {
        "sample_size": len(errors),
        "pending_labels": sum(record["label_status"] == "PENDING_LABEL" for record in records),
        "purged_labels": sum(record["label_status"] == "PURGED_LABEL_OVERLAP" for record in records),
        "mean_squared_error": sum(errors) / len(errors) if errors else None,
    }


def _deduplicate(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        if record["sample_id"] in unique:
            duplicates += 1
        else:
            unique[record["sample_id"]] = dict(record)
    return list(unique.values()), duplicates


class WalkForwardHarness:
    """Run expanding/rolling windows and a final holdout with fresh objects.

    model_factory can be a factory or a name→factory candidate mapping. Each
    candidate's validation MSE selects the winner; fold test and final holdout
    labels are never passed to selection. The winner is refit from scratch on
    eligible pre-holdout samples. No evaluator callback can inspect holdout data.
    """

    def __init__(
        self,
        model_factory: Callable[[], BatchModel] | Mapping[str, Callable[[], BatchModel]],
        *,
        preprocessor_factory: Callable[[], Preprocessor] = IdentityPreprocessor,
        embargo: timedelta = timedelta(0),
        session_timezone: str = "America/New_York",
    ) -> None:
        self.factories = dict(model_factory) if isinstance(model_factory, Mapping) else {"default": model_factory}
        if not self.factories or any(not isinstance(key, str) or not key or not callable(value)
                                     for key, value in self.factories.items()):
            raise TemporalContractError("at least one named model factory is required")
        if not isinstance(embargo, timedelta) or embargo < timedelta(0):
            raise TemporalContractError("embargo must be a nonnegative timedelta")
        self.preprocessor_factory = preprocessor_factory
        self.embargo = embargo
        self.timezone = ZoneInfo(session_timezone)

    def _bounds(self, start: date, end: date) -> tuple[datetime, datetime]:
        if type(start) is not date or type(end) is not date or start > end:
            raise TemporalContractError("window bounds must be ordered date values")
        return (
            aware_utc(datetime.combine(start, time.min, self.timezone), "window_start"),
            aware_utc(datetime.combine(end + timedelta(days=1), time.min, self.timezone), "window_end"),
        )

    def _training(
        self, candidates: Sequence[TemporalSample], cutoff: datetime, protected_start: datetime,
    ) -> tuple[list[TemporalSample], dict[str, list[str]]]:
        selected: list[TemporalSample] = []
        dropped: dict[str, list[str]] = {"unmatured_label": [], "overlapping_label": []}
        for sample in candidates:
            if sample.label_end >= protected_start:
                dropped["overlapping_label"].append(sample.sample_id)
            elif sample.label is None or sample.label_available_at > cutoff:
                dropped["unmatured_label"].append(sample.sample_id)
            else:
                selected.append(sample)
        if not selected:
            raise TemporalContractError("no mature, non-overlapping training samples")
        return selected, dropped

    @staticmethod
    def _fit(model: BatchModel, preprocessor: Preprocessor, samples: Sequence[TemporalSample]) -> None:
        preprocessor.fit([dict(sample.features) for sample in samples])
        rows = [dict(frozen_features(preprocessor.transform(dict(sample.features)))) for sample in samples]
        model.fit(rows, [sample.label for sample in samples])

    @staticmethod
    def _predict(
        model: BatchModel, preprocessor: Preprocessor, samples: Sequence[TemporalSample],
        *, evaluation_as_of: datetime, fold: int | str, information_end_before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        records = []
        for sample in samples:
            transformed = dict(frozen_features(preprocessor.transform(dict(sample.features))))
            prediction = finite(model.predict(transformed), "prediction")
            overlap = information_end_before is not None and sample.label_end >= information_end_before
            mature = not overlap and sample.label is not None and sample.label_available_at <= evaluation_as_of
            records.append({
                "sample_id": sample.sample_id,
                "fold": fold,
                "predicted_at": sample.predicted_at.isoformat(),
                "prediction": prediction,
                "label": sample.label if mature else None,
                "label_status": "PURGED_LABEL_OVERLAP" if overlap else "EVALUATED" if mature else "PENDING_LABEL",
                "label_available_at": sample.label_available_at.isoformat() if sample.label_available_at else None,
            })
        return records

    def run(
        self,
        samples: Sequence[TemporalSample],
        windows: Sequence[WalkForwardWindow],
        *,
        final_holdout_start: datetime,
        holdout_end: datetime,
        evaluation_as_of: datetime | None = None,
    ) -> dict[str, Any]:
        holdout_start = aware_utc(final_holdout_start, "final_holdout_start")
        end = aware_utc(holdout_end, "holdout_end")
        observed_at = aware_utc(evaluation_as_of or end, "evaluation_as_of")
        if holdout_start > end or observed_at < end:
            raise TemporalContractError("holdout_start <= holdout_end <= evaluation_as_of is required")
        rows = sorted(samples, key=lambda sample: (sample.predicted_at, sample.sample_id))
        if len({sample.sample_id for sample in rows}) != len(rows):
            raise TemporalContractError("sample IDs must be unique before cross-fold aggregation")
        if not windows:
            raise TemporalContractError("at least one development window is required")
        checked_windows = []
        for window in sorted(windows, key=lambda item: (item.train_start, item.validation_start, item.test_start)):
            train = self._bounds(window.train_start, window.train_end)
            validation = self._bounds(window.validation_start, window.validation_end)
            test = self._bounds(window.test_start, window.test_end)
            if train[1] > validation[0] or validation[1] > test[0] or test[1] > holdout_start:
                raise TemporalContractError("train/validation/test must be disjoint and entirely before frozen holdout")
            checked_windows.append((train, validation, test))
        # Keep references, not just IDs, so Python cannot reuse collected IDs.
        created: list[Any] = []

        def fresh(factory):
            result = factory()
            if any(result is item for item in created):
                raise TemporalContractError("factory reused an object; each fit requires fresh model and preprocessor")
            created.append(result)
            return result

        candidates: dict[str, Any] = {}
        for name, factory in sorted(self.factories.items()):
            folds = []
            validation_records: list[dict[str, Any]] = []
            test_records: list[dict[str, Any]] = []
            for fold_id, (train, validation, test) in enumerate(checked_windows):
                protected_start = validation[0] - self.embargo
                cutoff = min(train[1] - timedelta(microseconds=1), protected_start)
                training, dropped = self._training(
                    [sample for sample in rows if train[0] <= sample.predicted_at < train[1]], cutoff, protected_start,
                )
                model, preprocessor = fresh(factory), fresh(self.preprocessor_factory)
                self._fit(model, preprocessor, training)
                validation_rows = [sample for sample in rows if validation[0] <= sample.predicted_at < validation[1]]
                test_rows = [sample for sample in rows if test[0] <= sample.predicted_at < test[1]]
                # Validation labels must mature before fold test begins; labels
                # whose information interval reaches test are purged as well.
                selection_cutoff = holdout_start - timedelta(microseconds=1)
                validation_boundary = test[0] - self.embargo
                val_predictions = self._predict(model, preprocessor, validation_rows,
                                                evaluation_as_of=min(selection_cutoff, validation_boundary - timedelta(microseconds=1)),
                                                information_end_before=validation_boundary, fold=fold_id)
                test_predictions = self._predict(model, preprocessor, test_rows,
                                                 evaluation_as_of=selection_cutoff,
                                                 information_end_before=holdout_start, fold=fold_id)
                validation_records.extend(val_predictions)
                test_records.extend(test_predictions)
                folds.append({
                    "fold": fold_id,
                    "training_cutoff": cutoff.isoformat(),
                    "training_sample_ids": [sample.sample_id for sample in training],
                    "preprocessor_fit_sample_ids": [sample.sample_id for sample in training],
                    "training_label_available_at": [sample.label_available_at.isoformat() for sample in training],
                    "purged": dropped,
                    "validation": {**_metrics(val_predictions), "predictions": val_predictions},
                    "test": {**_metrics(test_predictions), "predictions": test_predictions},
                })
            val_unique, val_duplicates = _deduplicate(validation_records)
            test_unique, test_duplicates = _deduplicate(test_records)
            metrics = _metrics(val_unique)
            if metrics["sample_size"] == 0:
                raise TemporalContractError(f"candidate {name} has no mature validation labels before holdout")
            candidates[name] = {
                "folds": folds,
                "validation": {**metrics, "duplicate_predictions_removed": val_duplicates, "predictions": val_unique},
                "test": {**_metrics(test_unique), "duplicate_predictions_removed": test_duplicates, "predictions": test_unique},
            }
        # Freeze winner before reading/evaluating final-holdout labels.
        winner = min(candidates, key=lambda key: (candidates[key]["validation"]["mean_squared_error"], key))
        final_cutoff = holdout_start - self.embargo - timedelta(microseconds=1)
        final_training, final_dropped = self._training(
            [sample for sample in rows if sample.predicted_at < final_cutoff], final_cutoff, final_cutoff,
        )
        final_model, final_preprocessor = fresh(self.factories[winner]), fresh(self.preprocessor_factory)
        self._fit(final_model, final_preprocessor, final_training)
        holdout_rows = [sample for sample in rows if holdout_start <= sample.predicted_at <= end]
        if not holdout_rows:
            raise TemporalContractError("frozen holdout contains no prediction samples")
        holdout_predictions = self._predict(final_model, final_preprocessor, holdout_rows,
                                            evaluation_as_of=observed_at, fold="final_holdout")
        return {
            "research_only": True,
            "candidates": candidates,
            "selection": {"candidate": winner, "criterion": "validation_mean_squared_error_only", "holdout_used": False},
            "final_fit": {
                "training_cutoff": final_cutoff.isoformat(),
                "training_sample_ids": [sample.sample_id for sample in final_training],
                "preprocessor_fit_sample_ids": [sample.sample_id for sample in final_training],
                "purged": final_dropped,
            },
            "final_holdout": {**_metrics(holdout_predictions), "predictions": holdout_predictions,
                              "start": holdout_start.isoformat(), "end": end.isoformat()},
            "deduplication_policy": "earliest_chronological_fold_per_sample_id_per_candidate_and_partition",
            "embargo_policy": "time_delta_before_evaluation_start_not_sample_count_gap",
            "embargo_seconds": self.embargo.total_seconds(),
            "session_timezone": str(self.timezone),
            "uncertainty_intervals": None,
            "formal_pbo_dsr": "not_computed",
        }
