"""Local preregistered experiments with complete identities and a hash-chained log.

The log establishes accounting against a frozen plan, not statistical validity
or protection against an administrator rewriting every local file. An external
checkpoint of ``plan_fingerprint`` and ``head_hash`` is needed for that threat.
Workers receive only their registered specification; callers remain responsible
for declaring every actual input and seeding their model from ``spec['seed']``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import tempfile
from typing import Any


class ExperimentIntegrityError(ValueError):
    """The registered plan, event chain, or complete trial accounting is invalid."""


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetimes must include a timezone")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical mapping keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise ValueError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Stable UTF-8 JSON; ordering of records remains significant."""
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_dependency_versions(names: Iterable[str] = ()) -> dict[str, str]:
    """Capture the interpreter plus explicitly selected installed distributions."""
    result = {"python": platform.python_version(), "python_implementation": platform.python_implementation()}
    result.update({name: metadata.version(name) for name in sorted(set(names))})
    return result


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class ExperimentSpec:
    inputs: Mapping[str, Mapping[str, Any]]
    config: Mapping[str, Any]
    code_version: str
    code_digest: str
    dependency_lock_digest: str
    dependency_versions: Mapping[str, str]
    label_definition: Mapping[str, Any]
    split_protocol: Mapping[str, Any]
    seed: int
    input_scope: str = "used_records_only"

    def __post_init__(self) -> None:
        _nonempty(self.code_version, "code_version")
        _nonempty(self.input_scope, "input_scope")
        _digest(self.code_digest, "code_digest")
        _digest(self.dependency_lock_digest, "dependency_lock_digest")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("inputs", "config", "dependency_versions", "label_definition", "split_protocol"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            if name != "config" and not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, _canonical(value))
        for name, item in self.inputs.items():
            _nonempty(name, "input name")
            if not isinstance(item, Mapping) or "version" not in item or "content" not in item:
                raise ValueError(f"input {name} requires version and content")
            _nonempty(item["version"], f"inputs.{name}.version")
        for name, version in self.dependency_versions.items():
            _nonempty(name, "dependency name")
            _nonempty(version, f"dependency_versions.{name}")

    def canonical_payload(self) -> dict[str, Any]:
        return _canonical({"schema": "experiment-spec/1", **self.__dict__})

    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    spec: ExperimentSpec

    def __post_init__(self) -> None:
        _nonempty(self.trial_id, "trial_id")
        if not isinstance(self.spec, ExperimentSpec):
            raise ValueError("spec must be an ExperimentSpec")


class ExperimentRunner:
    """Execute a frozen trial list and retain successes, failures and retries.

    ``on_event(event)`` is an optional observer (for example MLflow). Events are
    fsynced locally before notification. Observer exceptions become separate
    ``integration_error`` events and never erase the original result.
    """

    def __init__(
        self,
        directory: str | Path,
        trials: Sequence[TrialSpec] | None = None,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.directory / "plan.json"
        self.log_path = self.directory / "events.jsonl"
        self.on_event = on_event
        with self._locked():
            if trials is not None:
                if not trials or len({trial.trial_id for trial in trials}) != len(trials):
                    raise ValueError("pre-registration requires a nonempty list of unique trial IDs")
                plan = {
                    "schema": "experiment-plan/1",
                    "trials": [
                        {"trial_id": trial.trial_id, "spec": trial.spec.canonical_payload(), "spec_fingerprint": trial.spec.fingerprint()}
                        for trial in trials
                    ],
                }
                plan["plan_fingerprint"] = fingerprint(plan)
                if self.plan_path.exists():
                    if self._read_plan() != plan:
                        raise ExperimentIntegrityError("existing plan cannot be replaced or extended")
                else:
                    if self.log_path.exists():
                        raise ExperimentIntegrityError("event log exists without its registered plan")
                    self._write_plan(plan)
                    self._append("plan_registered", None, 0, {"plan_fingerprint": plan["plan_fingerprint"]})
                    for trial in plan["trials"]:
                        self._append("trial_registered", trial["trial_id"], 0, trial)
            if not self.plan_path.exists():
                raise ExperimentIntegrityError("no registered plan; pass trials when creating a runner")
            self._verify(require_terminal=False)

    @contextmanager
    def _locked(self):
        # SQLite supplies an OS-released, cross-platform writer lock. The actual
        # audit record remains the readable, append-only JSONL file.
        connection = sqlite3.connect(self.directory / ".writer-lock.sqlite3", timeout=30)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _write_plan(self, plan: Mapping[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json(plan) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, self.plan_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_plan(self) -> dict[str, Any]:
        try:
            plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
            content = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
            if plan.get("schema") != "experiment-plan/1" or plan.get("plan_fingerprint") != fingerprint(content):
                raise ExperimentIntegrityError("registered plan fingerprint mismatch")
            ids = [item["trial_id"] for item in plan["trials"]]
            if not ids or len(set(ids)) != len(ids):
                raise ExperimentIntegrityError("registered trial IDs are empty or duplicated")
            for item in plan["trials"]:
                _nonempty(item["trial_id"], "registered trial_id")
                spec = dict(item["spec"])
                if spec.pop("schema", None) != "experiment-spec/1":
                    raise ExperimentIntegrityError("unsupported registered specification schema")
                ExperimentSpec(**spec)
                if fingerprint(item["spec"]) != item["spec_fingerprint"]:
                    raise ExperimentIntegrityError("registered specification fingerprint mismatch")
            return plan
        except ExperimentIntegrityError:
            raise
        except (OSError, TypeError, KeyError, ValueError, AttributeError) as exc:
            raise ExperimentIntegrityError(f"unreadable registered plan: {exc}") from exc

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        try:
            data = self.log_path.read_text(encoding="utf-8")
            if data and not data.endswith("\n"):
                raise ExperimentIntegrityError("event log has an incomplete final record")
            events = [json.loads(line) for line in data.splitlines()]
            previous = "0" * 64
            for sequence, event in enumerate(events):
                expected_keys = {"event_type", "trial_id", "attempt", "payload", "sequence", "previous_hash", "timestamp", "event_hash"}
                if (not isinstance(event, dict) or set(event) != expected_keys
                        or not isinstance(event["payload"], dict)
                        or isinstance(event["attempt"], bool) or not isinstance(event["attempt"], int)
                        or isinstance(event["sequence"], bool) or not isinstance(event["sequence"], int)):
                    raise ExperimentIntegrityError(f"invalid event structure at sequence {sequence}")
                content = {key: value for key, value in event.items() if key != "event_hash"}
                if (event.get("sequence") != sequence or event.get("previous_hash") != previous
                        or event.get("event_hash") != fingerprint(content)):
                    raise ExperimentIntegrityError(f"event hash-chain mismatch at sequence {sequence}")
                previous = event["event_hash"]
            return events
        except ExperimentIntegrityError:
            raise
        except (OSError, TypeError, KeyError, ValueError, AttributeError) as exc:
            raise ExperimentIntegrityError(f"unreadable event log: {exc}") from exc

    def _append(self, event_type: str, trial_id: str | None, attempt: int, payload: Mapping[str, Any], *, notify: bool = True) -> dict[str, Any]:
        events = self._read_events()
        event = {
            "event_type": event_type, "trial_id": trial_id, "attempt": attempt,
            "payload": _canonical(payload), "sequence": len(events),
            "previous_hash": events[-1]["event_hash"] if events else "0" * 64,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        event["event_hash"] = fingerprint(event)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if notify and self.on_event is not None:
            try:
                self.on_event(_canonical(event))
            except Exception as exc:
                self._append("integration_error", trial_id, attempt, {
                    "observed_sequence": event["sequence"], "error_type": type(exc).__name__, "message": str(exc),
                }, notify=False)
        return event

    def _verify(self, *, require_terminal: bool) -> dict[str, Any]:
        plan = self._read_plan()
        registered = {trial["trial_id"]: trial for trial in plan["trials"]}
        events = self._read_events()
        states: dict[str, str] = {}
        attempts: dict[str, int] = {}
        if not events or events[0]["event_type"] != "plan_registered" or events[0]["payload"].get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise ExperimentIntegrityError("event log is not anchored to the registered plan")
        terminal = {"trial_succeeded": "SUCCESS", "trial_failed": "FAILED", "trial_aborted": "ABORTED"}
        for event in events[1:]:
            kind, trial_id, attempt = event["event_type"], event["trial_id"], event["attempt"]
            if kind == "integration_error":
                continue
            if trial_id not in registered:
                raise ExperimentIntegrityError(f"event names an unregistered trial: {trial_id}")
            state = states.get(trial_id)
            if kind == "trial_registered":
                if state is not None or attempt != 0 or event["payload"] != registered[trial_id]:
                    raise ExperimentIntegrityError(f"invalid registration: {trial_id}")
                states[trial_id], attempts[trial_id] = "REGISTERED", 0
            elif kind == "trial_retry":
                if state not in {"FAILED", "ABORTED"} or attempt != attempts[trial_id] + 1 or event["payload"].get("previous_status") != state:
                    raise ExperimentIntegrityError(f"invalid retry: {trial_id}")
                states[trial_id], attempts[trial_id] = "RETRY_PENDING", attempt
            elif kind == "trial_started":
                expected = 1 if state == "REGISTERED" else attempts.get(trial_id)
                if (state not in {"REGISTERED", "RETRY_PENDING"} or attempt != expected
                        or event["payload"].get("spec_fingerprint") != registered[trial_id]["spec_fingerprint"]):
                    raise ExperimentIntegrityError(f"invalid start: {trial_id}")
                states[trial_id], attempts[trial_id] = "RUNNING", attempt
            elif kind in terminal:
                if (state != "RUNNING" and not (kind == "trial_aborted" and state in {"REGISTERED", "RETRY_PENDING"})) or attempt != attempts.get(trial_id):
                    raise ExperimentIntegrityError(f"invalid terminal event: {trial_id}")
                if kind == "trial_succeeded" and fingerprint(event["payload"].get("result")) != event["payload"].get("result_fingerprint"):
                    raise ExperimentIntegrityError(f"invalid result fingerprint: {trial_id}")
                states[trial_id] = terminal[kind]
            else:
                raise ExperimentIntegrityError(f"unsupported event type: {kind}")
        if set(states) != set(registered):
            raise ExperimentIntegrityError("registered trials are missing from the event log")
        incomplete = [trial_id for trial_id, state in states.items() if state not in terminal.values()]
        if require_terminal and incomplete:
            raise ExperimentIntegrityError(f"trials have no terminal outcome: {', '.join(incomplete)}")
        return {
            "complete": not incomplete,
            "trial_states": states,
            "attempts": attempts,
            "event_count": len(events),
            "plan_fingerprint": plan["plan_fingerprint"],
            "head_hash": events[-1]["event_hash"],
            "failed_attempts": sum(event["event_type"] == "trial_failed" for event in events),
            "aborted_attempts": sum(event["event_type"] == "trial_aborted" for event in events),
            "integration_errors": sum(event["event_type"] == "integration_error" for event in events),
            "formal_pbo": "NOT_COMPUTED",
            "formal_dsr": "NOT_COMPUTED",
            "statistical_validity": "NOT_ESTABLISHED_BY_ACCOUNTING",
        }

    def verify_integrity(
        self,
        *,
        require_terminal: bool = True,
        expected_plan_fingerprint: str | None = None,
        expected_head_hash: str | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            report = self._verify(require_terminal=require_terminal)
            if expected_plan_fingerprint is not None and report["plan_fingerprint"] != expected_plan_fingerprint:
                raise ExperimentIntegrityError("plan differs from the external checkpoint")
            if expected_head_hash is not None and report["head_hash"] != expected_head_hash:
                raise ExperimentIntegrityError("event head differs from the external checkpoint")
            for event in self._read_events():
                if event["event_type"] != "trial_succeeded":
                    continue
                for artifact in event["payload"]["result"].get("artifacts", {}).values():
                    path = (self.directory / artifact["path"]).resolve()
                    if not path.is_relative_to(self.directory.resolve()) or not path.is_file() or digest_file(path) != artifact["sha256"]:
                        raise ExperimentIntegrityError("recorded model/result artifact is missing or changed")
            return report

    def _capture_artifacts(self, result: Mapping[str, Any], trial_id: str, attempt: int) -> dict[str, Any]:
        result = _canonical(result)
        if "artifacts" in result:
            raise ValueError("artifacts is runner-owned; supply artifact_paths instead")
        paths = result.pop("artifact_paths", {})
        if not isinstance(paths, Mapping):
            raise ValueError("artifact_paths must map artifact names to existing file paths")
        if paths:
            destination = self.directory / "artifacts" / fingerprint(trial_id) / str(attempt)
            destination.mkdir(parents=True, exist_ok=True)
            captured = {}
            for name, path in paths.items():
                _nonempty(name, "artifact name")
                source = Path(path)
                target = destination / (fingerprint(name) + ".bin")
                with source.open("rb") as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                captured[name] = {"path": target.relative_to(self.directory).as_posix(), "sha256": digest_file(target)}
            result["artifacts"] = captured
        return result

    def run(self, trial_id: str, worker: Callable[[dict[str, Any]], Mapping[str, Any]], *, retry: bool = False) -> dict[str, Any]:
        with self._locked():
            report = self._verify(require_terminal=False)
            if trial_id not in report["trial_states"]:
                raise ValueError(f"trial was not preregistered: {trial_id}")
            state = report["trial_states"][trial_id]
            if retry:
                if state not in {"FAILED", "ABORTED"}:
                    raise ValueError("only failed or aborted trials can be retried")
                attempt = report["attempts"][trial_id] + 1
                self._append("trial_retry", trial_id, attempt, {"previous_status": state})
            else:
                if state != "REGISTERED":
                    raise ValueError("trial already started; explicit retry=True is required after failure/abort")
                attempt = 1
            spec = next(item["spec"] for item in self._read_plan()["trials"] if item["trial_id"] == trial_id)
            self._append("trial_started", trial_id, attempt, {"spec_fingerprint": fingerprint(spec)})
        try:
            result = worker(_canonical(spec))
            if not isinstance(result, Mapping):
                raise ValueError("worker result must be a JSON-compatible mapping")
            result = self._capture_artifacts(result, trial_id, attempt)
            payload = {"result": result, "result_fingerprint": fingerprint(result)}
            kind = "trial_succeeded"
        except BaseException as exc:
            kind = "trial_failed" if isinstance(exc, Exception) else "trial_aborted"
            payload = {"error_type": type(exc).__name__, "message": str(exc)}
            self._finish(trial_id, attempt, kind, payload)
            if not isinstance(exc, Exception):
                raise
            return {"trial_id": trial_id, "attempt": attempt, "status": "FAILED", **payload}
        self._finish(trial_id, attempt, kind, payload)
        return {"trial_id": trial_id, "attempt": attempt, "status": "SUCCESS", **payload}

    def _finish(self, trial_id: str, attempt: int, kind: str, payload: Mapping[str, Any]) -> None:
        with self._locked():
            report = self._verify(require_terminal=False)
            if report["trial_states"].get(trial_id) != "RUNNING" or report["attempts"].get(trial_id) != attempt:
                raise ExperimentIntegrityError("trial changed state while its worker was running")
            self._append(kind, trial_id, attempt, payload)

    def abort(self, trial_id: str, reason: str) -> dict[str, Any]:
        """Record a skipped trial or explicitly acknowledge an interrupted worker.

        This records state; it does not stop a still-running process. Ensure the
        worker has stopped before recovering a RUNNING attempt.
        """
        _nonempty(reason, "abort reason")
        with self._locked():
            report = self._verify(require_terminal=False)
            if report["trial_states"].get(trial_id) not in {"REGISTERED", "RUNNING", "RETRY_PENDING"}:
                raise ValueError("only registered, running or pending-retry trials can be aborted")
            return self._append("trial_aborted", trial_id, report["attempts"][trial_id], {"reason": reason})


def demo_experiment_specs() -> list[TrialSpec]:
    """Small synthetic preregistration list, with no market or account data."""
    base = {
        "inputs": {"synthetic_prices": {"version": "synthetic/1", "content": [100, 101, 99, 103]}},
        "code_version": "synthetic-demo/1",
        "code_digest": fingerprint("synthetic-demo-worker/1"),
        "dependency_lock_digest": fingerprint({"dependencies": "stdlib-only"}),
        "dependency_versions": capture_dependency_versions(),
        "label_definition": {"name": "next_observation_return", "horizon_observations": 1},
        "split_protocol": {"train_indices": [0, 1], "validation_indices": [2], "final_holdout_indices": [3]},
        "seed": 7,
    }
    return [TrialSpec(f"synthetic-{index}", ExperimentSpec(config={"threshold": value}, **base)) for index, value in enumerate((0.0, 0.01), 1)]


__all__ = [
    "ExperimentIntegrityError", "ExperimentSpec", "TrialSpec", "ExperimentRunner",
    "canonical_json", "fingerprint", "digest_file", "capture_dependency_versions", "demo_experiment_specs",
]
