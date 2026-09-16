"""Run explicit local profiles and preserve failures as auditable receipts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from uuid import uuid4


def profiles() -> dict[str, list[list[str]]]:
    python = sys.executable
    return {
        "tests": [[python, "-m", "pytest", "-q"]],
        "coverage": [[python, "-m", "pytest", "--cov=agipot_stock_prediction", "--cov-branch", "--cov-report=term-missing", "--cov-report=json:reports/coverage.json", "-q"]],
        "lint": [[python, "-m", "ruff", "check", "src", "tests", "scripts", "noxfile.py"]],
        "types": [[python, "-m", "mypy"]],
        "properties": [[python, "-m", "pytest", "tests/test_formula_properties.py", "tests/test_streaming_properties.py", "-q"]],
        "mutation": [[python, "-m", "mutmut", "run", "agipot_stock_prediction.research.cost_model.*"],
                     [python, "-m", "mutmut", "results"]],
        "security": [[python, "-m", "bandit", "-r", "src", "-c", "config/harness/bandit.yaml"],
                     [python, "-m", "pip_audit"]],
        "secrets": [["gitleaks", "git", "--redact", "--no-banner", "."]],
        "build": [[python, "-m", "build"], [python, "scripts/check_distribution.py"]],
        "benchmark": [[python, "-m", "pytest", "benchmarks", "--benchmark-json=reports/benchmark.json", "-q"]],
        "dvc": [["dvc", "repro"]],
        "promptfoo": [["npx", "--no-install", "promptfoo", "eval", "-c", "config/harness/promptfooconfig.yaml"]],
    }


def _persist(path: Path, receipt: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def _execute(argv: list[str], root: Path, output, timeout: int) -> int:
    process = subprocess.Popen(
        argv, cwd=root, text=True, stdout=output, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        start_new_session=os.name == "posix",
    )
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # On POSIX, stop the entire profile process group, including test or
        # mutation workers; a timed-out child must not keep modifying results.
        _stop(process)
        raise


def _code_identity(root: Path) -> dict:
    try:
        options = {"cwd": root, "stderr": subprocess.DEVNULL, "timeout": 10}
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], **options).decode().strip()
        names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], **options)
        digest = hashlib.sha256()
        for name in sorted(set(names.split(b"\0")) - {b""}):
            path = root / os.fsdecode(name)
            digest.update(len(name).to_bytes(8, "big") + name)
            if path.is_file():
                digest.update(b"file" + hashlib.sha256(path.read_bytes()).digest())
            else:
                digest.update(b"missing")
        return {"code_revision": revision, "code_worktree_digest": digest.hexdigest(),
                "code_digest_scope": "tracked and nonignored untracked files, current working tree bytes"}
    except (OSError, subprocess.SubprocessError):
        return {"code_revision": "unknown", "code_worktree_digest": None,
                "code_digest_scope": "unavailable; checkout identity not verified"}


def run_profile(name: str, root: Path, *, timeout: int = 600) -> dict:
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("root must be the project checkout")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer number of seconds")
    if name not in profiles():
        raise ValueError(f"unknown profile: {name}")
    commands = profiles()[name]
    if not commands or any(not argv or any(not isinstance(part, str) for part in argv) for argv in commands):
        raise ValueError("profile must contain nonempty command argument lists")
    reports = root / "reports" / "harness"
    reports.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    receipt = {"run_id": run_id, "profile": name, "started_at": datetime.now(timezone.utc).isoformat(),
               "commands": [], "status": "RUNNING", "timeout_seconds_per_command": timeout,
               "execution_interpreter": sys.executable, **_code_identity(root)}
    receipt_path = reports / f"{run_id}.json"
    lock = root / "uv.lock"
    receipt["dependency_lock_digest"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    _persist(receipt_path, receipt)
    failure = None
    for index, argv in enumerate(commands):
        if failure is not None:
            receipt["commands"].append({"argv": argv, "status": "NOT_RUN", "reason": "earlier command did not pass"})
            continue
        started = time.monotonic()
        log = reports / f"{run_id}-{index}.log"
        row = {"argv": argv, "log": str(log.relative_to(root)), "status": "RUNNING"}
        receipt["commands"].append(row)
        _persist(receipt_path, receipt)
        interrupt = None
        try:
            with log.open("w", encoding="utf-8") as output:
                returncode = _execute(argv, root, output, timeout)
            row.update(returncode=returncode, status="PASS" if returncode == 0 else "FAIL")
        except subprocess.TimeoutExpired:
            row.update(returncode=None, status="TIMEOUT")
        except OSError as error:
            row.update(returncode=None, status="UNAVAILABLE" if isinstance(error, (FileNotFoundError, PermissionError)) else "ERROR", reason=str(error))
        except BaseException as error:
            row.update(returncode=None, status="ABORTED", reason=f"{type(error).__name__}: {error}")
            interrupt = error
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if row["status"] != "PASS":
            failure = row["status"]
            receipt["status"] = failure
        _persist(receipt_path, receipt)
        if interrupt is not None:
            receipt["commands"].extend({"argv": pending, "status": "NOT_RUN", "reason": "run interrupted"} for pending in commands[index + 1:])
            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            _persist(receipt_path, receipt)
            raise interrupt
    receipt["status"] = failure or "PASS"
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    _persist(receipt_path, receipt)
    return receipt
