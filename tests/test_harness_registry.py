import json
from pathlib import Path
import subprocess
import sys

import pytest

from agipot_stock_prediction.harnesses import cli, registry, runner


@pytest.fixture
def checkout(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='synthetic-checkout'\n")
    return tmp_path


def _receipt(root):
    paths = list((root / "reports" / "harness").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_installed_distribution_is_not_erased_by_missing_cli(monkeypatch, checkout):
    monkeypatch.setattr(registry, "catalog", lambda: [{
        "id": "synthetic", "distribution": "synthetic-package", "executable": "absent-cli",
        "prerequisites": "configured backend", "local_path": None,
    }])
    monkeypatch.setattr(registry.importlib.metadata, "version", lambda _: "1.0")
    monkeypatch.setattr(registry.shutil, "which", lambda _: None)
    item = registry.doctor(checkout)["tools"][0]
    assert item["installed"] is True
    assert item["installation_probes"] == {"distribution_installed": True, "executable_on_path": False}
    assert item["verified"] is False
    assert item["runtime_status"] == "REQUIRES_PREREQUISITE"


def test_success_receipt_is_saved_and_exposed_without_claiming_tool_verification(monkeypatch, checkout):
    monkeypatch.setattr(runner, "profiles", lambda: {"synthetic": [[sys.executable, "-c", "print('synthetic evidence')"]]})
    result = runner.run_profile("synthetic", checkout)
    assert result["status"] == "PASS"
    assert _receipt(checkout) == result
    assert "synthetic evidence" in (checkout / result["commands"][0]["log"]).read_text()
    report = registry.doctor(checkout)
    assert report["profile_receipts"][0]["recorded_status"] == "PASS"
    assert all(tool["verified"] is False for tool in report["tools"])


def test_failed_command_prevents_later_execution_and_records_not_run(monkeypatch, checkout):
    sentinel = checkout / "should-not-exist"
    monkeypatch.setattr(runner, "profiles", lambda: {"synthetic": [
        [sys.executable, "-c", "raise SystemExit(7)"],
        [sys.executable, "-c", f"from pathlib import Path; Path({str(sentinel)!r}).write_text('bad')"],
    ]})
    result = runner.run_profile("synthetic", checkout)
    assert result["status"] == "FAIL"
    assert result["commands"][0]["returncode"] == 7
    assert result["commands"][1]["status"] == "NOT_RUN"
    assert not sentinel.exists()


@pytest.mark.parametrize("error,status", [
    (FileNotFoundError("tool absent"), "UNAVAILABLE"),
    (PermissionError("not executable"), "UNAVAILABLE"),
    (OSError("resource exhausted"), "ERROR"),
    (subprocess.TimeoutExpired("synthetic", 1), "TIMEOUT"),
])
def test_launch_errors_and_timeouts_always_write_terminal_receipts(monkeypatch, checkout, error, status):
    monkeypatch.setattr(runner, "profiles", lambda: {"synthetic": [["synthetic"]]})

    def fail(*args):
        pending = _receipt(checkout)
        assert pending["status"] == "RUNNING"
        assert pending["commands"][0]["status"] == "RUNNING"
        raise error

    monkeypatch.setattr(runner, "_execute", fail)
    result = runner.run_profile("synthetic", checkout)
    assert result["status"] == status
    assert _receipt(checkout) == result
    assert "finished_at" in result


def test_interrupt_is_recorded_before_propagating(monkeypatch, checkout):
    monkeypatch.setattr(runner, "profiles", lambda: {"synthetic": [["first"], ["second"]]})

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_profile("synthetic", checkout)
    receipt = _receipt(checkout)
    assert receipt["status"] == "ABORTED"
    assert receipt["commands"][1]["status"] == "NOT_RUN"


@pytest.mark.parametrize("timeout", [0, -1, True, 0.1])
def test_timeout_must_be_positive_integer(checkout, timeout):
    with pytest.raises(ValueError, match="positive integer"):
        runner.run_profile("tests", checkout, timeout=timeout)


def test_empty_profile_cannot_report_pass(monkeypatch, checkout):
    monkeypatch.setattr(runner, "profiles", lambda: {"empty": []})
    with pytest.raises(ValueError, match="nonempty"):
        runner.run_profile("empty", checkout)


@pytest.mark.parametrize("status", ["FAIL", "TIMEOUT", "UNAVAILABLE", "ERROR", "ABORTED", "RUNNING"])
def test_cli_returns_nonzero_for_every_nonpassing_execution(monkeypatch, checkout, capsys, status):
    monkeypatch.setattr(cli, "run_profile", lambda *args, **kwargs: {"status": status})
    assert cli.main(["--root", str(checkout), "run", "tests"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_doctor_retains_corrupt_receipt_diagnostic(monkeypatch, checkout):
    monkeypatch.setattr(registry, "catalog", lambda: [])
    directory = checkout / "reports" / "harness"
    directory.mkdir(parents=True)
    (directory / "bad.json").write_text('{"status":"PASS","commands":[]}')
    report = registry.doctor(checkout)
    assert report["profile_receipts"] == []
    assert "absent or unsuccessful" in report["receipt_errors"][0]["reason"]


def test_security_profile_uses_valid_environment_audit_arguments():
    assert runner.profiles()["security"][-1] == [sys.executable, "-m", "pip_audit"]


def test_code_identity_includes_uncommitted_source_bytes(checkout):
    # A clean commit hash alone cannot identify the code under local testing.
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "synthetic"], check=True)
    first = runner._code_identity(checkout)
    (checkout / "new_module.py").write_text("coefficient = 1\n")
    second = runner._code_identity(checkout)
    assert first["code_revision"] == second["code_revision"]
    assert first["code_worktree_digest"] != second["code_worktree_digest"]
