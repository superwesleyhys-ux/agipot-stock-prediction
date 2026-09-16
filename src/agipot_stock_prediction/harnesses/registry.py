"""Inspect integration capabilities without importing optional frameworks."""
from __future__ import annotations

import importlib.metadata
from importlib.resources import files
import json
from pathlib import Path
import shutil


def catalog() -> list[dict]:
    return json.loads(files(__package__).joinpath("catalog.json").read_text())


def doctor(root: Path) -> dict:
    root = root.resolve()
    rows = []
    for item in catalog():
        probes = {}
        version = None
        if item.get("distribution"):
            try:
                version = importlib.metadata.version(item["distribution"])
                probes["distribution_installed"] = True
            except importlib.metadata.PackageNotFoundError:
                probes["distribution_installed"] = False
        if item.get("executable"):
            probes["executable_on_path"] = shutil.which(item["executable"]) is not None
        if item.get("local_path"):
            probes["local_path_present"] = (root / item["local_path"]).exists()
        rows.append({**item, "installed": any(probes.values()), "installed_version": version,
                     "installation_probes": probes,
                     "runtime_status": "REQUIRES_PREREQUISITE" if item.get("prerequisites") else "NOT_EXECUTED",
                     "verified": False})
    receipts, receipt_errors = [], []
    for path in sorted((root / "reports" / "harness").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(record.get("commands"), list):
                raise ValueError("invalid receipt structure")
            if record.get("status") not in {"RUNNING", "PASS", "FAIL", "TIMEOUT", "UNAVAILABLE", "ERROR", "ABORTED"}:
                raise ValueError("unknown receipt status")
            if record["status"] == "PASS" and (not record["commands"] or any(
                not isinstance(command, dict) or command.get("status") != "PASS" or command.get("returncode") != 0
                for command in record["commands"]
            )):
                raise ValueError("PASS receipt has absent or unsuccessful commands")
            receipts.append({
                "receipt": path.relative_to(root).as_posix(),
                "run_id": record.get("run_id"), "profile": record.get("profile"),
                "recorded_status": record["status"], "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"), "code_revision": record.get("code_revision"),
                "code_worktree_digest": record.get("code_worktree_digest"),
                "scope": "recorded profile execution; not individual-tool verification",
            })
        except (OSError, ValueError, TypeError) as error:
            receipt_errors.append({"receipt": path.relative_to(root).as_posix(), "reason": str(error)})
    return {"schema_version": 1, "tool_count": len(rows), "tools": rows,
            "profile_receipts": receipts, "receipt_errors": receipt_errors,
            "note": "Installation and checked-in adapters do not prove successful runtime execution. See individual run receipts."}
