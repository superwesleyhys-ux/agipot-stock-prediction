"""Install project-local agent skills; optionally prepare isolated executors.

No model is invoked. External repositories are pinned and their license files
remain in the checkout. Existing skill names are never overwritten.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*argv, **kwargs):
    subprocess.run(list(argv), check=True, **kwargs)


def bootstrap(executors: bool = False, build_browser: bool = False):
    if build_browser and shutil.which("bun") is None:
        raise SystemExit("--build-browser requires Bun on PATH; install Bun explicitly first")
    manifest = json.loads((ROOT / "config/harness/agents.json").read_text())
    skills = ROOT / ".agents/skills"
    skills.mkdir(parents=True, exist_ok=True)
    for name in ("superpowers", "gstack"):
        entry = manifest[name]
        checkout = ROOT / entry["path"]
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--filter=blob:none", "--no-checkout", entry["source"], str(checkout))
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != entry["revision"]:
            run("git", "-C", str(checkout), "checkout", entry["revision"])
        candidates = (checkout / "skills").glob("*/SKILL.md") if name == "superpowers" else checkout.glob("*/SKILL.md")
        for path in candidates:
            target = skills / (path.parent.name if name == "superpowers" else f"gstack-{path.parent.name}")
            if not target.exists() and not target.is_symlink():
                target.symlink_to(os.path.relpath(path.parent, target.parent), target_is_directory=True)
    browser_receipt = None
    if build_browser:
        checkout = ROOT / manifest["gstack"]["path"]
        run("bun", "install", "--frozen-lockfile", cwd=checkout)
        run("bash", "scripts/build.sh", cwd=checkout)
        browser_receipt = {
            "tool": "gstack",
            "revision": manifest["gstack"]["revision"],
            "build_returncode": 0,
            "model_executed": False,
        }
    staged = ROOT / ".harness/spec-kit-project"
    if not staged.exists():
        run("uvx", "--from", manifest["spec-kit"]["packages"][0], "specify", "init", str(staged), "--integration", "codex", "--integration-options=--skills", "--non-interactive", "--ignore-agent-tools")
    if not (ROOT / ".specify").exists():
        shutil.copytree(staged / ".specify", ROOT / ".specify")
    for path in (staged / ".agents/skills").iterdir():
        target = skills / path.name
        if not target.exists():
            shutil.copytree(path, target)
    receipts = []
    if executors:
        for name in ("aider", "mini-swe-agent", "openhands", "rd-agent"):
            environment = ROOT / ".harness/executors" / name
            run("uv", "venv", "--python", "3.12", str(environment)) if not environment.exists() else None
            completed = subprocess.run(["uv", "pip", "install", "--python", str(environment / "bin/python"), *manifest[name]["packages"]], check=False)
            receipts.append({"tool": name, "installation_returncode": completed.returncode, "model_executed": False})
    (ROOT / ".harness/bootstrap-receipt.json").write_text(json.dumps({"executors": receipts, "browser_build": browser_receipt}, indent=2))
    print("Project skills installed. They become available on the next agent turn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executors", action="store_true", help="also install isolated model executor environments")
    parser.add_argument("--build-browser", action="store_true", help="also install pinned gstack Bun dependencies and build its local browser tools")
    args = parser.parse_args()
    bootstrap(args.executors, args.build_browser)
