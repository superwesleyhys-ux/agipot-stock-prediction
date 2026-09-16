from __future__ import annotations
import argparse
import json
from pathlib import Path

from .registry import catalog, doctor
from .runner import profiles, run_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AGIPOT research and engineering harnesses")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    commands.add_parser("doctor")
    run = commands.add_parser("run")
    run.add_argument("profile", choices=sorted(profiles()))
    run.add_argument("--timeout", type=int, default=600)
    commands.add_parser("demo")
    args = parser.parse_args(argv)
    if args.command == "catalog":
        result = {"tools": catalog()}
    elif args.command == "doctor":
        result = doctor(args.root)
    elif args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.root / "reports" / "harness-demo")
    else:
        try:
            result = run_profile(args.profile, args.root, timeout=args.timeout)
        except ValueError as error:
            parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") in (None, "PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
