"""Local JSON command-line interface."""
import argparse
import json
from pathlib import Path
import sys

from .api import analyze_stock
from .demo import demo_input


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AGIPOT offline stock research and scoring")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="path to local JSON input")
    source.add_argument("--demo", action="store_true", help="use deterministic fictional data")
    parser.add_argument("--output", type=Path, help="write a JSON report instead of stdout")
    args = parser.parse_args(argv)
    try:
        payload = demo_input() if args.demo else json.loads(args.input.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        report = analyze_stock(payload)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"agipot-predict: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
