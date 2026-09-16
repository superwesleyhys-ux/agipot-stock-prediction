"""Launch one configured executor in an explicit workspace, with a task file."""
from pathlib import Path
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["aider", "mini-swe-agent", "openhands", "rd-agent"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--rd-command", choices=["fin_factor", "fin_model"])
    args = parser.parse_args()
    env = ROOT / ".harness/executors" / args.agent / "bin"
    if not env.is_dir():
        parser.error("Run scripts/bootstrap_agents.py --executors first")
    if not args.workspace.is_dir():
        parser.error("workspace must exist")
    if args.agent != "rd-agent" and (args.task is None or not args.task.is_file()):
        parser.error("--task must name an existing task file")
    if args.agent == "aider":
        argv = [str(env / "aider"), "--message-file", str(args.task.resolve()), "--no-auto-commits"]
    elif args.agent == "mini-swe-agent":
        argv = [str(env / "mini"), "-t", args.task.read_text()]
    elif args.agent == "openhands":
        argv = [str(env / "python"), str(ROOT / "scripts/openhands_task.py"), str(args.task.resolve()), "--workspace", str(args.workspace.resolve())]
    else:
        if not args.rd_command:
            parser.error("RD-Agent requires --rd-command and separately configured credentials/data")
        argv = [str(env / "rdagent"), args.rd_command]
    return subprocess.run(argv, cwd=args.workspace, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
