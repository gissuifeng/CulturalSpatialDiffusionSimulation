from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

def execute(script, arguments=(), environment=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if environment:
        env.update({key: str(value) for key, value in environment.items()})
    subprocess.run([sys.executable, str(SRC / script), *map(str, arguments)],
                   cwd=ROOT, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description="CSDS parameter sensitivity experiment")
    parser.add_argument("action", choices=["plan", "smoke", "run", "verify", "status", "summarize", "analyze", "test"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "config/design.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.action == "test":
        execute("test_pipeline.py")
    elif args.action == "analyze":
        execute("analyze.py", ["--results", args.output, "--output", args.output / "analysis"])
    else:
        options = [args.action, "--data", args.data, "--output", args.output,
                   "--settings", args.config, "--workers", args.workers]
        if args.action == "run": options.append("--accept-current-engine")
        if args.limit is not None: options += ["--limit", args.limit]
        if args.partial: options.append("--partial")
        execute("pipeline.py", options)

if __name__ == "__main__": main()
