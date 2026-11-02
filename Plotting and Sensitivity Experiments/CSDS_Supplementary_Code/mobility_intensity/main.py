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
    parser = argparse.ArgumentParser(description="CSDS mobility intensity experiment")
    parser.add_argument("action", choices=["run", "analyze", "all", "test"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "config/design.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.action == "test":
        execute("test_pipeline.py")
        return
    if args.action in ["run", "all"]:
        execute("run_mobility.py", ["--data", args.data, "--output", args.output,
                                   "--config", args.config, "--workers", args.workers])
    if args.action in ["analyze", "all"]:
        execute("analyze_mobility.py", environment={"CSDS_RESULTS": args.output,
                "CSDS_ANALYSIS_OUTPUT": args.output / "analysis"})

if __name__ == "__main__": main()
