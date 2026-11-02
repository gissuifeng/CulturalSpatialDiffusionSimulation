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
    parser = argparse.ArgumentParser(description="CSDS spatial association and block resampling")
    parser.add_argument("action", choices=["moran", "blocks", "plot", "all"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.action in ["moran", "all"]:
        files = [args.data / "baseline_city_culture_endpoints.csv.gz",
                 args.data / "geography/city_point_shp/city.shp",
                 args.data / "geography/province_point_shp/province_point.shp"]
        missing = [str(p) for p in files if not p.is_file()]
        if missing: parser.error("Missing inputs: " + "; ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    env = {"CSDS_SPATIAL_OUTPUT": args.output.resolve(),
           "CSDS_ENDPOINTS": (args.data / "baseline_city_culture_endpoints.csv.gz").resolve(),
           "CSDS_GEOGRAPHY": (args.data / "geography").resolve()}
    tasks = {"moran": "make_residual_moran.py", "blocks": "spatial_block_refit.py", "plot": "optimized_spatial_layout.py"}
    for stage in (["moran", "blocks", "plot"] if args.action == "all" else [args.action]):
        execute(tasks[stage], environment=env)

if __name__ == "__main__": main()
