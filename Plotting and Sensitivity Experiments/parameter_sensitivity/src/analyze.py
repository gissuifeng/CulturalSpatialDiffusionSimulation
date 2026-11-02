"""Recompute the published core sensitivity summaries from existing outputs."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sensitivity_core as core
import data_io

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = p.parse_args()
    core.RESULTS = args.results.resolve()
    data_io.RESULTS = core.RESULTS
    design = pd.read_csv(core.RESULTS / "parameter_design.csv")
    design = design.loc[design.family.eq("core")]
    if len(design) != 375 or design.run_id.nunique() != 375:
        raise ValueError("This analysis requires exactly 375 distinct core runs.")
    levels = {f: sorted(design[f].unique().tolist()) for f in core.FACTORS}
    if any(len(v) != 5 for v in levels.values()):
        raise ValueError("Expected five levels for each factor.")
    counts = design.groupby(core.FACTORS).agg(n=("run_id","size"), seeds=("seed","nunique"))
    if len(counts) != 125 or not counts.n.eq(3).all() or not counts.seeds.eq(3).all():
        raise ValueError("Expected complete 5x5x5 design with three distinct seeds per cell.")
    expected_seeds = set(design.seed.unique())
    if len(expected_seeds) != 3 or any(set(g.seed) != expected_seeds for _,g in design.groupby(core.FACTORS)):
        raise ValueError("The same three seed blocks must occur in every configuration.")
    annual, endpoint = data_io.load_time_series()
    if annual.duplicated(["run_id", "year"]).any() or len(endpoint) != 375:
        raise ValueError("Duplicate or missing run-year observations.")
    if set(annual.year) != set(range(2020, 2061)) or not annual.groupby("run_id").size().eq(41).all():
        raise ValueError("Each run must contain all 41 annual boundaries, 2020–2060.")
    for name in ("D_TV_pp", "A_pct", "I_pct"):
        if not np.isfinite(annual[name]).all():
            raise ValueError(f"Nonfinite {name}")
    marginal = core.marginal_table(endpoint)
    effects = pd.concat([core.factorial_decomposition(endpoint, x)
                         for x in ("D_TV_pp","A_pct","I_pct")], ignore_index=True)
    if not np.allclose(effects.groupby("response").percent_total.sum(), 100, atol=1e-8):
        raise ValueError("Variance contributions do not sum to 100%.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    annual.to_csv(out / "all_core_run_year_metrics.csv.gz", index=False, compression="gzip")
    endpoint.to_csv(out / "endpoint_2060_metrics.csv", index=False, encoding="utf-8-sig")
    endpoint.to_csv(out / "run_level_summary.csv", index=False, encoding="utf-8-sig")
    marginal.to_csv(out / "marginal_responses.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(out / "factorial_variance_decomposition.csv", index=False, encoding="utf-8-sig")
    variability = endpoint.groupby(core.FACTORS)[["D_TV_pp","A_pct","I_pct"]].agg(["mean","std"])
    variability.to_csv(out / "replicate_variability.csv", encoding="utf-8-sig")
    (out / "analysis_manifest.json").write_text(json.dumps({
        "core_runs": len(endpoint), "run_year_rows": len(annual), "levels":levels,
        "source_results": str(core.RESULTS),
        "variance_method":"balanced factorial sum-of-squares decomposition; additive seed block",
        "percentile_interpretation":"scenario distribution, not confidence intervals",
        "seed_interactions":"included in residual; no inferential p-values"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(effects.to_string(index=False))
    print(f"Saved analysis and plot-ready data to {out}")

if __name__ == "__main__":
    main()
