from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
HERE = ROOT / "compact_sensitivity_analysis"
RESULTS = ROOT / "data"
FACTORS = ["r", "gamma", "theta"]
FACTOR_LABELS = {"r": r"$r$", "gamma": r"$\gamma$", "theta": r"$\theta$"}
BASELINE = {"r": 0.002, "gamma": 0.20, "theta": 0.15}


def load_run_summary() -> pd.DataFrame:
    design = pd.read_csv(RESULTS / "parameter_design.csv")
    design = design.loc[design["family"].eq("core")].copy()
    keep = ["run_id", "config_id", "replicate", "seed", *FACTORS]
    design = design[keep]

    annual = pd.read_csv(RESULTS / "all_national_annual.csv.gz")
    last_year = int(annual["year"].max())
    endpoint = annual.loc[
        annual["year"].eq(last_year) & annual["family"].eq("core")
    ]
    tv = (
        endpoint.groupby("run_id", as_index=False)["delta_share_pp"]
        .apply(lambda x: np.abs(x).sum() / 2)
        .rename(columns={"delta_share_pp": "D_TV_pp"})
    )

    combined_stages = RESULTS / "all_stages_annual.csv.gz"
    if combined_stages.exists():
        stage_data = pd.read_csv(combined_stages)
        endpoint_stages = stage_data.loc[
            stage_data["year"].eq(last_year)
            & stage_data["family"].eq("core")
            & stage_data["state"].isin(["I", "A"])
        ]
        stages = (
            endpoint_stages.pivot(index="run_id", columns="state", values="fraction")
            .mul(100)
            .rename(columns={"I": "I_pct", "A": "A_pct"})
            .reset_index()
        )
    else:
        stage_frames = []
        for run_id in design["run_id"]:
            path = RESULTS / "runs" / run_id / "stages_annual.csv"
            frame = pd.read_csv(path)
            row = frame.loc[
                frame["year"].eq(last_year) & frame["state"].isin(["I", "A"])
            ]
            values = row.set_index("state")["fraction"] * 100
            stage_frames.append(
                {"run_id": run_id, "I_pct": values["I"], "A_pct": values["A"]}
            )
        stages = pd.DataFrame(stage_frames)
    out = design.merge(tv, on="run_id", validate="one_to_one")
    out = out.merge(stages, on="run_id", validate="one_to_one")
    if len(out) != 375:
        raise ValueError(f"Expected 375 core runs, found {len(out)}")
    return out


def factorial_decomposition(data: pd.DataFrame, response: str) -> pd.DataFrame:
    """Balanced 5x5x5 factorial decomposition with seed as a complete block."""
    y = data[response]
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    levels = {f: np.sort(data[f].unique()) for f in FACTORS}
    means = {f: data.groupby(f)[response].mean() for f in FACTORS}
    effects: dict[str, float] = {}

    for factor in FACTORS:
        n_per_level = len(data) / len(levels[factor])
        effects[factor] = float(n_per_level * ((means[factor] - grand) ** 2).sum())

    pair_effects: dict[tuple[str, str], dict[tuple[float, float], float]] = {}
    for a, b in [("r", "gamma"), ("r", "theta"), ("gamma", "theta")]:
        cell = data.groupby([a, b])[response].mean()
        values = {}
        for i in levels[a]:
            for j in levels[b]:
                values[(i, j)] = float(
                    cell.loc[i, j] - means[a].loc[i] - means[b].loc[j] + grand
                )
        pair_effects[(a, b)] = values
        n_per_cell = len(data) / (len(levels[a]) * len(levels[b]))
        effects[f"{a}:{b}"] = float(n_per_cell * sum(v * v for v in values.values()))

    cell3 = data.groupby(FACTORS)[response].mean()
    residual3 = []
    for rv in levels["r"]:
        for gv in levels["gamma"]:
            for tv in levels["theta"]:
                fitted = (
                    grand
                    + (means["r"].loc[rv] - grand)
                    + (means["gamma"].loc[gv] - grand)
                    + (means["theta"].loc[tv] - grand)
                    + pair_effects[("r", "gamma")][(rv, gv)]
                    + pair_effects[("r", "theta")][(rv, tv)]
                    + pair_effects[("gamma", "theta")][(gv, tv)]
                )
                residual3.append(float(cell3.loc[rv, gv, tv] - fitted))
    effects["r:gamma:theta"] = float((len(data) / len(cell3)) * sum(v * v for v in residual3))

    seed_means = data.groupby("seed")[response].mean()
    effects["seed_block"] = float(
        (len(data) / seed_means.size) * ((seed_means - grand) ** 2).sum()
    )
    effects["residual"] = max(0.0, ss_total - sum(effects.values()))

    rows = []
    for term, ss in effects.items():
        rows.append(
            {
                "response": response,
                "term": term,
                "sum_squares": ss,
                "percent_total": 100 * ss / ss_total if ss_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def marginal_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response in ["D_TV_pp", "A_pct", "I_pct"]:
        for factor in FACTORS:
            for level, group in data.groupby(factor):
                by_seed = group.groupby("seed")[response].mean()
                rows.append(
                    {
                        "response": response,
                        "factor": factor,
                        "level": level,
                        "mean": group[response].mean(),
                        "q10": group[response].quantile(0.10),
                        "q90": group[response].quantile(0.90),
                        "seed_mean_sd": by_seed.std(ddof=1),
                        "n_runs": len(group),
                    }
                )
    return pd.DataFrame(rows)
