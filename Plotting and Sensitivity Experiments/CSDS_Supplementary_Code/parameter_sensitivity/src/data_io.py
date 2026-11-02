from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data"
OUTPUT = ROOT / "full_time_scatter_analysis"
BASELINE = {"r": 0.002, "gamma": 0.20, "theta": 0.15}
PALETTE = [
    "#e29690", "#8daec2", "#9ba3d6", "#c8929b",
    "#a888c1", "#a4cec8", "#c6b1d0",
]


def load_time_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    design = pd.read_csv(RESULTS / "parameter_design.csv")
    design = design.loc[design["family"].eq("core")].copy()
    params = design[["run_id", "config_id", "replicate", "seed", "r", "gamma", "theta"]]

    annual = pd.read_csv(RESULTS / "all_national_annual.csv.gz")
    annual = annual.loc[annual["family"].eq("core")]
    tv = (
        annual.groupby(["run_id", "year"], as_index=False)["delta_share_pp"]
        .apply(lambda x: np.abs(x).sum() / 2)
        .rename(columns={"delta_share_pp": "D_TV_pp"})
    )

    stages = pd.read_csv(RESULTS / "all_stages_annual.csv.gz")
    stages = stages.loc[stages["family"].eq("core") & stages["state"].isin(["I", "A"])]
    states = (
        stages.pivot(index=["run_id", "year"], columns="state", values="fraction")
        .mul(100)
        .rename(columns={"I": "I_pct", "A": "A_pct"})
        .reset_index()
    )
    time = tv.merge(states, on=["run_id", "year"], validate="one_to_one")
    time = time.merge(params, on="run_id", validate="many_to_one")
    expected = 375 * 41
    if len(time) != expected:
        raise ValueError(f"Expected {expected} run-year rows, found {len(time)}")
    endpoint = time.loc[time["year"].eq(time["year"].max())].copy()
    return time, endpoint
