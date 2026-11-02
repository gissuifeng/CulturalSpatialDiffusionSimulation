"""Auditable, single-run CSDS sensitivity pipeline; see README for engine deviations."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

KEYS = ["r", "muH", "muI", "muA", "gamma", "theta"]
STATE_TYPES = ["N", "H", "I", "A", "R"]
ENGINE_VERSION = "csds-count-conservative-v1"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulation_digest(path):
    """Hash simulation/design semantics, not later caption or plotting edits."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    names={'normalize_gene','load_inputs','make_design','migrate','status_update','acculturate','run_one'}
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            nodes.append(ast.dump(node,include_attributes=False))
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'KEYS','STATE_TYPES','ENGINE_VERSION'} for t in node.targets):
            nodes.append(ast.dump(node,include_attributes=False))
    return hashlib.sha256('\n'.join(nodes).encode()).hexdigest()


def read_settings(path=None):
    return json.loads(Path(path or HERE / "design.json").read_text(encoding="utf-8"))


def normalize_gene(gene):
    parts = gene.split("_")
    if parts[0] in ("I", "A"):
        return "_".join([parts[0], *sorted(parts[1:])])
    return gene


def load_inputs(data, settings):
    data = Path(data)
    city = pd.read_csv(data / "city_pop.csv", dtype={"city_id": str})
    init = pd.read_csv(data / "city_culture_pop.csv", dtype={"CityID": str})
    od = pd.read_csv(data / "city_pop_flow.csv", dtype={"o_city_id": str, "d_city_id": str})
    strength = pd.read_csv(data / "cultural_strength.csv")
    if city.city_id.duplicated().any() or init.CityID.duplicated().any():
        raise ValueError("Duplicate city IDs")
    ids = city.city_id.tolist()
    if set(ids) != set(init.CityID):
        raise ValueError("City metadata and initial-state city IDs differ")
    cultures = sorted(city.culture_type.unique().tolist())
    genes = [f"{s}_{c}_0" for c in cultures for s in ["N", "H"]] + ["R_0_0"]
    genes += [f"{s}_{a}_{b}" for a, b in itertools.combinations(cultures, 2) for s in ["I", "A"]]
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: i for i, c in enumerate(cultures)}
    ii = {c: i for i, c in enumerate(ids)}
    init = init.set_index("CityID").loc[ids]
    x = np.zeros((len(ids), len(genes)), dtype=np.int64)
    reversed_cols = []
    for g in init.columns:
        ng = normalize_gene(g)
        if ng not in gi:
            raise ValueError(f"Unknown state column {g}")
        values = init[g].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid initial count {g}")
        x[:, gi[ng]] += values.astype(np.int64)
        if ng != g:
            reversed_cols.append(g)
    unknown = ~od.o_city_id.isin(ids) | ~od.d_city_id.isin(ids)
    excluded = od[unknown].copy()
    if len(excluded) and not settings["exclude_unknown_od"]:
        raise ValueError("Unknown OD endpoints: explicit exclusion permission required")
    od = od[~unknown].copy()
    if (od.flow_pop < 0).any() or not np.isfinite(od.flow_pop).all():
        raise ValueError("Invalid OD population")
    out = []
    for ident in ids:
        f = od[(od.o_city_id == ident) & (od.flow_pop > 0)]
        dest = np.array([ii[z] for z in f.d_city_id], dtype=int)
        amounts = f.flow_pop.to_numpy(dtype=np.int64)
        out.append((dest, amounts, int(amounts.sum())))
    k = np.ones((len(cultures), len(cultures)))
    seen = set()
    for row in strength.itertuples(index=False):
        k[ci[row.culture1], ci[row.culture2]] = float(row.cultural_strength)
        seen.add((row.culture1, row.culture2))
    missing = [(a, b) for a in cultures for b in cultures if a != b and (a, b) not in seen]
    if missing or not np.isfinite(k).all() or (k < 0).any():
        raise ValueError(f"Invalid/incomplete cultural strength matrix: {missing}")
    # Carrier counts double-count I/A labels; population does not.
    carriers = np.zeros((len(genes), len(cultures)), dtype=np.int64)
    state = np.zeros((len(genes), 5), dtype=np.int64)
    a_idx = np.full(len(genes), -1, dtype=int)
    b_idx = a_idx.copy()
    for g, j in gi.items():
        s, a, b = g.split("_")
        state[j, STATE_TYPES.index(s)] = 1
        if s != "R":
            a_idx[j] = ci[a]
            carriers[j, ci[a]] = 1
        if s in ("I", "A"):
            b_idx[j] = ci[b]
            carriers[j, ci[b]] = 1
    populations = x.sum(axis=1)
    audit = {
        "cities": len(ids), "cultures": cultures, "states": len(genes),
        "initial_population": int(x.sum()), "metadata_population": int(city.city_pop.sum()),
        "city_population_metadata_disagreements": int((populations != city.city_pop.to_numpy()).sum()),
        "initial_I": int((x @ state)[:, 2].sum()), "initial_A": int((x @ state)[:, 3].sum()),
        "normalized_reversed_columns": reversed_cols,
        "original_od_rows": len(od) + len(excluded), "included_od_rows": len(od),
        "excluded_od_rows": len(excluded), "excluded_flow": int(excluded.flow_pop.sum()),
        "excluded_flow_fraction": float(excluded.flow_pop.sum() / (od.flow_pop.sum()+excluded.flow_pop.sum())),
        "annual_included_flow": int(od.flow_pop.sum()),
        "source_sha256": {p.name: sha(p) for p in sorted(data.glob("*.csv"))},
        "strength_min": float(strength.cultural_strength.min()),
        "strength_max": float(strength.cultural_strength.max()),
        "permission": "User explicitly approved exclusion of OD records involving absent city 442100.",
        "population_denominator": "sum of provided initial-state counts; city_pop supplies IDs and native cultures",
    }
    native = np.array([ci[c] for c in city.culture_type], dtype=int)
    return dict(x=x, ids=ids, genes=genes, gi=gi, ci=ci, cultures=cultures,
                out=out, k=k, carriers=carriers, state=state, a=a_idx, b=b_idx,
                native=native, audit=audit, excluded=excluded)


def make_design(settings):
    base = settings["baseline"]
    rows = [{"run_id": "baseline", "family": "baseline", "varied": "none", **base}]
    for p in KEYS:
        for i, v in enumerate(settings["levels"][p]):
            if v == base[p]:
                continue
            rows.append({"run_id": f"oat_{p}_{i}", "family": "oat", "varied": p, **base, p: v})
    u = qmc.LatinHypercube(6, seed=settings["design_seed"]).random(settings["joint_lhs"])
    low = np.array([min(settings["levels"][p]) for p in KEYS])
    high = np.array([max(settings["levels"][p]) for p in KEYS])
    samples = low + u*(high-low)
    samples[:, 0] = np.exp(np.log(low[0]) + u[:, 0]*(np.log(high[0])-np.log(low[0])))
    for i, vector in enumerate(samples):
        rows.append({"run_id": f"joint_lhs_{i+1:02}", "family": "joint", "varied": "all",
                     **dict(zip(KEYS, vector.tolist()))})
    stresses = [low, high,
                np.array([low[0], high[1], high[2], high[3], high[4], low[5]]),
                np.array([high[0], low[1], low[2], low[3], low[4], high[5]])]
    if settings["joint_stress"] != 4:
        raise ValueError("This design explicitly defines four stress cases")
    for i, vector in enumerate(stresses):
        rows.append({"run_id": f"joint_stress_{i+1}", "family": "joint", "varied": "all",
                     **dict(zip(KEYS, vector.tolist()))})
    frame = pd.DataFrame(rows)
    if frame[KEYS].duplicated().any():
        raise ValueError("Duplicate parameter combinations")
    frame["seed"] = settings["seed"]
    for p in ["H", "I", "A"]:
        frame[f"beta{p}"] = frame.r * frame[f"mu{p}"]
    frame["external_beta_envelope_all_three"] = frame[["betaH", "betaI", "betaA"]].apply(
        lambda r: bool(((r >= .000701) & (r <= .001137)).all()), axis=1)
    return frame


def migrate(x, inputs, year, seed):
    """Preserve input city order and stochastic destination allocation; no overdraw."""
    for c, (dest, amounts, total_out) in enumerate(inputs["out"]):
        if not total_out:
            continue
        if total_out > x[c].sum():
            raise ValueError(f"Origin {inputs['ids'][c]} cannot supply outflow in year {year}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, year, c, 0]))
        counts = rng.multivariate_hypergeometric(x[c], total_out, method="marginals")
        allocation = rng.multinomial(counts, amounts / total_out)
        x[c] -= counts
        # Multiple destination records, if present, still accumulate correctly.
        np.add.at(x, dest, allocation.T)
    return x


def status_update(x, inputs):
    gi, cultures = inputs["gi"], inputs["cultures"]
    for c, native in enumerate(inputs["native"]):
        native_name = cultures[native]
        for name in cultures:
            if name != native_name:
                src, dst = gi[f"N_{name}_0"], gi[f"H_{name}_0"]
            else:
                src, dst = gi[f"H_{name}_0"], gi[f"N_{name}_0"]
            x[c, dst] += x[c, src]
            x[c, src] = 0


def acculturate(x, inputs, params, choice_bits):
    gi, cultures, k = inputs["gi"], inputs["cultures"], inputs["k"]
    for c, native in enumerate(inputs["native"]):
        before = x[c].copy()
        total = int(before.sum())
        n_key = gi[f"N_{cultures[native]}_0"]
        n = int(before[n_key])
        delta = np.zeros_like(before)
        for j in np.flatnonzero(before):
            s = inputs["genes"][j][0]
            count = int(before[j])
            if s not in ("H", "I", "A"):
                continue
            a, b = inputs["a"][j], inputs["b"][j]
            if s == "H":
                target = a
            else:
                target = b if native == a else a if native == b else (a if choice_bits[c, j] == 0 else b)
            if target == native:
                raise ValueError("Invalid homogeneous dual label")
            strength = k[target, native]
            pair = sorted([cultures[native], cultures[target]])
            dest = gi[f"{'A' if s == 'A' else 'I'}_{pair[0]}_{pair[1]}"]
            # Same count on each side of each channel: exact mass conservation.
            transfer = int(params['r'] * params[f'mu{s}'] * strength * n * count / total)
            delta[n_key] -= transfer
            delta[dest] += transfer
            if s == "H":
                converted = int(count * params["gamma"] * strength)
                delta[j] -= converted
                delta[dest] += converted
            elif s == "I":
                converted = int(count * params["theta"])
                a_pair = sorted([cultures[a], cultures[b]])
                a_dest = gi[f"A_{a_pair[0]}_{a_pair[1]}"]
                delta[j] -= converted
                delta[a_dest] += converted
        if (before + delta < 0).any() or delta.sum() != 0:
            raise ValueError("Invalid acculturation update")
        x[c] += delta


def run_one(row, data, settings, output, fingerprint):
    folder = Path(output) / "runs" / row["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    meta_file = folder / "metadata.json"
    if meta_file.exists() and (folder / "states.npz").exists():
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        if old.get("fingerprint") == fingerprint and old.get("params") == {p: row[p] for p in KEYS}:
            return row["run_id"], "cached", old["seconds"]
        raise ValueError(f"Existing result has a different provenance: {folder}")
    t0 = time.perf_counter()
    inp = load_inputs(data, settings)
    x = inp["x"].copy()
    total = int(x.sum())
    history = np.zeros((settings["years"] + 1, *x.shape), dtype=np.int64)
    history[0] = x
    seed = settings["seed"]
    choices = np.random.default_rng(seed + 101).integers(0, 2, size=(settings["years"], *x.shape), dtype=np.int8)
    for y in range(1, settings["years"] + 1):
        migrate(x, inp, y, seed)
        status_update(x, inp)
        acculturate(x, inp, row, choices[y-1])
        if (x < 0).any() or int(x.sum()) != total:
            raise ValueError(f"Conservation/nonnegativity failure in {row['run_id']} year {y}")
        history[y] = x
    np.savez_compressed(folder / "states.npz", counts=history,
                        city_ids=np.asarray(inp["ids"]), genes=np.asarray(inp["genes"]),
                        years=np.arange(settings["years"]+1)+settings["start_year"])
    seconds = time.perf_counter() - t0
    write_json(meta_file, dict(run_id=row["run_id"], params={p: row[p] for p in KEYS},
                              fingerprint=fingerprint, seconds=seconds, seed=seed,
                              population_conserved=True, min_state=int(history.min()), repeats=1))
    return row["run_id"], "completed", seconds


def compare_metric(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def analyze(output, data, settings):
    output = Path(output)
    inp = load_inputs(data, settings)
    provenance=json.loads((output/'provenance.json').read_text(encoding='utf-8'))
    if provenance['input_sha256']!=inp['audit']['source_sha256'] or provenance['settings']!=settings:
        raise ValueError('Saved simulations do not match the requested input data/settings')
    design = pd.read_csv(output / "parameter_design.csv")
    expected=make_design(settings)
    if design.run_id.tolist()!=expected.run_id.tolist():
        raise ValueError('Saved parameter design differs from the recorded design')
    np.testing.assert_allclose(design[KEYS],expected[KEYS],rtol=1e-12,atol=1e-15)
    initial = inp["x"] @ inp["carriers"]
    ini_carriers = initial.sum(axis=0)
    ini_share = ini_carriers / ini_carriers.sum()
    ini_city_share = initial / initial.sum(axis=1, keepdims=True)
    base_city = None
    annual, city_rows, states, final = [], [], [], {}
    for row in design.to_dict("records"):
        path = output / "runs" / row["run_id"] / "states.npz"
        if not path.exists():
            raise ValueError(f"Missing run {row['run_id']}; cannot report complete experiment")
        meta=json.loads((path.parent/'metadata.json').read_text(encoding='utf-8'))
        if meta['fingerprint']!=provenance['fingerprint'] or meta['repeats']!=1:
            raise ValueError(f"Incompatible run metadata for {row['run_id']}")
        np.testing.assert_allclose([meta['params'][p] for p in KEYS],[row[p] for p in KEYS],rtol=1e-12,atol=1e-15)
        with np.load(path) as z:
            hist = z["counts"]
        carried = hist @ inp["carriers"]
        total_pop = hist.sum(axis=(1, 2))
        national = carried.sum(axis=1)
        shares = national / national.sum(axis=1, keepdims=True)
        state_count = (hist @ inp["state"]).sum(axis=1)
        for y in range(len(hist)):
            for i, culture in enumerate(inp["cultures"]):
                annual.append(dict(run_id=row["run_id"], family=row["family"],
                                   year=settings["start_year"]+y, elapsed=y, culture=culture,
                                   carriers=int(national[y,i]), label_share=shares[y,i],
                                   prevalence=national[y,i]/total_pop[y],
                                   growth_pct=100*(national[y,i]/ini_carriers[i]-1),
                                   delta_share_pp=100*(shares[y,i]-ini_share[i])))
            for i,s in enumerate(STATE_TYPES):
                states.append(dict(run_id=row["run_id"], year=settings["start_year"]+y,
                                   elapsed=y, state=s, population=int(state_count[y,i]),
                                   fraction=state_count[y,i]/total_pop[y]))
        end_city = carried[-1] / carried[-1].sum(axis=1, keepdims=True)
        final[row["run_id"]] = (shares[-1], national[-1], end_city)
        if row["run_id"] == "baseline":
            base_city = end_city
        for y in [0, 10, 20, 40]:
            if y >= len(hist):
                continue
            for c, ident in enumerate(inp["ids"]):
                denom = carried[y,c].sum()
                for i,culture in enumerate(inp["cultures"]):
                    city_rows.append(dict(run_id=row["run_id"], elapsed=y,
                                          year=settings["start_year"]+y, city_id=ident, culture=culture,
                                          carriers=int(carried[y,c,i]), label_share=carried[y,c,i]/denom))
    annual = pd.DataFrame(annual)
    annual.to_csv(output / "national_annual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(states).to_csv(output / "state_annual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(city_rows).to_csv(output / "city_checkpoints.csv.gz", index=False, compression="gzip")
    baseline = annual[(annual.run_id == "baseline") & (annual.elapsed == settings["years"])].set_index("culture")
    tol = settings["direction_tolerance_pp"]
    def direction(v):
        return np.where(v > tol, 1, np.where(v < -tol, -1, 0))
    robust = []
    for row in design.to_dict("records"):
        share, national, city_share = final[row["run_id"]]
        delta = 100*(share-ini_share)
        base_delta = baseline.loc[inp["cultures"],"delta_share_pp"].to_numpy()
        for i,culture in enumerate(inp["cultures"]):
            a = (city_share-ini_city_share)[:,i]
            b = (base_city-ini_city_share)[:,i]
            robust.append(dict(run_id=row["run_id"], family=row["family"], culture=culture,
                               delta_share_pp=delta[i], departure_from_baseline_pp=delta[i]-base_delta[i],
                               direction=int(direction(delta)[i]),
                               baseline_direction=int(direction(base_delta)[i]),
                               direction_matches=bool(direction(delta)[i]==direction(base_delta)[i]),
                               city_change_spearman=compare_metric(a,b),
                               city_level_spearman=compare_metric(city_share[:,i],base_city[:,i]),
                               city_share_rmse_pp=float(np.sqrt(np.mean((city_share[:,i]-base_city[:,i])**2))*100)))
    robust = pd.DataFrame(robust)
    robust.to_csv(output / "scenario_comparison.csv", index=False, encoding="utf-8-sig")
    oat = annual[(annual.family == "oat") & (annual.elapsed == settings["years"])].merge(
        design[["run_id","varied"]], on="run_id")
    heat = pd.DataFrame(index=KEYS, columns=inp["cultures"], dtype=float)
    for p in KEYS:
        for c in inp["cultures"]:
            vals = oat[(oat.varied==p)&(oat.culture==c)].delta_share_pp.to_numpy()
            heat.loc[p,c] = np.max(np.abs(vals-baseline.loc[c,"delta_share_pp"]))
    heat.to_csv(output / "oat_effect_pp.csv", encoding="utf-8-sig")
    # Descriptive rank associations, not variance contributions or significance tests.
    joint = annual[(annual.family == "joint") & (annual.elapsed == settings["years"])].merge(design[["run_id",*KEYS]], on="run_id")
    correlations = pd.DataFrame(index=KEYS, columns=inp["cultures"], dtype=float)
    for p in KEYS:
        for c in inp["cultures"]:
            z = joint[joint.culture==c]
            correlations.loc[p,c] = compare_metric(z[p],z.delta_share_pp)
    correlations.to_csv(output / "joint_rank_associations.csv", encoding="utf-8-sig")
    plot_figures(output, design, annual, robust, heat, correlations, inp, settings)
    make_report(output, design, annual, robust, heat, inp, settings)


def plot_figures(output, design, annual, robust, heat, correlations, inp, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    colors = plt.colormaps["tab20"](np.linspace(0,1,11))
    cultures=inp["cultures"]
    labels={"r":"r", "muH":r"$\mu_H$", "muI":r"$\mu_I$", "muA":r"$\mu_A$", "gamma":r"$\gamma$", "theta":r"$\theta$"}
    def save(fig,name):
        fig.savefig(figdir/f"{name}.png",bbox_inches="tight")
        fig.savefig(figdir/f"{name}.pdf",bbox_inches="tight")
        plt.close(fig)
    endpoint=annual[annual.elapsed==settings["years"]].merge(design[["run_id",*KEYS,"varied"]],on="run_id")
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for ax,p in zip(axs.flat,KEYS):
        df=endpoint[(endpoint.varied==p)|(endpoint.run_id=="baseline")]
        for c,color in zip(cultures,colors):
            z=df[df.culture==c].sort_values(p)
            ax.plot(z[p],z.delta_share_pp,"o-",color=color,ms=3,lw=1,label=c)
        ax.axvline(settings["baseline"][p],color=".4",ls=":",lw=1)
        ax.axhline(0,color=".6",lw=.7)
        if p=="r":
            ax.set_xscale("log")
            ax.set_xticks(settings["levels"][p], [f"{v:g}" for v in settings["levels"][p]], minor=False)
            ax.tick_params(axis="x",which="minor",bottom=False,labelbottom=False)
            plt.setp(ax.get_xticklabels(),rotation=25,ha="right",fontsize=8)
        ax.set(xlabel=labels[p],ylabel="Change in national label share (pp)",title=f"({chr(97+KEYS.index(p))}) {labels[p]}")
    handles,leg=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,leg,loc="outside lower center",ncol=11,title="Culture code")
    save(fig,"01_oat_response")
    def heatmap(table,name,title,label,cmap="viridis",signed=False):
        fig,ax=plt.subplots(figsize=(10, max(3,table.shape[0]*.3+1.5)),layout="constrained")
        values=table.to_numpy(dtype=float)
        norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1) if signed else None
        im=ax.imshow(values,aspect="auto",cmap=cmap,norm=norm)
        ax.set(xticks=np.arange(len(table.columns)),xticklabels=table.columns,
               yticks=np.arange(len(table.index)),yticklabels=table.index,title=title,xlabel="Culture code")
        if len(table.index)<=6:
            for i in range(len(table.index)):
                for j in range(len(table.columns)):
                    text_value=f"{values[i,j]:.2f}" if signed else f"{values[i,j]:.4f}"
                    ax.text(j,i,text_value,ha="center",va="center",fontsize=8,
                            color="white" if (abs(values[i,j])>.8 if signed else values[i,j] < np.nanmax(values)*.55) else "black")
        fig.colorbar(im,ax=ax,label=label,shrink=.85)
        save(fig,name)
    heatmap(heat.rename(index=labels),"02_oat_effect","Single-factor endpoint departure from baseline","Maximum absolute departure (pp)")
    table=robust[robust.family=="joint"].pivot(index="run_id",columns="culture",values="departure_from_baseline_pp").reindex(columns=cultures)
    fig,ax=plt.subplots(figsize=(10,8),layout="constrained")
    limit=float(np.abs(table.to_numpy()).max()) or 1
    im=ax.imshow(table,aspect="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set(xticks=range(11),xticklabels=cultures,yticks=range(len(table)),yticklabels=table.index,
           title="Joint parameter scenarios: endpoint departures",xlabel="Culture code")
    fig.colorbar(im,ax=ax,label="Departure from baseline label share (pp)",shrink=.8)
    save(fig,"03_joint_departures")
    heatmap(correlations.rename(index=labels),"04_joint_rank_associations","Descriptive rank associations (20 joint cases; not Sobol indices)","Spearman rank correlation",cmap="RdBu_r",signed=True)
    fig,axs=plt.subplots(3,4,figsize=(12,8),layout="constrained")
    for ax,c in zip(axs.flat,cultures):
        z=annual[annual.culture==c]
        baseline=z[z.run_id=="baseline"]
        envelope=z[z.family=="joint"].groupby("elapsed").delta_share_pp.agg(["min","max"])
        ax.fill_between(envelope.index,envelope["min"],envelope["max"],color="#dd8452",alpha=.25,label="20-case scenario envelope")
        ax.plot(baseline.elapsed,baseline.delta_share_pp,color="black",lw=1.5,label="Baseline, single run")
        ax.axhline(0,color=".6",lw=.7)
        ax.set(title=c,xlabel="Years after supplied initial state",ylabel="Label share change (pp)")
    axs.flat[-1].axis("off")
    handles,leg=axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(handles,leg,loc="center",frameon=False,fontsize=9)
    save(fig,"05_scenario_envelopes")
    table=robust[robust.family=="joint"].pivot(index="run_id",columns="culture",values="city_change_spearman").reindex(columns=cultures)
    heatmap(1-table,"06_city_change_similarity","City-change rank deviation from baseline (small values)","1 - Spearman correlation (zero = identical ranks)",cmap="viridis",signed=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    df=robust[robust.family=="joint"]
    match=df.groupby("culture").direction_matches.mean().reindex(cultures)*100
    axs[0].bar(cultures,match,color="#4477aa")
    axs[0].set(ylim=(0,105),ylabel="Scenarios matching baseline direction (%)",title="National share direction; tolerance 0.01 pp")
    vals=[df[df.culture==c].city_share_rmse_pp.to_numpy() for c in cultures]
    axs[1].boxplot(vals,tick_labels=cultures,showfliers=True)
    axs[1].set(ylabel="City label-share RMSE vs baseline (pp)",title="Scenario spread, not replicate uncertainty")
    save(fig,"07_robustness_summary")
    # Different outputs can have different sensitivities: theta affects I/A, not labels.
    stage=pd.read_csv(output/"state_annual.csv")
    stage=stage[stage.elapsed==settings['years']].merge(design[["run_id",*KEYS,"varied"]],on='run_id')
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    stage_colors={'H':'#4477aa','I':'#ee7733','A':'#228833'}
    for ax,p in zip(axs.flat,KEYS):
        df=stage[(stage.varied==p)|(stage.run_id=='baseline')]
        for s,color in stage_colors.items():
            z=df[df.state==s].sort_values(p)
            ax.plot(z[p],z.fraction*100,'o-',color=color,ms=4,label=s)
        ax.axvline(settings['baseline'][p],color='.4',ls=':',lw=1)
        if p=='r':
            ax.set_xscale('log')
            ax.set_xticks(settings['levels'][p],[f'{v:g}' for v in settings['levels'][p]])
            ax.tick_params(axis='x',which='minor',bottom=False,labelbottom=False)
            plt.setp(ax.get_xticklabels(),rotation=25,ha='right',fontsize=8)
        ax.set(xlabel=labels[p],ylabel='Fraction of total population (%)',title=f'({chr(97+KEYS.index(p))}) {labels[p]}')
    handles,leg=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,leg,loc='outside lower center',ncol=3,title='Cultural state at year 40')
    save(fig,'08_state_composition_response')


def make_report(output,design,annual,robust,heat,inp,settings):
    joint=robust[robust.family=="joint"]
    endpoint=annual[(annual.elapsed==settings["years"]) & (annual.family=="joint")]
    base=annual[(annual.elapsed==settings["years"]) & (annual.run_id=="baseline")].set_index("culture")
    summary=[]
    for c in inp["cultures"]:
        z=joint[joint.culture==c]
        e=endpoint[endpoint.culture==c]
        summary.append(dict(culture=c,baseline_change_pp=base.loc[c,"delta_share_pp"],
                            joint_min_change_pp=e.delta_share_pp.min(),joint_max_change_pp=e.delta_share_pp.max(),
                            joint_direction_match_pct=z.direction_matches.mean()*100,
                            min_city_change_rho=z.city_change_spearman.min(),
                            median_city_change_rho=z.city_change_spearman.median(),
                            max_city_rmse_pp=z.city_share_rmse_pp.max()))
    summary=pd.DataFrame(summary)
    summary.to_csv(output/"culture_robustness_summary.csv",index=False,encoding="utf-8-sig")
    strengths=heat.max(axis=1).sort_values(ascending=False)
    names={"B":"内蒙古","C":"新疆","D":"黄土高原","E":"华北","F":"西藏","G":"云贵","J":"川渝","K":"荆楚","L":"江南","M":"华南","X":"东北"}
    lines=["# 第一轮CSDS参数敏感性实验报告", "", "## 1 实验完成情况", "",
           f"完成{len(design)}组不同参数、每组单次运行，模拟{settings['years']}年；未做任何重复实验。初态记作t=0，2020—2060为稿件情景年份标签，不把输入重新认定为实测2020年数据。",
           "", f"输入：{inp['audit']['cities']}城市、11种文化、133种人数状态，总人口{inp['audit']['initial_population']:,}。",
           f"经用户授权排除{inp['audit']['excluded_od_rows']}条未知城市OD，流量{inp['audit']['excluded_flow']:,}，占原流量{inp['audit']['excluded_flow_fraction']*100:.6f}%。其余原始输入不改写。",
           "", "25组基准/单参数实验、16组分层联合样本和4组预设压力情景。r在对数空间分层，其余参数在线性空间分层；这些分布是抽样设计，不是现实参数分布。",
           "", "## 2 主要结果", "",
           "下表统计20组联合情景。保持比例是选定情景中的描述性比例，不是总体概率或置信水平；方向以±0.01个百分点为近零带，这是事先设置的数值容差，不是显著性检验。", "",
           "| 文化 | 基准标签份额变化（百分点） | 联合情景最小—最大变化 | 方向保持比例 | 城市变化相关系数中位数 |",
           "| --- | ---: | --- | ---: | ---: |"]
    for z in summary.to_dict("records"):
        lines.append(f"| {names[z['culture']]}（{z['culture']}） | {z['baseline_change_pp']:.4f} | {z['joint_min_change_pp']:.4f}—{z['joint_max_change_pp']:.4f} | {z['joint_direction_match_pct']:.0f}% | {z['median_city_change_rho']:.3f} |")
    lines += ["", "单参数扫描的最大文化份额影响排序（依赖本轮扫描宽度，不是方差贡献）：", ""]
    for p,v in strengths.items():lines.append(f"- {p}：最大偏离基准{v:.4f}个百分点。")
    unstable=summary[summary.joint_direction_match_pct<100].culture.tolist()
    stage=pd.read_csv(output/'state_annual.csv')
    theta_low=stage[(stage.run_id=='oat_theta_0')&(stage.elapsed==40)&(stage.state=='A')].fraction.iloc[0]*100
    theta_high=stage[(stage.run_id=='oat_theta_4')&(stage.elapsed==40)&(stage.state=='A')].fraction.iloc[0]*100
    growth=endpoint.pivot(index='run_id',columns='culture',values='growth_pct').reindex(columns=inp['cultures'])
    base_growth=base.loc[inp['cultures'],'growth_pct'].to_numpy()
    growth_rhos=[compare_metric(row,base_growth) for row in growth.to_numpy()]
    summary_signs=endpoint.pivot(index='run_id',columns='culture',values='delta_share_pp').reindex(columns=inp['cultures'])
    strict_matches=(np.sign(summary_signs.to_numpy())==np.sign(base.loc[inp['cultures'],'delta_share_pp'].to_numpy())).all()
    lines += ["", "方向出现不同判断的文化："+("、".join(names[c] for c in unstable) if unstable else "本轮20个联合情景中未出现")+"。不能据此声称未抽样区域也稳定。",
              "", "补充：以上方向分类包含近零带。华南的基准与所有联合情景实际均为负变化；区别在于部分情景从“近零”进入“小幅下降”，不是增长与下降之间反转。数学正负号稳定与实质变化稳定必须分开。",
              "", f"20个联合情景的11文化份额数学正负号全部与基准相同：{bool(strict_matches)}。文化携带人数相对增长率的跨文化排名与基准的Spearman相关系数最低为{min(growth_rhos):.4f}。城市变化向量的排名相关系数最低为{joint.city_change_spearman.min():.6f}；这是高度相似，不是每座城市数值完全一致。",
              "", "## 3 如何解读", "",
              "文化携带人数C按每个标签各计一次；I/A两标签分别进入两个文化。C/P是携带率，ΣC/P可超过1；C/ΣC是标签份额，求和为1。上述变化方向指标签份额，而不是人口计数；两者均保存在national_annual.csv。",
              "", "联合排名相关图只描述参数与输出的关联：20点不足以可靠分解主效应和交互作用，不报告p值、Sobol指数或重要性贡献百分比。r与μ的乘积结构也限制独立归因。",
              "", "城市分析比较同一城市、同一文化的变化向量及份额偏离。它不是Moran’s I检验、地理热点显著性或三类扩散模式的验证；本轮没有擅自新增距离分类或复用旧代码的预设类别。",
              "", "本轮固定人口流动及文化作用矩阵；尚未完成它们的敏感性实验。不能将六参数结果称为全部审稿要求已完成。",
              "", "参数敏感性依赖输出指标。γ对全国标签份额的扫描影响较大；θ虽然对标签份额影响小，却可能明显改变I与A的构成，需结合图8，不能称θ对整个模型不敏感。I→A保留双标签，本来就不会直接改变文化携带计数。",
              "", f"具体地，θ由0.02到0.18时，40年末A状态占总人口比例由{theta_low:.4f}%升至{theta_high:.4f}%。因此本轮支持“标签结构较稳定、状态构成对阶段速率敏感”，而不支持“所有输出对所有参数都不敏感”。",
              "", "另一个结构性限制：本轮保存结果中所有文化的全国携带人数逐年非减。迁移只移动标签，H→I及本地接触增加标签，I→A保留标签；因此携带人数非减部分来自模型通道定义，不能当成对现实文化必然增长的经验发现。份额下降表示增长慢于标签总量，而不是携带人数减少。",
              "", "## 4 实验引擎与原代码的差异", "",
              "使用独立的守恒实验引擎，原Code、原输入、原图和原稿未修改。本轮不是原投稿曲线复现。保留当前Python的N受A影响进入A、I→A不乘作用矩阵、城市顺序依次迁移和双标签第三文化随机选择等定义；它们与稿件仍有待统一之处。",
              "", "实现修正包括：城市ID按字符串读取；I/A双标签规范化并合并；保留全部初始状态人数；迁移文化状态抽样由有放回多项分布改为无放回多元超几何，避免抽走多于现存人数；目的地仍按原OD权重多项分配，并非逐条实现固定OD；每条文化转化通道用同一个取整数量做源减、目标增。",
              "", "这些修正改变了原实现的随机抽样分布及取整细节，虽然保留转化率公式和迁移期望分配机制，不能声称与原实现完全等价。城市人口以初态状态之和为准：元数据逐城人口与初态人数存在差异，详见audit.json。",
              "", "每组使用同一主种子和分离的城市/年份随机流，减少无关随机差异；不等于重复实验，也不能估计随机误差。所有年份均检验非负与全国人口精确守恒；无事后删异常情景、平滑曲线或添加扰动。",
              "", "## 5 图件与复用", "",
              "图件均输出300 dpi PNG和矢量PDF。阴影为所抽取情景的最小—最大包络，不是95%置信区间。",
              ""]
    for name,caption in [("01_oat_response","图1 六参数响应曲线"),("02_oat_effect","图2 单参数影响幅度"),
                         ("03_joint_departures","图3 联合情景偏离"),("04_joint_rank_associations","图4 描述性秩相关"),
                         ("05_scenario_envelopes","图5 时间演化情景包络"),("06_city_change_similarity","图6 城市变化模式相似性"),
                         ("07_robustness_summary","图7 方向与城市偏离汇总"),("08_state_composition_response","图8 H/I/A状态构成敏感性")]:
        lines += [f"### {caption}","",f"![{caption}](figures/{name}.png)",""]
    lines += ["输入哈希、实验引擎哈希、参数表与逐组种子均已保存。重新运行可断点续跑；输出目录中已有结果的来源指纹不一致时拒绝覆盖，需选择新输出目录。", ""]
    (output/"第一轮敏感性分析报告.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["audit","plan","run","analyze","all"])
    parser.add_argument("--data",type=Path,default=PROJECT/"Code"/"CSDS（项目代码）"/"data"/"source_data")
    parser.add_argument("--settings",type=Path,default=HERE/"design.json")
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--limit",type=int,default=None,help="Smoke test only; analyze requires all results")
    args=parser.parse_args()
    settings=read_settings(args.settings)
    args.output.mkdir(parents=True,exist_ok=True)
    inputs=load_inputs(args.data,settings)
    semantic_hash=simulation_digest(__file__)
    fingerprint=hashlib.sha256((semantic_hash+sha(args.settings)+json.dumps(inputs['audit']['source_sha256'],sort_keys=True)).encode()).hexdigest()
    prior_file=args.output/'provenance.json'
    archive=args.output/'executed_pipeline.py'
    reuse_provenance=None
    if prior_file.exists():
        prior=json.loads(prior_file.read_text(encoding='utf-8'))
        inputs_equal=(prior['input_sha256']==inputs['audit']['source_sha256'] and prior['settings']==settings and prior['numpy']==np.__version__)
        source_valid=archive.exists() and sha(archive)==prior['engine_sha256']
        if inputs_equal and source_valid and simulation_digest(archive)==semantic_hash:
            # Preserve the original first-round full-source fingerprint for its cached records.
            reuse_provenance=prior
            fingerprint=prior['fingerprint']
    if args.command in ["audit","plan","run","all"]:
        if reuse_provenance is None and prior_file.exists() and any((args.output/'runs').glob('*/metadata.json')):
            raise ValueError('Simulation inputs/design/engine differ: choose a new --output directory; existing results were not modified')
        write_json(args.output/"audit.json",inputs["audit"])
        inputs['excluded'].to_csv(args.output/"excluded_od.csv",index=False,encoding="utf-8-sig")
        plan=make_design(settings)
        plan.to_csv(args.output/"parameter_design.csv",index=False,encoding="utf-8-sig")
        if reuse_provenance is None:
            shutil.copyfile(__file__,archive)
            write_json(args.output/"provenance.json",dict(fingerprint=fingerprint,engine=ENGINE_VERSION,
                      engine_sha256=sha(__file__),simulation_semantic_sha256=semantic_hash,settings=settings,
                      input_sha256=inputs['audit']['source_sha256'],python=platform.python_version(),
                      numpy=np.__version__,pandas=pd.__version__,repeats=1))
        print(f"AUDIT: {inputs['audit']['cities']} cities; {len(plan)} cases; {len(inputs['excluded'])} OD records excluded",flush=True)
    if args.command in ["run","all"]:
        rows=plan.to_dict("records")[:args.limit]
        missing=[]
        done=0
        # Check caches before launching workers; cached runs do not need heavy imports.
        for row in rows:
            folder=args.output/'runs'/row['run_id']
            metadata=folder/'metadata.json'
            if metadata.exists() and (folder/'states.npz').exists():
                old=json.loads(metadata.read_text(encoding='utf-8'))
                if old.get('fingerprint')!=fingerprint or old.get('params')!={p:row[p] for p in KEYS}:
                    raise ValueError(f'Incompatible cached result: {folder}')
                done+=1
                print(f"[{done}/{len(rows)}] {row['run_id']}: cached; no simulation repeated",flush=True)
            else:
                missing.append(row)
        if missing and args.workers==1:
            for row in missing:
                ident,status,seconds=run_one(row,args.data,settings,args.output,fingerprint)
                done+=1
                print(f"[{done}/{len(rows)}] {ident}: {status} ({seconds:.1f}s)",flush=True)
        elif missing:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(run_one,row,args.data,settings,args.output,fingerprint) for row in missing]
                for future in as_completed(futures):
                    ident,status,seconds=future.result()
                    done+=1
                    print(f"[{done}/{len(rows)}] {ident}: {status} ({seconds:.1f}s)",flush=True)
    if args.command in ["analyze","all"]:
        analyze(args.output,args.data,settings)
        write_json(args.output/'analysis_provenance.json',dict(analysis_code_sha256=sha(__file__),
                   settings_sha256=sha(args.settings),simulation_repeated=False,
                   note='Analysis/figure refinement only; executed_pipeline.py archives original simulation source when present.'))
        print("Analysis, report and figures complete.",flush=True)


if __name__ == "__main__":
    main()
