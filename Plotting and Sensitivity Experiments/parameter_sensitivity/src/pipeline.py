"""Five-level CSDS factorial simulation pipeline."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import platform

# Limit BLAS threads per worker; parallelism is at independent-run level.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "engine.py"
spec = importlib.util.spec_from_file_location("csds_round1", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
np, pd = engine.np, engine.pd
KEYS = engine.KEYS
FILES = ("city_pop.csv", "city_culture_pop.csv", "city_pop_flow.csv", "cultural_strength.csv")
DEFAULT_DATA = HERE.parent / "data/inputs"
DEFAULT_OUTPUT = HERE.parent / "results"

class Progress:
    """TTY progress bar; compact milestone logs when output is redirected."""
    def __init__(self,total):
        self.total=total;self.done=0;self.completed=0;self.cached=0;self.failed=0
        self.started=time.perf_counter();self.tty=sys.stderr.isatty();self.last_bucket=-1
    def update(self,status):
        self.done+=1
        if status=="completed":self.completed+=1
        elif status=="cached":self.cached+=1
        else:self.failed+=1
        elapsed=time.perf_counter()-self.started
        rate=self.done/elapsed if elapsed else 0
        eta=(self.total-self.done)/rate if rate else float("inf")
        frac=self.done/self.total if self.total else 1
        width=30;filled=min(width,int(frac*width))
        bar=("█"*filled+"░"*(width-filled)) if self.tty else ("#"*filled+"-"*(width-filled))
        eta_text="--:--" if not np.isfinite(eta) else f"{int(eta)//60:02d}:{int(eta)%60:02d}"
        line=(f"[{bar}] {self.done}/{self.total} {frac:6.1%} | "
              f"new {self.completed} cached {self.cached} failed {self.failed} | "
              f"elapsed {int(elapsed)//60:02d}:{int(elapsed)%60:02d} ETA {eta_text}")
        if self.tty:
            print("\r"+line,end="\n" if self.done==self.total else "",file=sys.stderr,flush=True)
        else:
            bucket=int(frac*20)
            if bucket!=self.last_bucket or self.done==self.total:
                print("PROGRESS "+line,file=sys.stderr,flush=True);self.last_bucket=bucket

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def simulation_digest(path):
    """Hash simulation semantics, excluding progress/UI/reporting-only edits."""
    tree=ast.parse(Path(path).read_text(encoding="utf-8"))
    names={"design","checked_inputs","transformed","migrate_record","simulate"}
    nodes=[ast.dump(n,include_attributes=False) for n in tree.body
           if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,"name",None) in names]
    return hashlib.sha256("\n".join(nodes).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def design(cfg):
    if cfg["years"] < 1 or len(cfg["seeds"]) != 3 or len(set(cfg["seeds"])) != 3:
        raise ValueError("Positive years and exactly three distinct seeds required")
    if cfg["engine"] != "current_python_channels_v1":
        raise ValueError("Unsupported model; do not silently change channel semantics")
    for k in ("r", "gamma", "theta"):
        v = cfg["levels"][k]
        if len(v) != 5 or len(set(v)) != 5 or cfg["baseline"][k] not in v:
            raise ValueError("Five distinct levels including baseline required")
        if any(not np.isfinite(x) or x < 0 or x > 1 for x in v):
            raise ValueError("Invalid parameter level")
    configs = []
    for i, (r, g, t) in enumerate(itertools.product(*(cfg["levels"][p] for p in ("r","gamma","theta"))), 1):
        configs.append({**cfg["baseline"], "config_id":f"C{i:03}", "family":"core",
                        "r":r, "gamma":g, "theta":t, "strength_a":1., "mobility_m":1.})
    for name, family, a, m in (("S01","strength",0.,1.),("S02","strength",2.,1.),
                                ("M01","mobility",1.,.8),("M02","mobility",1.,1.2)):
        configs.append(dict(config_id=name,family=family,**cfg["baseline"],strength_a=a,mobility_m=m))
    rows = []
    for c in configs:
        c["is_baseline"] = c["family"] == "core" and all(c[k] == cfg["baseline"][k] for k in KEYS)
        for j, seed in enumerate(cfg["seeds"], 1):
            rows.append(dict(c, replicate=j, seed=int(seed), run_id=f'{c["config_id"]}_rep{j:02}'))
    assert len(rows) == 387 and sum(r["is_baseline"] for r in rows) == 3
    return rows

def checked_inputs(data, cfg):
    city = pd.read_csv(Path(data)/"city_pop.csv", dtype={"city_id":str})
    od = pd.read_csv(Path(data)/"city_pop_flow.csv",dtype={"o_city_id":str,"d_city_id":str})
    unknown = (set(od.o_city_id)|set(od.d_city_id))-set(city.city_id)
    if unknown - {"442100"}:
        raise ValueError(f"Unknown cities not authorized for exclusion: {unknown}")
    values = od.flow_pop.to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
        raise ValueError("OD counts must be finite nonnegative integers")
    return engine.load_inputs(data, cfg)

def prepare(output, data, cfg):
    output, data = Path(output), Path(data)
    rows = design(cfg)
    inp = checked_inputs(data, cfg)
    identity = dict(settings=cfg, inputs={f:digest(data/f) for f in FILES},
                    pipeline_simulation_sha256=simulation_digest(__file__),
                    engine_simulation_sha256=engine.simulation_digest(ENGINE_PATH))
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest = output/"manifest.json"
    if manifest.exists():
        old = read(manifest)
        if old["fingerprint"] != fingerprint:
            raise ValueError("Output belongs to different code/data/settings; choose a new output directory")
        for f in FILES:
            if digest(output/"inputs"/f) != identity["inputs"][f]:
                raise ValueError("Archived input corrupted")
        if read(output/"plan.json") != rows:
            raise ValueError("Plan changed")
        return old, rows
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output without manifest; use new output")
    output.mkdir(parents=True,exist_ok=True)
    (output/"inputs").mkdir()
    (output/"source").mkdir()
    for f in FILES:
        shutil.copy2(data/f,output/"inputs"/f)
    shutil.copy2(__file__,output/"source"/"pipeline.py")
    shutil.copy2(ENGINE_PATH,output/"source"/"engine_round1.py")
    shutil.copy2(HERE/"test_pipeline.py",output/"source"/"test_pipeline.py")
    write(output/"settings.json",cfg)
    write(output/"plan.json",rows)
    pd.DataFrame(rows).to_csv(output/"parameter_design.csv",index=False,encoding="utf-8-sig")
    write(output/"input_audit.json",inp["audit"])
    inp["excluded"].to_csv(output/"excluded_od.csv",index=False,encoding="utf-8-sig")
    old=dict(identity, fingerprint=fingerprint, pipeline_source_sha256=digest(__file__),
             engine_source_sha256=digest(ENGINE_PATH), expected_runs=len(rows),
             expected_configs=129, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
             environment=dict(python=sys.version, numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()))
    write(manifest,old)
    return old,rows

def transformed(inp,row):
    inp["k"] = inp["k"].copy()
    mask = ~np.eye(len(inp["cultures"]),dtype=bool)
    inp["k"][mask] = 1 + row["strength_a"]*(inp["k"][mask]-1)
    if (inp["k"]<0).any():
        raise ValueError("Invalid transformed strength")
    # Conservative sufficient probability checks, no silent clipping.
    if row["gamma"]*inp["k"].max()>1 or row["theta"]>1 or row["r"]*max(row[k] for k in ("muH","muI","muA"))*inp["k"].max()>1:
        raise ValueError("Transition probability bound violated")
    return inp

def migrate_record(x,inp,year,row):
    realized, origins, destinations, requested = [], [], [], []
    for c,(dest,amounts,original_total) in enumerate(inp["out"]):
        total = int(np.floor(original_total*row["mobility_m"]+.5))
        allocation = np.zeros((len(inp["genes"]),len(dest)),dtype=np.int64)
        if total:
            if total>x[c].sum():
                raise ValueError(f"Overdraw: city={inp['ids'][c]}, year={year}")
            rng=np.random.default_rng(np.random.SeedSequence([row["seed"],year,c,0]))
            counts=rng.multivariate_hypergeometric(x[c],total,method="marginals")
            allocation=rng.multinomial(counts,amounts/original_total)
            x[c]-=counts
            np.add.at(x,dest,allocation.T)
        realized.extend(allocation.sum(axis=0).tolist())
        origins.extend([c]*len(dest)); destinations.extend(dest.tolist()); requested.append(total)
    return np.asarray(realized,dtype=np.int64),np.asarray(origins),np.asarray(destinations),np.asarray(requested)

def valid_run(folder,row,fingerprint,deep=False):
    folder=Path(folder)
    meta=read(folder/"metadata.json")
    if meta["fingerprint"]!=fingerprint or meta["configuration"]!=row:
        raise ValueError(f"Provenance mismatch: {folder}")
    required={"states.npz","metrics.npz","migration.npz","national_annual.csv","stages_annual.csv"}
    if not required.issubset(meta["files"]):
        raise ValueError("Incomplete file manifest")
    for name,h in meta["files"].items():
        if digest(folder/name)!=h:
            raise ValueError(f"Corrupt output: {folder/name}")
    if deep:
        with np.load(folder/"states.npz",allow_pickle=False) as z:
            pop=z["counts"][0].sum()
            for name in ("counts","after_migration","after_reclassification"):
                v=z[name]
                if (v<0).any() or not np.all(v.sum(axis=(1,2))==pop):
                    raise ValueError("Conservation failure")
    return meta

def simulate(row,output,manifest):
    output=Path(output); cfg=manifest["settings"]
    final=output/"runs"/row["run_id"]
    final.parent.mkdir(parents=True,exist_ok=True)
    if final.exists():
        valid_run(final,row,manifest["fingerprint"])
        return row["run_id"],"cached"
    pending=Path(tempfile.mkdtemp(prefix=".pending-"+row["run_id"]+"-",dir=final.parent))
    start=time.perf_counter()
    try:
        inp=transformed(checked_inputs(output/"inputs",cfg),row)
        x=inp["x"].copy(); total=int(x.sum()); years=cfg["years"]
        history=np.empty((years+1,*x.shape),dtype=np.int64);history[0]=x
        migrated=np.empty((years,*x.shape),dtype=np.int64)
        classified=np.empty_like(migrated)
        choices=np.random.default_rng(row["seed"]+101).integers(0,2,size=(years,*x.shape),dtype=np.int8)
        flow=[]
        for y in range(1,years+1):
            f,orig,dest,requested=migrate_record(x,inp,y,row);flow.append(f);migrated[y-1]=x
            engine.status_update(x,inp);classified[y-1]=x
            engine.acculturate(x,inp,row,choices[y-1])
            if (x<0).any() or int(x.sum())!=total:
                raise ValueError(f"Invalid year {y}")
            history[y]=x
        labels=history@inp["carriers"]; city_pop=history.sum(axis=2)
        national=labels.sum(axis=1); stage=(history@inp["state"]).sum(axis=1)
        share=np.divide(national,national.sum(axis=1,keepdims=True),out=np.zeros_like(national,dtype=float),where=national.sum(axis=1,keepdims=True)>0)
        prevalence=np.divide(labels,city_pop[:,:,None],out=np.zeros_like(labels,dtype=float),where=city_pop[:,:,None]>0)
        np.savez_compressed(pending/"states.npz",counts=history,after_migration=migrated,
                            after_reclassification=classified,city_ids=np.asarray(inp["ids"]),
                            genes=np.asarray(inp["genes"]),years=np.arange(years+1)+cfg["start_year"])
        np.savez_compressed(pending/"metrics.npz",city_carriers=labels,city_population=city_pop,
                            city_prevalence=prevalence,city_prevalence_defined=city_pop>0,
                            national_carriers=national,national_label_share=share,
                            national_stages=stage,cultures=np.asarray(inp["cultures"]),
                            stage_names=np.asarray(engine.STATE_TYPES))
        np.savez_compressed(pending/"migration.npz",realized_od=np.asarray(flow),origin_index=orig,
                            destination_index=dest,requested_outflow=requested,k=inp["k"],
                            third_culture_choice_bits=choices)
        nr=[];sr=[]
        for y in range(years+1):
            for j,c in enumerate(inp["cultures"]):
                nr.append(dict(year=cfg["start_year"]+y,culture=c,carriers=int(national[y,j]),
                    prevalence=national[y,j]/total,label_share=share[y,j],
                    delta_share_pp=100*(share[y,j]-share[0,j])))
            for j,s in enumerate(engine.STATE_TYPES):
                sr.append(dict(year=cfg["start_year"]+y,state=s,count=int(stage[y,j]),fraction=stage[y,j]/total))
        pd.DataFrame(nr).to_csv(pending/"national_annual.csv",index=False)
        pd.DataFrame(sr).to_csv(pending/"stages_annual.csv",index=False)
        files={p.name:digest(p) for p in pending.iterdir() if p.is_file()}
        write(pending/"metadata.json",dict(configuration=row,fingerprint=manifest["fingerprint"],
            files=files,seconds=time.perf_counter()-start,total_population=total,years=years,
            status="complete",independent_realizations=1))
        pending.rename(final) # Publish only after all files and checksums are complete.
        return row["run_id"],"completed"
    except Exception as exc:
        write(pending/"failure.json",dict(run_id=row["run_id"],error=repr(exc)))
        raise

def status(output,manifest,rows,deep=False):
    complete=[];errors=[]
    for row in rows:
        folder=Path(output)/"runs"/row["run_id"]
        if folder.exists():
            try:
                valid_run(folder,row,manifest["fingerprint"],deep)
                complete.append(row)
            except Exception as exc:
                errors.append(dict(run_id=row["run_id"],error=str(exc)))
    return complete,dict(expected=len(rows),complete=len(complete),missing=len(rows)-len(complete)-len(errors),errors=errors)

def summarize(output,manifest,rows,partial=False):
    done,report=status(output,manifest,rows)
    if report["errors"] or (len(done)!=len(rows) and not partial):
        raise ValueError(f"Cannot produce full report: {report}")
    if not done:
        raise ValueError("No completed runs")
    frames=[]; stage_frames=[]
    for row in done:
        f=pd.read_csv(Path(output)/"runs"/row["run_id"]/"national_annual.csv")
        for k in ("run_id","config_id","replicate","seed","family"):
            f[k]=row[k]
        frames.append(f)
        s=pd.read_csv(Path(output)/"runs"/row["run_id"]/"stages_annual.csv")
        for k in ("run_id","config_id","replicate","seed","family"):
            s[k]=row[k]
        stage_frames.append(s)
    annual=pd.concat(frames,ignore_index=True)
    stages=pd.concat(stage_frames,ignore_index=True)
    tag="partial_" if len(done)!=len(rows) else ""
    annual.to_csv(Path(output)/(tag+"all_national_annual.csv.gz"),index=False,compression="gzip")
    stages.to_csv(Path(output)/(tag+"all_stages_annual.csv.gz"),index=False,compression="gzip")
    agg=annual.groupby(["config_id","year","culture"]).agg(
        n=("delta_share_pp","size"),mean_delta_pp=("delta_share_pp","mean"),
        sd_delta_pp=("delta_share_pp","std"),min_delta_pp=("delta_share_pp","min"),
        max_delta_pp=("delta_share_pp","max"))
    agg.to_csv(Path(output)/(tag+"replicate_summary.csv"))
    stage_agg=stages.groupby(["config_id","year","state"]).agg(
        n=("fraction","size"),mean_fraction=("fraction","mean"),
        sd_fraction=("fraction","std"),min_fraction=("fraction","min"),
        max_fraction=("fraction","max"))
    stage_agg.to_csv(Path(output)/(tag+"stage_replicate_summary.csv"))
    write(Path(output)/(tag+"summary_status.json"),report)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["plan","run","verify","status","summarize","smoke"])
    ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--data",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--settings",type=Path,default=HERE.parent/"config/design.json")
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--limit",type=int)
    ap.add_argument("--accept-current-engine",action="store_true")
    ap.add_argument("--partial",action="store_true")
    args=ap.parse_args();cfg=read(args.settings)
    if args.workers<1 or (args.limit is not None and args.limit<1):
        ap.error("workers and limit must be positive")
    if args.action=="smoke":
        cfg["years"]=2
        if args.output==DEFAULT_OUTPUT:
            args.output=HERE.parent/"results/smoke"
    if args.action=="run" and not args.accept_current_engine:
        ap.error("Use --accept-current-engine to run the configured experiment engine")
    manifest,rows=prepare(args.output,args.data,cfg)
    if args.action=="plan":
        print(json.dumps(dict(runs=len(rows),configs=129,baseline=[r["config_id"] for r in rows if r["is_baseline"]][0]),ensure_ascii=False));return
    if args.action in ("status","verify"):
        _,report=status(args.output,manifest,rows,deep=args.action=="verify")
        print(json.dumps(report,ensure_ascii=False))
        if report["errors"] or (args.action=="verify" and report["missing"] and not args.partial):
            raise SystemExit(1)
        return
    if args.action=="summarize":
        summarize(args.output,manifest,rows,args.partial);return
    if args.action=="smoke":
        selected=[next(r for r in rows if r["is_baseline"])]
    else:
        # Baseline replicates first, then remaining planned configurations.
        selected=sorted(rows,key=lambda r:not r["is_baseline"])
        if args.limit: selected=selected[:args.limit]
    lock=args.output/".running.lock"
    with lock.open("x",encoding="utf-8") as f:
        f.write(json.dumps(dict(pid=os.getpid(),started=time.time())))
    failures=[]
    progress=Progress(len(selected))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(simulate,r,str(args.output),manifest):r for r in selected}
            for job in as_completed(jobs):
                try:
                    result=job.result();print(*result,flush=True);progress.update(result[1])
                except Exception as exc:
                    failures.append(dict(run_id=jobs[job]["run_id"],error=repr(exc)))
                    print("FAILED",failures[-1],flush=True);progress.update("failed")
        write(args.output/"last_execution.json",dict(submitted=len(selected),failures=failures))
    finally:
        lock.unlink()
    if failures: raise SystemExit(1)

if __name__=="__main__":
    main()
