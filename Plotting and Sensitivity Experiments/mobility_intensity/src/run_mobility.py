"""Run the three-scenario mobility intensity experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import pipeline as p

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/'data/inputs'
OUT = ROOT/'results'
original_design = p.design

def design(cfg):
    three = dict(cfg, seeds=cfg['seeds'][:3])
    templates = {}
    for r in original_design(three):
        if r['is_baseline'] or r['family']=='mobility':
            templates.setdefault(r['mobility_m'], r)
    rows = []
    for m in [.8, 1., 1.2]:
        for i, seed in enumerate(cfg['seeds'], 1):
            row = dict(templates[m], seed=seed, replicate=i)
            row['run_id'] = f"{row['config_id']}_rep{i:02}"
            rows.append(row)
    assert len(rows)==30 and len(set(cfg['seeds']))==10
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--config',type=Path,default=ROOT/'config/design.json')
    args=parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    p.design=design
    cfg=p.read(args.config)
    cfg['seeds']=list(range(20261012,20261022))
    cfg['experiment']='mobility_intensity_ten_replicates'
    cfg['population_basis']='Initial compartment totals define city population.'
    output=args.output
    manifest,rows=p.prepare(output,args.data,cfg)
    manifest['expected_configs']=3
    manifest['orchestrator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p.write(output/'manifest.json',manifest)
    shutil.copy2(__file__,output/'source'/'run_mobility.py')
    lock=output/'.running.lock'
    with lock.open('x',encoding='utf-8') as f: json.dump({'pid':os.getpid()},f)
    failures=[]
    try:
        progress=p.Progress(len(rows))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(p.simulate,r,str(output),manifest):r for r in rows}
            for future in as_completed(jobs):
                row=jobs[future]
                try:
                    run,status=future.result()
                    print(run,status,flush=True);progress.update(status)
                except Exception as exc:
                    failures.append({'run_id':row['run_id'],'error':repr(exc)})
                    print('FAILED',failures[-1],flush=True);progress.update('failed')
        p.write(output/'last_execution.json',{'submitted':len(rows),'failures':failures})
    finally:
        lock.unlink(missing_ok=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
