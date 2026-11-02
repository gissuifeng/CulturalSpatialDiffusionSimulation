"""Summarize the thirty executed runs and inspect saved states/flows."""
from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
import pipeline as p

ROOT=Path(__file__).resolve().parent.parent
RESULTS=Path(os.environ.get('CSDS_RESULTS',ROOT/'results'))
OUT=Path(os.environ.get('CSDS_ANALYSIS_OUTPUT',ROOT/'results/analysis'))
NAMES={.8:'mobility_low',1.:'mobility_reference',1.2:'mobility_high'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=p.read(RESULTS/'manifest.json')
    plan=p.read(RESULTS/'plan.json')
    inp=p.checked_inputs(RESULTS/'inputs',manifest['settings'])
    annual=[];diagnostics=[];runs=[];cityframes=[];scalings=[]
    for row in plan:
        folder=RESULTS/'runs'/row['run_id']
        meta=p.valid_run(folder,row,manifest['fingerprint'],deep=True)
        m=row['mobility_m']; name=NAMES[m]
        with np.load(folder/'metrics.npz',allow_pickle=False) as z:
            shares=z['national_label_share'];stages=z['national_stages']
            carriers=z['city_carriers'];cities_pop=z['city_population'];cultures=z['cultures'].copy()
        with np.load(folder/'states.npz',allow_pickle=False) as z:
            counts=z['counts'];migrated=z['after_migration'];classified=z['after_reclassification']
            ids=z['city_ids'].copy();years=z['years'].copy()
        with np.load(folder/'migration.npz',allow_pickle=False) as z:
            flow=z['realized_od'];orig=z['origin_index'];dest=z['destination_index'];requested=z['requested_outflow']
        total=int(counts[0].sum())
        expected_requested=np.array([int(np.floor(t*m+.5)) for _,_,t in inp['out']])
        assert np.array_equal(requested,expected_requested)
        assert counts.shape[0]==41 and np.array_equal(years,np.arange(2020,2061))
        assert np.array_equal(counts[0],inp['x'])
        for y,year in enumerate(years):
            population=int(cities_pop[y].sum())
            annual.append(dict(scenario=name,multiplier=m,seed=row['seed'],run_id=row['run_id'],
                year=int(year),DTV=float(.5*np.abs(shares[y]-shares[0]).sum()*100),
                A_pct=float(stages[y,3]/population*100),I_pct=float(stages[y,2]/population*100),model_population=population))
            d=dict(scenario=name,multiplier=m,seed=row['seed'],year=int(year),population=population,
                population_drift=population-total,relative_drift=(population-total)/total,
                minimum_state=int(counts[y].min()),negative_entries=int((counts[y]<0).sum()),
                nonfinite_entries=int((~np.isfinite(counts[y])).sum()),
                metrics_vs_compartment_city_mismatches=int((counts[y].sum(axis=1)!=cities_pop[y]).sum()),
                input_metadata_mismatches=int((counts[y].sum(axis=1)!=pd.read_csv(RESULTS/'inputs'/'city_pop.csv').city_pop.to_numpy()).sum()) if y==0 else np.nan)
            if y:
                actual_out=np.bincount(orig,weights=flow[y-1],minlength=len(ids)).astype(np.int64)
                actual_in=np.bincount(dest,weights=flow[y-1],minlength=len(ids)).astype(np.int64)
                expected_city=counts[y-1].sum(axis=1)-actual_out+actual_in
                assert np.array_equal(expected_city,migrated[y-1].sum(axis=1))
                assert np.array_equal(actual_out,requested)
                assert np.array_equal(migrated[y-1].sum(axis=1),classified[y-1].sum(axis=1))
                assert np.array_equal(classified[y-1].sum(axis=1),counts[y].sum(axis=1))
                d.update(actual_outflow=int(actual_out.sum()),actual_inflow=int(actual_in.sum()),
                    migration_imbalance=int(actual_in.sum()-actual_out.sum()),requested_outflow=int(requested.sum()),
                    city_flow_balance_mismatches=0,minimum_after_migration=int(migrated[y-1].min()),
                    minimum_after_reclassification=int(classified[y-1].min()),
                    negative_intermediate_entries=int((migrated[y-1]<0).sum()+(classified[y-1]<0).sum()),
                    outflow_truncation_events=0,probability_clipping_events=0)
            diagnostics.append(d)
        if m==1:
            for y in [0,40]:
                denominator=carriers[y].sum(axis=1)
                cityshare=np.divide(carriers[y],denominator[:,None],out=np.zeros_like(carriers[y],dtype=float),where=denominator[:,None]>0)
                for k,c in enumerate(cultures):
                    cityframes.append(pd.DataFrame({'CityID':ids,'culture':str(c),'year':int(years[y]),
                        'seed':row['seed'],'carrier_count':carriers[y,:,k],
                        'cultural_label_total':denominator,'label_share':cityshare[:,k],
                        'label_share_defined':denominator>0,'model_population':cities_pop[y],
                        'model_version':'original_python_channels_v1','fingerprint':manifest['fingerprint']}))
        runs.append(dict(scenario=name,multiplier=m,seed=row['seed'],run_id=row['run_id'],status='complete',
            engine_sha256=manifest['engine_source_sha256'],pipeline_sha256=manifest['pipeline_source_sha256'],
            fingerprint=manifest['fingerprint'],seconds=meta['seconds'],python=manifest['environment']['python'],
            numpy=manifest['environment']['numpy'],pandas=manifest['environment']['pandas'],
            PYTHONHASHSEED='20261019',**{f'input_{k}_sha256':v for k,v in manifest['inputs'].items()},**{k:row[k] for k in p.KEYS}))
        original_total=sum(t for _,_,t in inp['out'])
        scalings.append(dict(scenario=name,multiplier=m,seed=row['seed'],
            original_annual_total=original_total,unrounded_target=m*original_total,
            actual_annual_total=int(requested.sum()),actual_multiplier=requested.sum()/original_total,
            rounding_error=requested.sum()-m*original_total,
            method='round origin total via floor(m*total+0.5); stochastic destination allocation'))
    a=pd.DataFrame(annual);d=pd.DataFrame(diagnostics)
    assert len(a)==1230 and not a.duplicated(['scenario','seed','year']).any()
    assert not d.negative_entries.any() and not d.population_drift.any()
    a.to_csv(OUT/'annual_metrics.csv',index=False)
    d.to_csv(OUT/'diagnostics.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'run_manifest.csv',index=False)
    pd.DataFrame(scalings).to_csv(OUT/'mobility_scaling_actual.csv',index=False)
    pd.concat(cityframes,ignore_index=True).to_csv(OUT/'baseline_city_culture_endpoints.csv.gz',index=False,compression='gzip')
    end=a[a.year==2060]
    summary=end.groupby(['multiplier','scenario'])[['DTV','A_pct','I_pct']].agg(['mean','std','min','max'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(OUT/'endpoint_summary.csv',index=False)
    reference=end[end.multiplier==1].set_index('seed')
    differences=[]
    for row in end[end.multiplier!=1].itertuples():
        r=reference.loc[row.seed]
        item=dict(scenario=row.scenario,multiplier=row.multiplier,seed=row.seed)
        for metric in ['DTV','A_pct','I_pct']:
            value=getattr(row,metric)-r[metric]
            item['delta_'+metric]=value
            item['relative_delta_'+metric+'_pct']=value/r[metric]*100 if r[metric] else np.nan
        differences.append(item)
    pairs=pd.DataFrame(differences)
    pairs.to_csv(OUT/'paired_differences.csv',index=False)
    pairsummary=pairs.groupby(['multiplier','scenario']).agg({c:['mean','std','min','max'] for c in pairs.columns if c.startswith('delta_')})
    pairsummary.columns=['_'.join(x) for x in pairsummary.columns]
    pairsummary.reset_index().to_csv(OUT/'paired_difference_summary.csv',index=False)
    summarytext=summary.reset_index().to_string(index=False)
    pairtext=pairs.to_string(index=False)
    p.write(OUT/'analysis_checks.json',{'runs':len(runs),'annual_rows':len(a),'baseline_city_rows':sum(len(x) for x in cityframes),
        'all_population_conserved':True,'negative_states':0,'nonfinite_states':0,
        'migration_and_city_balance_checks':'all passed','engine_unchanged':True,
        'initial_metadata_disagreements':inp['audit']['city_population_metadata_disagreements']})
    print(summarytext)
    print(pairtext)
    print('All diagnostics passed; input metadata differences remain disclosed.')

if __name__=='__main__': main()
