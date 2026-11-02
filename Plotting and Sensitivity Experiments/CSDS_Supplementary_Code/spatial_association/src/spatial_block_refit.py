"""Refit descriptive distance curves under spatially grouped resampling."""
from pathlib import Path
import hashlib
import json
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from read_points import read_points
from make_residual_moran import GEO,ORDER,NAMES

ROOT=Path(os.environ.get('CSDS_SPATIAL_OUTPUT',Path(__file__).resolve().parent.parent/'results'))
OUT=ROOT/'block_refit'
SEEDS=[20261012,20261013,20261014]
SETTINGS=[('250km',250,0),('500km',500,0),('750km',750,0),('500km_shift',500,250)]
B=300

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(ROOT/'city_residuals.csv.gz',dtype={'CityID':str})
    geo=read_points(GEO/'city_point_shp/city.shp')[['CityID','NAME','x','y']]
    geo.loc[(geo.CityID=='650200')&(geo.NAME=='博尔塔拉蒙古自治州'),'CityID']='652700'
    geo=geo[geo.CityID!='0'].drop_duplicates('CityID').set_index('CityID')
    summaries=[];diagnostics=[];drawmetrics=[];arrays={};originals={};blockrecords=[]
    for seed in SEEDS:
        per_seed=raw[raw.seed==seed]
        ids=sorted(per_seed.CityID.unique())
        xy=geo.loc[ids,['x','y']].to_numpy(dtype=float)/1000
        for setting_i,(setting,size,offset) in enumerate(SETTINGS):
            blockxy=np.floor((xy-offset)/size).astype(int)
            labels,which=np.unique(blockxy,axis=0,return_inverse=True)
            nblocks=len(labels)
            rng=np.random.default_rng(np.random.SeedSequence([20261019,seed,setting_i]))
            draws=rng.integers(0,nblocks,size=(B,nblocks))
            counts=np.array([np.bincount(draw,minlength=nblocks) for draw in draws])
            cityweights=counts[:,which]
            arrays[f'block_multiplicity_{seed}_{setting}']=counts
            for ident,b in zip(ids,blockxy):
                blockrecords.append(dict(seed=seed,setting=setting,CityID=ident,block_x=b[0],block_y=b[1]))
            for c in ORDER:
                f=per_seed[per_seed.culture==c].set_index('CityID').reindex(ids)
                if f[['distance_km','growth_raw']].isna().any().any():
                    raise ValueError('Culture-specific sample differs; requires explicit common-block filtering')
                x=f.distance_km.to_numpy();y=f.growth_raw.to_numpy()
                mu=x.mean();sd=x.std()
                X=np.polynomial.polynomial.polyvander((x-mu)/sd,5)
                grid=np.linspace(x.min(),x.max(),201)
                Xgrid=np.polynomial.polynomial.polyvander((grid-mu)/sd,5)
                coef=np.linalg.lstsq(X,y,rcond=None)[0]
                original=Xgrid@coef
                assert np.allclose(X@coef,f.fitted_growth.to_numpy(),rtol=1e-8,atol=1e-10)
                originals[(seed,c)]=(x,y,grid,original)
                predicted=np.full((B,len(grid)),np.nan)
                failure=rankfail=0;conditions=[];coverage=[]
                for repeat,w in enumerate(cityweights):
                    present=w>0
                    sample_n=int(w.sum());unique_n=int(present.sum())
                    if unique_n<6 or len(np.unique(x[present]))<6:
                        rankfail+=1
                        drawmetrics.append(dict(seed=seed,culture=c,setting=setting,repeat=repeat,status='insufficient_distances',sample_n=sample_n,unique_n=unique_n))
                        continue
                    sw=np.sqrt(w[present])
                    matrix=X[present]*sw[:,None]
                    beta,_,rank,singular=np.linalg.lstsq(matrix,y[present]*sw,rcond=None)
                    condition=float(singular[0]/singular[-1])
                    conditions.append(condition)
                    if rank<6 or not np.isfinite(beta).all():
                        rankfail+=1
                        drawmetrics.append(dict(seed=seed,culture=c,setting=setting,repeat=repeat,status='rank_deficient',sample_n=sample_n,unique_n=unique_n,condition=condition))
                        continue
                    domain=(grid>=x[present].min())&(grid<=x[present].max())
                    curve=Xgrid@beta
                    if not np.isfinite(curve[domain]).all():
                        failure+=1
                        drawmetrics.append(dict(seed=seed,culture=c,setting=setting,repeat=repeat,status='nonfinite',sample_n=sample_n,unique_n=unique_n,condition=condition))
                        continue
                    predicted[repeat,domain]=curve[domain]
                    coverage.append(domain.mean())
                    error=curve[domain]-original[domain]
                    drawmetrics.append(dict(seed=seed,culture=c,setting=setting,repeat=repeat,status='success',
                        sample_n=sample_n,unique_n=unique_n,condition=condition,
                        grid_coverage_fraction=domain.mean(),curve_rmse=float(np.sqrt(np.mean(error**2))),
                        sample_min_distance=float(x[present].min()),sample_max_distance=float(x[present].max())))
                valid=np.isfinite(predicted).sum(axis=0)
                # Original extreme observations always appear in some bootstrap samples.
                median=np.nanmedian(predicted,axis=0)
                lo,hi=np.nanquantile(predicted,[.1,.9],axis=0)
                show=valid>=int(np.ceil(.8*B))
                arrays[f'curves_{seed}_{setting}_{c}']=predicted
                arrays[f'grid_{seed}_{c}']=grid
                arrays[f'original_{seed}_{c}']=original
                summaries.append(pd.DataFrame(dict(seed=seed,culture=c,name=NAMES[c],setting=setting,
                    block_km=size,offset_km=offset,distance_km=grid,original=original,
                    median=median,q10=lo,q90=hi,n_valid=valid,display_band=show)))
                diagnostics.append(dict(seed=seed,culture=c,setting=setting,block_km=size,offset_km=offset,
                    n_cities=len(ids),n_blocks=nblocks,attempts=B,successes=B-rankfail-failure,
                    rank_failures=rankfail,other_failures=failure,
                    displayed_grid_fraction=show.mean(),max_condition=max(conditions) if conditions else np.nan,
                    mean_covered_grid_fraction=np.mean(coverage) if coverage else np.nan))
            print(seed,setting,'completed',flush=True)
    summary=pd.concat(summaries,ignore_index=True)
    summary.to_csv(OUT/'curve_resampling_summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'resampling_diagnostics.csv',index=False)
    pd.DataFrame(drawmetrics).to_csv(OUT/'individual_refit_diagnostics.csv.gz',index=False,compression='gzip')
    pd.DataFrame(blockrecords).to_csv(OUT/'city_block_assignments.csv',index=False)
    np.savez_compressed(OUT/'all_resampled_curves.npz',**arrays)
    display=summary[(summary.seed==20261012)&(summary.setting=='500km')]
    plot(display,originals,'spatial_block_refit_500km')
    for setting,_,_ in SETTINGS:
        if setting!='500km': plot(summary[(summary.seed==20261012)&(summary.setting==setting)],originals,'spatial_block_refit_'+setting)
    # Quantify common-support differences without defining an arbitrary pass threshold.
    stability=[]
    for (seed,c,setting),f in summary.groupby(['seed','culture','setting']):
        f=f[f.display_band]
        if len(f)==0: continue
        stability.append(dict(seed=seed,culture=c,setting=setting,
            median_band_width=float((f.q90-f.q10).median()),
            median_absolute_curve_shift=float((f['median']-f.original).abs().median()),
            mean_absolute_curve_shift=float((f['median']-f.original).abs().mean()),
            max_absolute_curve_shift=float((f['median']-f.original).abs().max()),
            evaluable_distance_min=float(f.distance_km.min()),evaluable_distance_max=float(f.distance_km.max())))
    pd.DataFrame(stability).to_csv(OUT/'curve_stability_descriptors.csv',index=False)
    manifest=dict(simulation_rerun=False,seeds=SEEDS,display_seed=20261012,degree=5,repeats=B,
        attempted_fits=3*4*11*B,grid_points=201,minimum_display_count=240,
        settings=[dict(name=n,block_km=s,offset_x_km=o,offset_y_km=o) for n,s,o in SETTINGS],
        resampling='sample nonempty projected square blocks with replacement; include all cities with block multiplicity; identical draws across cultures within each seed/setting',
        fit='weighted least squares equivalent to expanding repeated blocks; original distance basis fixed within each culture/seed',
        prediction='only inside the sampled distance minimum and maximum; no extrapolated predictions retained',
        range='10th–90th percentiles; exploratory resampling variation, not calibrated confidence bands',
        classification_performed=False,source_residuals_sha256=hashlib.sha256((ROOT/'city_residuals.csv.gz').read_bytes()).hexdigest(),
        coordinate_audit='See parent analysis_manifest.json for geometry corrections, projection and excluded cities')
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(diagnostics).groupby('setting')[['attempts','successes','rank_failures','other_failures']].sum().to_string())

def plot(data,originals,filename):
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(3,4,figsize=(13,9.2))
    fig.subplots_adjust(left=.065,right=.985,bottom=.10,top=.95,wspace=.35,hspace=.43)
    for i,c in enumerate(ORDER):
        ax=axes.flat[i];f=data[data.culture==c].sort_values('distance_km')
        x,y,grid,original=originals[(20261012,c)]
        ax.scatter(x,y*100,s=8,c='#ac9ada',alpha=.28,linewidths=0,rasterized=True)
        ax.fill_between(f.distance_km,f.q10*100,f.q90*100,where=f.display_band,color='#bfccfe',alpha=.65,lw=0)
        ax.plot(grid,original*100,color='#ba433b',lw=1.7,label='Original fit')
        ax.plot(f.distance_km,np.where(f.display_band,f['median']*100,np.nan),color='#6577b3',lw=1.3,ls='--')
        ax.axhline(0,color='#b8bec7',lw=.6)
        ax.set_title(f'{chr(97+i)}  {NAMES[c]}',loc='left',fontsize=10)
        ax.tick_params(labelsize=8)
        if i%4==0:ax.set_ylabel('Relative cultural-share growth (%)')
        if i>=8:ax.set_xlabel('Distance to cultural center (km)')
    ax=axes.flat[11];ax.axis('off')
    handles=[Line2D([0],[0],marker='o',ls='',color='#ac9ada',alpha=.6,label='City result'),
        Line2D([0],[0],color='#ba433b',lw=1.7,label='Original fifth-degree fit'),
        Line2D([0],[0],color='#6577b3',lw=1.3,ls='--',label='Resampled median'),
        Patch(facecolor='#bfccfe',alpha=.65,label='10th–90th percentile')]
    ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=9)
    size=int(data.block_km.iloc[0]);offset=int(data.offset_km.iloc[0])
    ax.text(.04,.42,f'Baseline m = 1.0; seed 20261012\n{size} km square blocks\nGrid offset: ({offset}, {offset}) km\n300 block resamples per curve\n359 matched cities\nBands require ≥240 valid fits',transform=ax.transAxes,va='top',linespacing=1.65,fontsize=9,color='#4e596c')
    fig.text(.065,.035,'Spatial block resampling of 2020–2060 descriptive distance curves.\nExploratory variation ranges, not confidence bands; each prediction is restricted to its resampled distance range.',fontsize=9,color='#566171')
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{filename}.{ext}',dpi=320,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
