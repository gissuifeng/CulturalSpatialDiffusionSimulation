"""Residual Moran scatter diagnostics, using actual 2020–2060 baseline outputs."""
from pathlib import Path
import hashlib
import json
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from read_points import read_points

ROOT=Path(os.environ.get('CSDS_SPATIAL_OUTPUT',Path(__file__).resolve().parent.parent/'results'))
BASE=ROOT.parent
GEO=Path(os.environ['CSDS_GEOGRAPHY'])
SOURCE=Path(os.environ['CSDS_ENDPOINTS'])
ORDER=['L','J','E','D','K','M','G','C','F','B','X']
NAMES=dict(zip(['J','L','G','B','E','X','F','D','C','K','M'],[
    'Sichuan–Chongqing','Jiangnan','Yunnan–Guizhou','Inner Mongolian','North China',
    'Northeast','Tibetan','Loess Plateau','Xinjiang','Jingchu','South China']))

def weights(points,k):
    distances=cdist(points,points)
    np.fill_diagonal(distances,np.inf)
    # Input cities are sorted by ID; stable sort resolves equal-distance ties.
    neighbours=np.argsort(distances,axis=1,kind='stable')[:,:k]
    adjacency=np.zeros_like(distances,dtype=bool)
    adjacency[np.arange(len(points))[:,None],neighbours]=True
    adjacency=adjacency|adjacency.T
    return adjacency/adjacency.sum(axis=1,keepdims=True)

def diagnostic(e,Q,W):
    n=len(e);rank=Q.shape[1];q=n-rank;s0=W.sum()
    S=(W+W.T)/2
    M=np.eye(n)-Q@Q.T
    B=M@S@M
    tr=np.trace(B)
    expectation=(n/s0)*tr/q
    variance=2*(n/s0)**2*(np.sum(B*B)-tr*tr/q)/(q*(q+2))
    I=(n/s0)*float(e@W@e/(e@e))
    # Independent low-rank trace expression, equivalent to the regression correction.
    A=Q.T@S@Q
    trB2=np.sum((S@Q)**2)
    alternative=(n/s0)**2/(q*(q+2))*(2*np.sum(S*S)+2*np.trace(A@A)-4*trB2-2*np.trace(A)**2/q)
    assert np.isclose(variance,alternative,rtol=1e-9,atol=1e-12)
    z=(I-expectation)/np.sqrt(variance)
    return I,expectation,variance,z,float(2*norm.sf(abs(z)))

def bh(p):
    p=np.asarray(p);idx=np.argsort(p);out=np.empty(len(p))
    out[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return out

def main():
    cities=read_points(GEO/'city_point_shp/city.shp')[['CityID','NAME','x','y']]
    corrected=(cities.CityID=='650200') & (cities.NAME=='博尔塔拉蒙古自治州')
    cities.loc[corrected,'CityID']='652700'
    cities=cities[cities.CityID!='0'].copy()
    centers=read_points(GEO/'province_point_shp/province_point.shp')[['CDT_type','x','y']].set_index('CDT_type')
    # Match the original plotting reader: one center attribute contains a concatenated string.
    centers.index=centers.index.astype(str).str[0]
    assert not centers.index.has_duplicates and set(centers.index)==set(ORDER)
    duplicate=cities[cities.CityID.duplicated(keep=False)]
    # Only identical coordinates may be deduplicated; conflicting duplicates require a decision.
    for _,g in duplicate.groupby('CityID'):
        if len(g[['x','y']].drop_duplicates())!=1: raise ValueError('Conflicting city coordinates')
    cities=cities.drop_duplicates('CityID').sort_values('CityID')
    cp=GEO/'city_point_shp/city.prj';rp=GEO/'province_point_shp/province_point.prj'
    if cp.read_text()!=rp.read_text(): raise ValueError('Coordinate systems differ')
    raw=pd.read_csv(SOURCE,dtype={'CityID':str})
    raw=raw[raw.seed.isin([20261012,20261013,20261014])]
    assert not raw.duplicated(['CityID','culture','seed','year']).any()
    allids=set(raw.CityID); missing=sorted(allids-set(cities.CityID))
    rows=[];details=[];cache={};exclusions=[]
    for seed in [20261012,20261013,20261014]:
        sample=raw[raw.seed==seed]
        for c in ORDER:
            g=sample[sample.culture==c].pivot(index='CityID',columns='year',values='label_share').reset_index()
            zero=g[2020]<=0
            exclusions.append({'seed':seed,'culture':c,'zero_initial_share':int(zero.sum()),'missing_coordinates':len(missing)})
            g=g.loc[~zero].merge(cities,on='CityID',how='inner',validate='one_to_one').sort_values('CityID')
            y=(g[2060].to_numpy()-g[2020].to_numpy())/g[2020].to_numpy()
            points=g[['x','y']].to_numpy()
            distance=np.linalg.norm(points-centers.loc[c,['x','y']].to_numpy(dtype=float),axis=1)/1000
            scaled=(distance-distance.mean())/distance.std()
            X=np.polynomial.polynomial.polyvander(scaled,5)
            Q,R=np.linalg.qr(X,mode='reduced')
            assert np.linalg.matrix_rank(X)==6
            fitted=Q@(Q.T@y);e=y-fitted
            assert np.max(np.abs(X.T@e))<1e-6
            standard=e/np.sqrt(np.mean(e*e))
            for k in [4,8,12]:
                W=weights(points,k)
                I,expected,var,z,p=diagnostic(e,Q,W)
                rows.append(dict(seed=seed,culture=c,name=NAMES[c],k=k,n=len(g),degree=5,
                    I=I,expected_I_regression=expected,variance_I_regression=var,z_normal=z,p_normal=p,
                    condition_number=np.linalg.cond(X)))
                if seed==20261012 and k==8: cache[c]=(standard,W@standard,I,len(g))
            details.append(pd.DataFrame({'CityID':g.CityID,'culture':c,'seed':seed,'distance_km':distance,
                'growth_raw':y,'fitted_growth':fitted,'residual':e}))
    stats=pd.DataFrame(rows)
    stats['q_BH_normal']=stats.groupby(['seed','k']).p_normal.transform(lambda x:bh(x.to_numpy()))
    stats.to_csv(ROOT/'residual_moran.csv',index=False)
    pd.concat(details).to_csv(ROOT/'city_residuals.csv.gz',index=False,compression='gzip')
    pd.DataFrame(exclusions).to_csv(ROOT/'exclusions.csv',index=False)
    chosen=stats[(stats.seed==20261012)&(stats.k==8)].set_index('culture')
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(3,4,figsize=(13,9.2))
    fig.subplots_adjust(left=.065,right=.985,bottom=.10,top=.94,wspace=.32,hspace=.43)
    colors=['#ba433b','#ac9ada','#8b9fdb','#a6b2d4']
    for i,c in enumerate(ORDER):
        ax=axes.flat[i];x,lag,I,n=cache[c]
        quadrant=np.where((x>=0)&(lag>=0),0,np.where((x<0)&(lag<0),1,np.where((x<0)&(lag>=0),2,3)))
        for q in range(4):
            m=quadrant==q;ax.scatter(x[m],lag[m],s=14,color=colors[q],alpha=.65,linewidths=0,rasterized=True)
        ax.axhline(0,color='#bcc1ca',lw=.7);ax.axvline(0,color='#bcc1ca',lw=.7)
        lim=max(1.1,np.max(np.abs(x))*1.08)
        xx=np.array([-lim,lim]);ax.plot(xx,I*xx,color='#343b49',lw=1.3)
        ax.set_xlim(-lim,lim)
        ax.set_title(f'{chr(97+i)}  {NAMES[c]}',loc='left',fontsize=10.2,pad=7)
        row=chosen.loc[c];qtext='< 0.001' if row.q_BH_normal<.001 else f'= {row.q_BH_normal:.3f}'
        ax.text(.04,.96,f"I = {I:.3f}   |   n = {n}\nBH q (normal) {qtext}",transform=ax.transAxes,va='top',fontsize=8,
                bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':2})
        ax.tick_params(labelsize=8)
        if i%4==0: ax.set_ylabel('Spatial lag of standardized residual')
        if i>=8: ax.set_xlabel('Standardized distance-fit residual')
    ax=axes.flat[11];ax.axis('off')
    handles=[Line2D([0],[0],marker='o',ls='',color=colors[i],label=label) for i,label in enumerate(['High–high','Low–low','Low–high','High–low'])]
    ax.legend(handles=handles,loc='upper left',frameon=False,title='Residual / neighbour residual',fontsize=9,title_fontsize=9)
    count=int((chosen.q_BH_normal<.05).sum())
    ax.text(.04,.37,f'Baseline: m = 1.0\nSeed: 20261012\nFifth-degree distance fit\nSymmetrized 8-nearest neighbours\nRow-standardized spatial weights\n{count}/11 diagnostics: BH q < 0.05',transform=ax.transAxes,va='top',fontsize=9,linespacing=1.65,color='#485062')
    fig.text(.065,.035,'2020–2060 cultural-label share growth; regression-adjusted normal-reference diagnostics.\nBH correction across 11 cultures. Diagnostic evidence does not remove spatial dependence.',fontsize=9,color='#566171')
    for ext in ['png','svg','pdf']: fig.savefig(ROOT/f'residual_moran_diagnostic.{ext}',dpi=320,facecolor='white')
    plt.close(fig)
    audit={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'seeds':[20261012,20261013,20261014],
        'display_seed':20261012,'display_k':8,'degree':5,'missing_coordinate_city_ids':missing,
        'identical_duplicate_coordinate_records':len(duplicate),'zero_share_exclusions':exclusions,
        'coordinate_id_correction':{'name':'博尔塔拉蒙古自治州','old':'650200','new':'652700','rows':int(corrected.sum()),'scope':'in-memory geometry join only; source shapefile unchanged','source':'https://xxgk.mot.gov.cn/2020/jigou/syj/202006/P020240417564270916061.pdf'},
        'method':'Gaussian homoskedastic regression-null moment correction using projection matrix; two-sided normal approximation',
        'p_value_limit':'Not a robust distribution-free test; skewness and heteroskedasticity may affect reference p-values.',
        'geometry':'Original CGCS2000 Lambert city points and supplied culture centers; Euclidean km; no rescaling or jitter',
        'geometry_sha256':{str(p.relative_to(GEO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder,stem in [('city_point_shp','city'),('province_point_shp','province_point')]
            for ext in ['shp','dbf','prj'] for p in [GEO/folder/f'{stem}.{ext}']},
        'center_label_rule':'First character of CDT_type, matching the existing plotting reader; G attribute contains concatenated values. Geometry unchanged.',
        'diagnostic_source':'https://pysal.org/spreg/_modules/spreg/diagnostics_sp.html'}
    (ROOT/'analysis_manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(chosen[['name','n','I','p_normal','q_BH_normal']].to_string())
    print('Missing coordinate IDs:',missing)

if __name__=='__main__': main()
