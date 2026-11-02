"""Full paired Moran scatter / spatial block refit panels for eleven cultures."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from make_residual_moran import ORDER,NAMES,GEO,weights
from read_points import read_points

ROOT=Path(os.environ.get('CSDS_SPATIAL_OUTPUT',Path(__file__).resolve().parent.parent/'results'))

def main():
    residuals=pd.read_csv(ROOT/'city_residuals.csv.gz',dtype={'CityID':str})
    points=residuals[residuals.seed==20261012]
    moran=pd.read_csv(ROOT/'residual_moran.csv')
    primary=moran[(moran.seed==20261012)&(moran.k==8)].set_index('culture')
    summaries=pd.read_csv(ROOT/'block_refit/curve_resampling_summary.csv')
    summaries=summaries[(summaries.seed==20261012)&(summaries.setting=='500km')]
    geo=read_points(GEO/'city_point_shp/city.shp')
    geo.loc[(geo.CityID=='650200')&(geo.NAME=='博尔塔拉蒙古自治州'),'CityID']='652700'
    geo=geo[geo.CityID!='0'].drop_duplicates('CityID').set_index('CityID')
    arrays=np.load(ROOT/'block_refit/all_resampled_curves.npz',allow_pickle=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(16.5,20))
    outer=fig.add_gridspec(6,2,left=.065,right=.985,top=.955,bottom=.065,wspace=.24,hspace=.55)
    colors=['#ba433b','#ac9ada','#8b9fdb','#a6b2d4']
    for i,c in enumerate(ORDER):
        cell=outer[i//2,i%2].subgridspec(1,2,width_ratios=[1,1.55],wspace=.38)
        left=fig.add_subplot(cell[0]);right=fig.add_subplot(cell[1])
        p=points[points.culture==c].sort_values('CityID')
        W=weights(geo.loc[p.CityID,['x','y']].to_numpy(),8)
        e=p.residual.to_numpy();z=e/np.sqrt(np.mean(e**2));lag=W@z
        quadrants=np.where((z>=0)&(lag>=0),0,np.where((z<0)&(lag<0),1,np.where((z<0)&(lag>=0),2,3)))
        for q in range(4):
            mask=quadrants==q
            left.scatter(z[mask],lag[mask],s=9,c=colors[q],alpha=.63,lw=0,rasterized=True)
        I=float(primary.loc[c,'I'])
        assert np.isclose(z@lag/(z@z),I)
        limit=max(abs(z))*1.1
        line=np.array([-limit,limit])
        left.plot(line,I*line,color='#374151',lw=1.3)
        left.axhline(0,color='#bdc3ce',lw=.6);left.axvline(0,color='#bdc3ce',lw=.6)
        left.set_xlim(-limit,limit)
        left.set_title('Moran scatter',fontsize=9,color='#596273',pad=5)
        left.set_xlabel('Standardized residual',fontsize=9)
        left.set_ylabel('Spatial lag',fontsize=9)
        values=moran[moran.culture==c].I
        left.text(.03,.97,f'I = {I:.3f}\nRange {values.min():.3f}–{values.max():.3f}',
            transform=left.transAxes,va='top',fontsize=8.5,
            bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1.5})
        f=summaries[summaries.culture==c].sort_values('distance_km')
        curves=arrays[f'curves_20261012_500km_{c}']
        # Show a fixed, evenly spaced subset only as visual context; bands use all 300 draws.
        for draw in range(0,300,10):
            right.plot(f.distance_km,curves[draw]*100,color='#c4cff4',lw=.45,alpha=.22,zorder=1)
        right.scatter(p.distance_km,p.growth_raw*100,s=6,c='#ac9ada',alpha=.25,lw=0,rasterized=True,zorder=2)
        right.fill_between(f.distance_km,f.q10*100,f.q90*100,where=f.display_band,
            color='#bfccfe',alpha=.65,lw=0,zorder=3)
        right.plot(f.distance_km,f.original*100,color='#ba433b',lw=1.5,zorder=4)
        right.plot(f.distance_km,np.where(f.display_band,f['median']*100,np.nan),
            color='#6577b3',lw=1.2,ls='--',zorder=5)
        right.axhline(0,color='#bdc3ce',lw=.6)
        right.set_title('Spatial block refits',fontsize=9,color='#596273',pad=5)
        right.set_xlabel('Distance to cultural center (km)',fontsize=9)
        right.set_ylabel('Relative share growth (%)',fontsize=9)
        # Keep all finite values visible, including unphysical polynomial-boundary artifacts.
        left.tick_params(labelsize=8);right.tick_params(labelsize=8)
        for ax in [left,right]:
            ax.spines['left'].set_color('#8d95a3');ax.spines['bottom'].set_color('#8d95a3')
            ax.spines['left'].set_linewidth(.7);ax.spines['bottom'].set_linewidth(.7)
            ax.tick_params(color='#8d95a3',length=3)
        right.ticklabel_format(axis='y',style='plain',useOffset=False)
        pos=left.get_position()
        fig.text(pos.x0,pos.y1+.026,f'{chr(97+i)}',fontsize=11,weight='bold',ha='left',color='#ba433b')
        fig.text(pos.x0+.017,pos.y1+.026,NAMES[c],fontsize=11,weight='bold',ha='left',color='#263140')
        from matplotlib.lines import Line2D as Rule
        fig.add_artist(Rule([pos.x0,right.get_position().x1],[pos.y1+.020]*2,transform=fig.transFigure,color='#e1e5ee',lw=.8))
    key=fig.add_subplot(outer[5,1]);key.axis('off')
    handles=[Line2D([0],[0],marker='o',ls='',color=colors[0],label='High–high residuals'),
        Line2D([0],[0],marker='o',ls='',color=colors[1],label='Low–low residuals'),
        Line2D([0],[0],marker='o',ls='',color=colors[2],label='Low–high residuals'),
        Line2D([0],[0],marker='o',ls='',color=colors[3],label='High–low residuals'),
        Line2D([0],[0],color='#ba433b',lw=1.5,label='Original fifth-degree fit'),
        Line2D([0],[0],color='#6577b3',ls='--',lw=1.2,label='Resampled median'),
        Patch(facecolor='#bfccfe',alpha=.65,label='10th–90th percentile range'),
        Line2D([0],[0],color='#c4cff4',lw=.6,label='30 illustrative refits / 300 draws')]
    key.legend(handles=handles,loc='upper left',frameon=False,ncol=2,fontsize=9,columnspacing=1.3,handlelength=2.2,labelspacing=.65)
    key.text(.03,.23,'Baseline m = 1.0; seed 20261012; 359 cities\nMoran: symmetrized 8-NN, row standardized\nI range: 3 seeds × 3 neighbourhood settings\nRefits: 500 km blocks; bands use all 300 draws\nBands require ≥240 valid predictions per distance',
        transform=key.transAxes,va='top',linespacing=1.5,fontsize=8.5,color='#566171')
    fig.text(.065,.018,'Spatial dependence diagnostics and descriptive-curve resampling assess different properties. Ranges are not confidence intervals.\nQuadrant colours are not local significance classes. Polynomial boundary artifacts are retained; these panels do not establish diffusion-category preservation.',
        fontsize=9,color='#566171')
    for ext in ['png','svg','pdf']:
        fig.savefig(ROOT/f'paired_moran_and_block_refits_optimized.{ext}',dpi=240 if ext=='png' else 150,facecolor='white')
    plt.close(fig)
    arrays.close()

if __name__=='__main__': main()
