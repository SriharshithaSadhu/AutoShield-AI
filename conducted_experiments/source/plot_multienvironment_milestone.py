"""Static research figures from completed development results, no refitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
O=R/'results/multienvironment_2026-11-21'

def main():
    rows=[]
    for folder in ['ranking_objective','matched_lightgbm_objective_reference']:
        rows.extend(json.loads((O/folder/'results.json').read_text())['results'])
    methods=['lightgbm_logloss_balanced','logloss_unweighted','logloss_balanced',
             'pairwise_unweighted','pairwise_environment_balanced']
    labels=['LightGBM balanced','XGB log-loss','XGB balanced','Pairwise','Pairwise env-balanced']
    colors=plt.cm.tab10(np.arange(5))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,rep,title in zip(axes,['raw46','harmonized14'],['46 exporter features','14 harmonized physical features']):
        for method,label,color in zip(methods,labels,colors):
            rr=[r for r in rows if r['method']==method and r['representation']==rep
                and r['environment']=='ton_stress' and r['policy']=='pooled_budget']
            for mix,marker in [('two_source','o'),('three_source','^')]:
                ss=[r for r in rr if r['mixture']==mix]
                ax.scatter([r['fpr']*100 for r in ss],[r['recall']*100 for r in ss],
                           color=color,marker=marker,s=30,label=label if mix=='two_source' else None)
        ax.axvline(.1,color='black',ls='--',lw=1);ax.axhline(50,color='black',ls=':',lw=1)
        ax.axvspan(0,.1,color='green',alpha=.08)
        ax.set(xlabel='False-positive rate (%)',ylabel='Attack recall (%)',title=title,xlim=(-.08,8.5),ylim=(-2,100))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Later ToN development stress period: no matched model meets both budgets\nCircles: two sources; triangles: three sources; each point is one training seed',fontsize=12)
    fig.savefig(O/'matched_generalization_tradeoff.png',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,rep,title in zip(axes,['raw46','harmonized14'],['46 exporter features','14 harmonized physical features']):
        for offset,mix,color in [(-.18,'two_source','#336699'),(.18,'three_source','#cc7733')]:
            means=[];mins=[];maxs=[]
            for method in methods:
                values=[100*r['ranking']['posthoc_threshold_oracle_recall']['0.001'] for r in rows
                        if r['representation']==rep and r['method']==method and r['mixture']==mix
                        and r['environment']=='ton_stress' and r['policy']=='pooled_budget']
                means.append(np.mean(values));mins.append(min(values));maxs.append(max(values))
            ax.bar(np.arange(5)+offset,means,width=.35,color=color,label=mix.replace('_',' '),
                   yerr=np.array([np.array(means)-mins,np.array(maxs)-means]),capsize=3)
        ax.axhline(50,color='black',ls=':',label='Required recall')
        ax.set(title=title,ylabel='Posthoc oracle recall at FPR ≤0.1% (%)',ylim=(0,55))
        ax.set_xticks(np.arange(5),['LGB bal.','XGB','XGB bal.','Pairwise','Pairwise bal.'],rotation=25,ha='right')
        ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('Ranking diagnostic: even label-assisted threshold choice falls far short\nBars: mean across 3 seeds; whiskers: full seed range. These are not deployment results.',fontsize=12)
    fig.savefig(O/'matched_low_fpr_ranking_diagnostic.png',dpi=180)
    plt.close(fig)
    print('Saved two development research figures.')

if __name__=='__main__':main()
