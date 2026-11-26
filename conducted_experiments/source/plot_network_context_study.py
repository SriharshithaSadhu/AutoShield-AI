"""Publication-style development figures; no model or threshold selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent/'results/network_context_2026-11-21'

def main():
    rows=json.loads((O/'results.json').read_text())['results']
    variants=['physical14','direction15','service26','protocol19','context29','harmonized14','harmonized_context29']
    labels=['Baseline14','Direction15','Service26','Protocol19','Context29','Harmonized14','Harm. context29'];colors=plt.cm.tab10(np.arange(7))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    means=[];low=[];high=[]
    for variant,label,color in zip(variants,labels,colors):
        rr=[r for r in rows if r['variant']==variant and r['policy']=='shared_window' and r['environment']=='ton_stress']
        axes[0].scatter([100*r['fpr'] for r in rr],[100*r['recall'] for r in rr],color=color,label=label,s=35)
        values=np.array([100*r['ranking']['posthoc_threshold_oracle_recall']['0.001'] for r in rr]);means.append(values.mean());low.append(values.min());high.append(values.max())
    axes[0].axvline(.1,color='black',ls='--',lw=1);axes[0].axhline(50,color='black',ls=':',lw=1)
    axes[0].set(xlabel='False-positive rate (%)',ylabel='Attack recall (%)',xlim=(0,3),ylim=(0,100),title='Frozen historical threshold: 3 seeds')
    axes[0].legend(fontsize=8,ncol=2,loc='upper left');axes[0].grid(alpha=.15)
    axes[1].bar(np.arange(7),means,color=colors,yerr=np.array([np.array(means)-low,np.array(high)-means]),capsize=3)
    axes[1].axhline(50,color='black',ls=':',label='Required recall')
    axes[1].set_xticks(np.arange(7),labels,rotation=35,ha='right');axes[1].set(ylabel='Posthoc oracle recall at FPR ≤0.1% (%)',ylim=(0,55),title='Ranking diagnostic, not deployment performance')
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.15)
    fig.suptitle('Static context does not close the later ToN development gap\nSame training rows, model capacity, benign calibration and fit budgets; whiskers show the full seed range.',fontsize=12)
    fig.savefig(O/'context_features_tradeoff.png',dpi=180);plt.close(fig)

    full=json.loads((O/'full_development_replay/results.json').read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    family_order=['ddos','password','xss','Backdoor','mitm','ransomware']
    for offset,variant,label,color in [(-.18,'physical14','Baseline14','#336699'),(.18,'context29','Context29','#cc7733')]:
        rr=[r for r in full if r['variant']==variant];w=rr[0]['flow_windows'];ids=np.array([z['window'] for z in w]);good=np.array([z['benign']>=100 for z in w])
        rates=np.array([[z['fpr']*100 if z['fpr'] is not None else np.nan for z in r['flow_windows']] for r in rr]);rates[:,~good]=np.nan
        axes[0].plot(ids*.01,np.mean(rates,axis=0),color=color,label=label,lw=.8)
        axes[0].fill_between(ids*.01,np.min(rates,axis=0),np.max(rates,axis=0),color=color,alpha=.15)
        values=np.array([[r['family'][f]['recall']*100 for f in family_order] for r in rr]);mean=values.mean(axis=0)
        axes[1].bar(np.arange(6)+offset,mean,width=.35,color=color,label=label,
                    yerr=np.array([mean-values.min(axis=0),values.max(axis=0)-mean]),capsize=2)
    axes[0].axhline(.1,color='black',ls='--',lw=.8);axes[0].set(xlabel='Flow offset within replay (millions)',ylabel='10,000-flow-window FPR (%)',title='All 9,160,253 chronological development flows',ylim=(0,60));axes[0].legend(fontsize=8)
    axes[1].set_xticks(np.arange(6),family_order,rotation=25,ha='right');axes[1].set(ylabel='Attack-family recall (%)',ylim=(0,105),title='Full replay: gains and regressions by family');axes[1].legend(fontsize=8)
    fig.suptitle('Full development replay confirms unstable false alarms and uneven family coverage\nLines/bars: mean across 3 trained seeds; shading/whiskers: seed range. Windows require ≥100 benign flows.',fontsize=11)
    fig.savefig(O/'full_context_window_risk.png',dpi=180);plt.close(fig)

    info=json.loads((O/'family_information_audit.json').read_text())['results'];names=['physical14','context29','raw46'];rr=[next(r for r in info if r['environment']=='ton_training' and r['family']=='scanning' and r['variant']==v) for v in names]
    rr.append(json.loads((O/'complete_context_observability_audit.json').read_text())['bound'])
    fig,ax=plt.subplots(figsize=(8,5.5));fig.subplots_adjust(bottom=.23,top=.81,left=.11,right=.98);values=[100*r['label_assisted_optimal_recall'] for r in rr]
    ax.bar(range(4),values,color=['#6688aa','#cc7733','#336699','#339966'])
    for i,v in enumerate(values):ax.text(i,v+.6,f'{v:.2f}%',ha='center',fontsize=10)
    ax.axhline(50,color='black',ls=':',label='50% recall reference');ax.set_xticks(range(4),['Physical14','Context29','All46 aggregates','All46 + complete\nservice/role hints'])
    ax.set(ylabel='Maximum empirical scanning recall (%)',ylim=(0,55),title='Scanning observability on the fixed ToN training sample\n31,326 scanning flows; 75,180 benign flows; at most 75 false positives')
    ax.tick_params(axis='x',labelsize=9)
    ax.legend(fontsize=8);fig.text(.5,.035,'Label-assisted lookup/knapsack optimum, not a trained model or future guarantee.\nIndependent MILP verification; same recorded vectors require the same decision.',ha='center',fontsize=8)
    fig.savefig(O/'scanning_observability_gap.png',dpi=180);plt.close(fig)
    print('Saved three context-study scientific figures.')
if __name__=='__main__':main()
