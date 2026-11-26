"""Static scientific figures from completed frozen development evaluations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results/behavioral_observation_2026-11-22'
LABELS={'flow14':'Flow14','flow14_reference':'Flow14 reference','behavior24':'Behavior24','hybrid38':'Hybrid38','quantile_tail_anomaly':'Tail anomaly','behavior26_global_ablation':'Behavior+global'}

def main():
    full=json.loads((O/'full_replay_results.json').read_text())['results'];full=sorted(full,key=lambda x:list(LABELS).index(x['name']))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');names=[LABELS[x['name']] for x in full];pos=np.arange(len(full))
    for a,key,title in [(ax[0],'recall','Attack recall (%)'),(ax[1],'fpr','False-positive rate (%)'),(ax[2],'worst_supported_time_window_fpr','Worst supported 60s FPR (%)')]:
        a.barh(pos,[100*x[key] for x in full],color=['#6d7c90' if 'flow14' in x['name'] else '#007f85' if x['name']=='hybrid38' else '#b36b44' for x in full]);a.set_yticks(pos,names);a.invert_yaxis();a.set_xlabel(title)
        if key=='recall':a.axvline(50,color='black',ls='--',lw=1)
        else:a.axvline(.1,color='black',ls='--',lw=1)
        a.grid(axis='x',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Full ToN development replay: 9,160,253 flows, frozen seed42\nFPR cap 0.1%; recall target 50%; supported time windows contain at least100 benign flows',fontsize=12)
    fig.savefig(O/'full_development_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    for r in full:
        if r['name'] not in ['flow14_reference','behavior24','hybrid38']:continue
        windows=r['time_windows'];xx=[w['window']/60 for w in windows if w['benign']>=100];yy=[100*w['fpr'] for w in windows if w['benign']>=100]
        axes[0].scatter(xx,yy,label=LABELS[r['name']],alpha=.7,s=9)
        xx=[w['window']/60 for w in windows if w['attacks']>=100];yy=[100*w['recall'] for w in windows if w['attacks']>=100]
        axes[1].scatter(xx,yy,label=LABELS[r['name']],alpha=.7,s=9)
    axes[0].axhline(.1,color='black',ls='--',lw=1);axes[0].set_ylabel('60s FPR (%)');axes[1].set_ylabel('60s attack recall (%)');axes[1].set_xlabel('Hours since development-tail completion-time origin')
    for a in axes:a.legend(loc='upper right');a.grid(alpha=.2)
    fig.suptitle('Tail risk over chronological development traffic\nOnly supported windows plotted; missing periods are not zero-error observations',fontsize=12)
    fig.savefig(O/'chronological_tail_risk.png',dpi=180);plt.close(fig)
    rows=json.loads((O/'family_coverage_control/nearby_results.json').read_text())['results']
    fig,aa=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');cats=['flow14','hybrid38'];colors=['#6d7c90','#007f85']
    for j,condition in enumerate(['matched_coverage_control','withhold_ton_ddos']):
        for i,method in enumerate(cats):
            rr=[r for r in rows if r['condition']==condition and r['method']==method and r['environment']=='ton_future'];x=j+(i-.5)*.25
            for ax,key in zip(aa,['recall','fpr']):
                vals=[100*r[key] for r in rr];ax.bar(x,np.mean(vals),width=.23,color=colors[i],label=LABELS[method] if j==0 else None);ax.scatter([x]*len(vals),vals,color='black',s=15,zorder=3)
    for ax in aa:ax.set_xticks([0,1],['Matched coverage control','ToN DDoS excluded']);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    aa[0].set_ylabel('Nearby ToN attack recall (%)');aa[0].set_ylim(0,105);aa[1].set_ylabel('Nearby ToN FPR (%)');aa[1].axhline(.1,color='black',ls='--',lw=1)
    fig.suptitle('Coverage ablation: same fitting count and calibration\nExploratory development control; points are three seeds, not independent networks',fontsize=12)
    fig.savefig(O/'family_coverage_control.png',dpi=180);plt.close(fig)
    print('PLOTTED three behavioral figures',flush=True)
if __name__=='__main__':main()
