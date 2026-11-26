"""Scientific development plots; whiskers are seed ranges, never confidence intervals."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results/generalization_study_2026-11-19'
def main():
 O.mkdir(exist_ok=True)
 rows=[]
 for folder in ('causal_context_ablation','causal_context_components'):rows+=json.loads((R/'results'/folder/'results.json').read_text())['results']
 groups=[(False,False,'46 features'),(False,True,'46 + family weights'),(True,False,'46 + four history features'),(True,True,'History + family weights'),('local',False,'46 + local history'),('global',False,'46 + global traffic rate')]
 fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True,layout='constrained')
 for pi,policy in enumerate(('pooled','worst_block')):
  color=['#2166ac','#d95f02'][pi]
  for i,(context,balanced,label) in enumerate(groups):
   rs=[r for r in rows if r['context']==context and r['family_balanced']==balanced and r['cap']==.001 and r['policy']==policy]
   for ax,metric in zip(axes,('fpr','recall')):
    values=np.array([r[metric]*100 for r in rs]);mean=values.mean();pos=i+(pi-.5)*.23
    ax.errorbar(mean,pos,xerr=[[mean-values.min()],[values.max()-mean]],fmt='o',color=color,capsize=3,label=policy if i==0 else None)
 axes[0].set_yticks(range(len(groups)),[g[2] for g in groups]);axes[0].invert_yaxis();axes[0].set_xscale('symlog',linthresh=.1)
 axes[0].set_xlim(0,100);axes[0].set_xticks([0,.1,1,10,100],['0','0.1','1','10','100'])
 axes[0].axvline(.1,color='#b2182b',ls='--',lw=1);axes[1].axvline(50,color='#b2182b',ls='--',lw=1)
 axes[0].set_xlabel('False-positive rate (%) — nonlinear axis');axes[1].set_xlabel('Attack recall (%)');axes[1].set_xlim(0,100)
 for ax in axes:ax.grid(axis='x',alpha=.2)
 fig.legend(*axes[0].get_legend_handles_labels(),title='Calibration policy',loc='lower center',bbox_to_anchor=(.6,-.12),ncol=2);fig.suptitle('Fixed 0.1% calibration target: chronological development performance\nDots: three-seed mean; whiskers: seed range; red lines: FPR cap / recall target',fontsize=12)
 fig.savefig(O/'context_ablation.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print('Saved',O/'context_ablation.png')
if __name__=='__main__':main()
