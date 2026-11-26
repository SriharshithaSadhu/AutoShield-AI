"""Paper figure: paired capture-level FPR/recall changes after periodic CSE adaptation."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results'/'iotid20_cse_transfer_by_capture_seed42.csv'
OUT=ROOT/'results'/'IOTID20_CSE_TRANSFER_CAPTURE_RISK_TRADEOFF'
ARMS={
 'default_static':'reference_default_no_adaptation',
 'default_periodic':'reference_default_periodic_refit',
 'feasible_static':'mopso_best_feasible_no_adaptation',
 'feasible_periodic':'mopso_best_feasible_periodic_refit',
}

def main():
 d=pd.read_csv(SRC)
 captures=sorted(d.capture_file.unique())
 fig,axes=plt.subplots(1,2,figsize=(11.2,5.0),sharey=True,constrained_layout=True)
 panels=[('Default source model','default_static','default_periodic'),('Feasible MOPSO source model','feasible_static','feasible_periodic')]
 for ax,(title,static_key,adapt_key) in zip(axes,panels):
  static=d[d.arm==ARMS[static_key]].set_index('capture_file')
  adapt=d[d.arm==ARMS[adapt_key]].set_index('capture_file')
  for c in captures:
   a,b=static.loc[c],adapt.loc[c]
   x1=max(float(a.false_positive_rate),1e-5); x2=max(float(b.false_positive_rate),1e-5)
   y1=float(a.attack_recall); y2=float(b.attack_recall)
   ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#9aa5b1',alpha=.72,lw=1.25,shrinkA=4,shrinkB=4))
   ax.scatter(x1,y1,marker='o',s=43,color='#3e668e',edgecolor='white',linewidth=.45,zorder=3)
   ax.scatter(x2,y2,marker='s',s=43,color='#df7f3c',edgecolor='white',linewidth=.45,zorder=3)
  ax.axvline(.001,color='#b33a3a',ls='--',lw=1.3)
  ax.set_xscale('log'); ax.set_xlim(1e-5,1.15); ax.set_ylim(-.03,1.04)
  ax.set_title(title,fontsize=11,fontweight='semibold')
  ax.set_xlabel('Capture-level benign false-positive rate (log scale)')
  ax.grid(True,which='both',alpha=.2)
  ax.text(.0012,.03,'0.1% cap',color='#9e2f2f',fontsize=8,rotation=90,va='bottom')
 axes[0].set_ylabel('Capture-level attack recall')
 handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#3e668e',markeredgecolor='white',markersize=7,label='No adaptation'),Line2D([0],[0],marker='s',color='none',markerfacecolor='#df7f3c',markeredgecolor='white',markersize=7,label='Periodic supervised refit')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.075),ncol=2,frameon=False)
 fig.suptitle('Capture-specific risk/recall trade-off under cross-dataset shift',y=1.14,fontsize=13,fontweight='bold')
 fig.text(.5,-.04,'Each arrow pairs one CSE capture; circles are static predictions and squares are refitted predictions. One source seed; ten captures from one already explored collection. Not independent-network evidence.',ha='center',fontsize=8.5,color='#444')
 fig.savefig(OUT.with_suffix('.png'),dpi=220,bbox_inches='tight')
 fig.savefig(OUT.with_suffix('.pdf'),bbox_inches='tight')
 maplines=['capture_id,capture_file']+[f'C{i+1},{c}' for i,c in enumerate(captures)]
 OUT.with_name(OUT.name+'_capture_key.csv').write_text('\n'.join(maplines)+'\n',encoding='utf-8')
 print(OUT.with_suffix('.png'));print(OUT.with_suffix('.pdf'))
if __name__=='__main__': main()
