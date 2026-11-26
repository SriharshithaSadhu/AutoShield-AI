"""Exact fractional envelope of the completed raw-text training census."""
import json,sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observability_relaxation import rank_groups
O=Path(__file__).resolve().parent/'results/network_context_2026-11-21'

def main():
    b=O/'full_training_role_text_audit';r=json.loads((b/'results.json').read_text());env=json.loads((b/'fractional_envelope.json').read_text())
    conn=sqlite3.connect('file:'+str((b/'groups.sqlite').resolve())+'?mode=ro',uri=True)
    pairs=conn.execute('SELECT scanning,benign FROM groups WHERE scanning>0 AND benign>0').fetchall();conn.close()
    pairs=rank_groups([p[0] for p in pairs],[p[1] for p in pairs])
    fp=np.r_[0,np.cumsum([p[1] for p in pairs])];tp=np.r_[r['free_attack_only_tp'],r['free_attack_only_tp']+np.cumsum([p[0] for p in pairs])]
    fig,axes=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained')
    ax=axes[0];ax.plot(fp/r['benign']*100,tp/r['scanning']*100,color='#336699',label='Fractional upper envelope')
    ax.scatter([.1],[100*r['optimal_scanning_recall']],s=65,color='#228855',zorder=3,label='Exact deterministic optimum at 0.1% FPR')
    ax.axvline(.1,color='black',ls='--',lw=1);ax.axhline(50,color='black',ls=':',lw=1,label='50% scanning recall reference')
    ax.set(xlim=(0,5),ylim=(0,60),xlabel='Empirical benign false-positive rate (%)',ylabel='Optimistic scanning recall (%)',title='Recorded inputs impose an empirical ceiling')
    ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.15)
    axes[1].axis('off');text=(f"Full permitted training prefix\n\n"
        f"Flows: {r['prefix_rows']:,}\nBenign: {r['benign']:,}\nScanning: {r['scanning']:,}\n\n"
        f"Inputs: all 46 raw aggregate fields\n+ fixed service category and both role hints\n\n"
        f"At 0.1% FPR:\n  Maximum scanning recall: {100*r['optimal_scanning_recall']:.2f}%\n  Exact TP ceiling: {r['optimal_tp']:,}\n\n"
        f"For 50% scanning recall:\n  FPR must be at least\n  {100*env['minimum_fpr_fractional_lower_bound_for_50percent_scanning']:.3f}% (optimistic lower bound).")
    axes[1].text(.02,.96,text,va='top',fontsize=11,linespacing=1.35)
    fig.suptitle('Raw-exporter-text census confirms the stateless scanning limitation',fontsize=13)
    fig.supxlabel('Label-assisted empirical bounds, not detector performance or future guarantees. Scanning-family bounds do not prove overall-recall infeasibility.',fontsize=8)
    fig.savefig(O/'full_training_observability_ceiling.png',dpi=180);plt.close(fig)
    print('Saved full-prefix empirical observability figure.')
if __name__=='__main__':main()
