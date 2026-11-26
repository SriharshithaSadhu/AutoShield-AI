"""Plot full-data CSE daily policy performance and attack prevalence."""
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

source=Path('results/cse_cic_ids2018_adaptation_all10_seed42.json')
d=json.loads(source.read_text(encoding='utf-8'))
days=d['protocol']['replayed_capture_files']
labels=[datetime.strptime(Path(x).name.split('_')[0][-10:],'%d-%m-%Y').strftime('%d %b') for x in days]
policies=['no_adaptation','feature_psi','label_error','periodic_5_chunks']
pretty={'no_adaptation':'No adaptation','feature_psi':'Feature PSI','label_error':'Labeled error (end-of-chunk)','periodic_5_chunks':'Periodic (5 chunks)'}
colors={'no_adaptation':'#555555','feature_psi':'#2b8cbe','label_error':'#e34a33','periodic_5_chunks':'#756bb1'}
def daily_metrics(rows, filename):
    chunks=[r for r in rows if r['capture_file']==filename]
    n=sum(r['rows'] for r in chunks); attack=sum(r['attack_rows'] for r in chunks)
    tn=sum(r['tn'] for r in chunks); fp=sum(r['fp'] for r in chunks)
    fn=sum(r['fn'] for r in chunks); tp=sum(r['tp'] for r in chunks)
    f1_attack=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    f1_benign=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.0
    weighted=(attack*f1_attack+(n-attack)*f1_benign)/max(1,n)
    return weighted, attack/max(1,n)

fig,(ax,bx)=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[3,1]})
x=np.arange(len(days))
for policy in policies:
    rows=d['summaries'][policy]['chunk_records']
    values=[daily_metrics(rows,filename)[0] for filename in days]
    ax.plot(x,values,marker='o',linewidth=2,label=pretty[policy],color=colors[policy])
prevalence=[daily_metrics(d['summaries']['no_adaptation']['chunk_records'],filename)[1] for filename in days]
bx.bar(x,np.array(prevalence)*100,color='#9ecae1',edgecolor='#3182bd')
ax.set_ylim(-0.03,1.03); ax.set_ylabel('Per-day weighted F1')
ax.set_title('CSE-CIC-IDS2018 chronological replay: policy quality follows capture-day shifts')
ax.grid(axis='y',alpha=.25); ax.legend(ncol=2,loc='lower left',frameon=True)
bx.set_ylabel('Attack rows (%)'); bx.set_ylim(0,100); bx.grid(axis='y',alpha=.25)
bx.set_xticks(x,labels,rotation=25,ha='right')
bx.set_xlabel('Timestamp-sorted capture day')
fig.text(.01,.01,'Seed 42; all ten official files acquired; nine replay days; 15.18M valid rows. Daily F1 pools confusion counts within each day.',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,0.04,1,1])
out=Path('results/cse_daily_policy_f1_seed42.png')
fig.savefig(out,dpi=220,bbox_inches='tight')
print(out.resolve())

