"""Chronological time-window diagnostics on fixed saved scores/thresholds."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation'
def main():
 with np.load(D/'evaluation.npz') as z:y=z['y']
 t=np.load(D/'evaluation_completion_ns.npy');records=[]
 for folder in ('causal_context_ablation','causal_context_components'):
  directory=R/'results'/folder
  source=directory/'results.json'
  data=json.loads(source.read_text());rows=data['results']
  for r in rows:
   if r['cap']!=.001:continue
   path=directory/f"scores_context{r['context']}_balanced{r['family_balanced']}_seed{r['seed']}.npz"
   with np.load(path) as z:pred=z['evaluation']>r['threshold']
   windows={}
   for seconds in (60,600):
    _,index=np.unique((t-t[0])//int(seconds*1e9),return_inverse=True)
    benign=np.bincount(index,weights=(y==0));attacks=np.bincount(index,weights=(y==1));fp=np.bincount(index,weights=pred&(y==0));tp=np.bincount(index,weights=pred&(y==1))
    has=benign>0;supported=benign>=100
    worst=int(np.argmax(np.where(has,fp/np.maximum(1,benign),-1)))
    windows[str(seconds)]=dict(nonempty=len(benign),benign_windows=int(has.sum()),benign_windows_under_100=int((has&~supported).sum()),worst_fpr=float(fp[worst]/benign[worst]),worst_fpr_benign_support=int(benign[worst]),max_false_positives_in_window=int(fp.max()),worst_fpr_min100benign=float((fp[supported]/benign[supported]).max()) if supported.any() else None)
   records.append(dict(context=r['context'],family_balanced=r['family_balanced'],seed=r['seed'],policy=r['policy'],windows=windows))
 (R/'results/detection_foundation/context_time_window_diagnostics.json').write_text(json.dumps(dict(scope='Fixed 60s and 600s completion-time bins. Counts refer to the 200000 sampled evaluation rows, not exhaustive full-capture alert exposure. Small-window FPR has high sampling uncertainty. No feature or threshold selection uses these labels.',results=records),indent=2));print('completed',len(records),'fixed-threshold time-window diagnostics')
if __name__=='__main__':main()
