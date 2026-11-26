"""Published ARW mean-estimator baseline adapted to fixed IDS candidate selection.

Han et al. Algorithm 1 is implemented; their pairwise tournament is not.
FPR and recall estimates use the same chronological validation labels/bank as
our other rules. This adaptation supplies no new statistical guarantee.
"""
from pathlib import Path
import json,numpy as np,joblib
from threadpoolctl import threadpool_limits
from adaptive_rolling_mean import adaptive_rolling_mean
from run_representation_selection import load,predict
from run_tolerance_threshold_audit import tolerance_threshold
R=Path(__file__).resolve().parent;O=R/'results/representation_selection'
def main():
 protocol=json.loads((O/'protocol.json').read_text());runs=[]
 for seed in protocol['seeds']:
  candidates=[]
  for ci,cfg in enumerate(protocol['bank']):
   record=json.loads((O/f'candidate_seed{seed}_id{ci}.json').read_text());model=joblib.load(O/f'candidate_seed{seed}_id{ci}.joblib');d=load('tune',cfg['context'])
   p=predict(model,d['x'])>record['proxy_threshold'];windows=np.array_split(np.arange(len(p)),3)
   fpr=adaptive_rolling_mean([p[ix][d['y'][ix]==0].astype(float) for ix in windows],delta=.05/(2*len(protocol['bank'])))
   recall=adaptive_rolling_mean([p[ix][d['y'][ix]==1].astype(float) for ix in windows],delta=.05/(2*len(protocol['bank'])))
   candidates.append(dict(id=ci,fpr=fpr,recall=recall,bytes=record['bytes'],p95_ms=record['p95_ms']))
  eligible=[c for c in candidates if c['bytes']<=1572864 and c['p95_ms']<=5]
  chosen=max(eligible or candidates,key=lambda c:(c['fpr']['estimate']<=.001,-max(0,c['fpr']['estimate']-.001),c['recall']['estimate'],-c['id']))
  (O/f'arw_decision_seed{seed}.json').write_text(json.dumps(dict(candidates=candidates,chosen=chosen),indent=2))
  ci=chosen['id'];cfg=protocol['bank'][ci];model=joblib.load(O/f'candidate_seed{seed}_id{ci}.joblib');cal=load('calibration',cfg['context']);ev=load('evaluation',cfg['context'])
  cutoff=tolerance_threshold(predict(model,cal['x']),.001,.05)['threshold'];pred=predict(model,ev['x'])>cutoff;y=ev['y']
  worst=max(float(pred[ix][y[ix]==0].mean()) for ix in np.array_split(np.arange(len(y)),20) if (y[ix]==0).any())
  runs.append(dict(seed=seed,selected=ci,selection_fpr=chosen['fpr']['estimate'],selection_recall=chosen['recall']['estimate'],fpr=float(pred[y==0].mean()),recall=float(pred[y==1].mean()),worst_window_fpr=worst,fp=int(pred[y==0].sum()),tp=int(pred[y==1].sum())))
  print(runs[-1],flush=True)
 (O/'arw_results.json').write_text(json.dumps(dict(scope=__doc__,runs=runs),indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
