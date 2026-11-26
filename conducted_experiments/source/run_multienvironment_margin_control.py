"""Matched calibration safety-margin search; evaluation FPR cap remains 0.001."""
from pathlib import Path
import json,numpy as np,joblib
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,D,U,read,metrics,ranking_metrics,ton_temporal_boundaries
from multienvironment_features import transform
from run_tolerance_threshold_audit import tolerance_threshold
O=R/'results/multienvironment_2026-11-21';B=O/'network_calibration';C=O/'calibration_margin'
def main():
 C.mkdir(exist_ok=True);cal_stop,sel_stop=ton_temporal_boundaries();tune=read(D/'tune.npz');us=read(U/'selection.npz');sel={'ton':{k:v[cal_stop:sel_stop] for k,v in tune.items()},'unsw':us};protocol=json.loads((O/'calibration_margin_protocol.json').read_text());sources=json.loads((B/'selections.json').read_text());choices=[]
 for original in sources:
  target=C/(original['identity']+'_selection.json')
  if target.exists():choices.append(json.loads(target.read_text()));continue
  candidates=[]
  for old in original['candidates']:
   with np.load(B/(original['identity']+f'_step{old["step"]}_scores.npz')) as z:scores={k:z[k] for k in z.files}
   for alpha in protocol['calibration_targets']:
    thresholds={e:tolerance_threshold(scores['cal_'+e],alpha,.025)['threshold'] for e in sel}
    if any(v is None for v in thresholds.values()):continue
    v={e:metrics(d['y'],scores['selection_'+e],thresholds[e],d['family']) for e,d in sel.items()}
    candidates.append(dict(step=old['step'],artifact=old['artifact'],alpha=alpha,thresholds=thresholds,validation=v,worst_fpr=max(w['worst_window_fpr'] for w in v.values()),worst_recall=min(w['worst_window_recall'] for w in v.values()),bytes=old['bytes'],p95_ms=old['p95_ms']))
  eligible=[v for v in candidates if v['bytes']<=1572864 and v['p95_ms']<=5];chosen=max(eligible or candidates,key=lambda v:(v['worst_fpr']<=.001,-max(0,v['worst_fpr']-.001),v['worst_recall'],-v['step'],-v['alpha']))
  choice=dict(identity=original['identity'],method=original['method'],representation=original['representation'],seed=original['seed'],selected=chosen,candidates=candidates)
  target.write_text(json.dumps(choice,indent=2),encoding='utf-8');choices.append(choice)
 (C/'selections.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
 # All new choices frozen before development holdout outcomes are loaded.
 hold={'ton_future':{k:v[sel_stop:] for k,v in tune.items()},'unsw_future':read(U/'development_holdout.npz'),'ton_stress':read(D/'evaluation.npz')};features=json.loads((D/'manifest.json').read_text())['features'];out=[]
 for rep in sorted({v['representation'] for v in choices}):
  hx={e:transform(d['x'],features,rep) for e,d in hold.items()}
  for choice in [v for v in choices if v['representation']==rep]:
   m=joblib.load(R/choice['selected']['artifact']);saved={};joblib.dump(m,C/(choice['identity']+'_selected.joblib'));np.save(C/(choice['identity']+'_input.npy'),hx['unsw_future'][:1000]);np.save(C/(choice['identity']+'_raw_input.npy'),hold['unsw_future']['x'][:1000])
   for env,d in hold.items():
    threshold=choice['selected']['thresholds']['unsw' if env.startswith('unsw') else 'ton'];score=m.booster_.predict(hx[env],num_threads=1);saved[env]=score
    out.append(dict(identity=choice['identity'],method=choice['method'],representation=rep,seed=choice['seed'],environment=env,policy='selected_calibration_margin',calibration_alpha=choice['selected']['alpha'],ranking=ranking_metrics(d['y'],score),**metrics(d['y'],score,threshold,d['family'],windows=20)))
   np.savez_compressed(C/(choice['identity']+'_holdout_scores.npz'),**saved);(C/'results.partial.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 (C/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2),encoding='utf-8');print('COMPLETE margin control',len(out),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
