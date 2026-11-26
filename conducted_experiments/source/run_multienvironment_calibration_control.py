"""Static per-deployment calibration control on the identical frozen fitted models.
No new training, features, architectures or protected labels. This distinguishes
training improvements from an overly restrictive shared probability cutoff.
"""
from pathlib import Path
import json,datetime,numpy as np,joblib
from threadpoolctl import threadpool_limits
from run_multienvironment_training import read,metrics,ranking_metrics,ton_temporal_boundaries,R,D,U
from multienvironment_features import transform
from run_tolerance_threshold_audit import tolerance_threshold
O=R/'results/multienvironment_2026-11-21';B=O/'network_calibration'
def main():
 B.mkdir(exist_ok=True);ton_cal_stop,ton_select_stop=ton_temporal_boundaries();protocol=dict(created_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,selection_rule='Same feasibility-first worst-window ranking as shared-threshold experiment; only static calibration scope differs.',calibration=dict(benign_per_network=20000,cap=.001,delta=.025),protected_policy='No protected label access in this executable. Protected evaluation requires a separate locked run after original joint criteria pass on all declared development stress environments plus measured resources.',note='Introduced during development before outcome arrays were scored; it is an additional methodological control, not the original preregistration.')
 (B/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 p=json.loads((O/'protocol.json').read_text());features=json.loads((D/'manifest.json').read_text())['features'];ton=read(D/'train.npz');unsw=read(U/'train.npz');tune=read(D/'tune.npz');uc=read(U/'calibration.npz');us=read(U/'selection.npz');rng=np.random.default_rng(20261121)
 rng.choice(len(ton['y']),150000,replace=False);rng.choice(len(unsw['y']),50000,replace=False)
 tc={k:v[:ton_cal_stop] for k,v in tune.items()};ts={k:v[ton_cal_stop:ton_select_stop] for k,v in tune.items()};cal={}
 for name,d in [('ton',tc),('unsw',uc)]:
  ix=np.flatnonzero(d['y']==0);ii=np.sort(rng.choice(ix,20000,replace=False));cal[name]=d['x'][ii]
 selections={'ton':ts,'unsw':us};choices=[]
 for representation in p['representations']:
  cx={e:transform(x,features,representation) for e,x in cal.items()};sx={e:transform(d['x'],features,representation) for e,d in selections.items()}
  for method in p['methods']:
   for seed in p['seeds']:
    identity=f'{representation}_{method}_seed{seed}';target=B/(identity+'_selection.json')
    if target.exists():choices.append(json.loads(target.read_text()));continue
    original=json.loads((O/(identity+'_selection.json')).read_text());candidates=[]
    for record in original['candidates']:
     m=joblib.load(R/record['artifact']);threshold={e:tolerance_threshold(m.booster_.predict(x,num_threads=1),.001,.025)['threshold'] for e,x in cx.items()}
     assert abs(max(threshold.values())-record['thresholds']['max_environment'])<1e-12,'calibration budget/sample mismatch'
     validation_scores={e:m.booster_.predict(x,num_threads=1) for e,x in sx.items()}
     calibration_scores={e:m.booster_.predict(x,num_threads=1) for e,x in cx.items()}
     np.savez_compressed(B/(identity+f'_step{record["step"]}_scores.npz'),**{f'cal_{e}':v for e,v in calibration_scores.items()},**{f'selection_{e}':v for e,v in validation_scores.items()})
     v={e:metrics(selections[e]['y'],validation_scores[e],threshold[e],selections[e]['family']) for e in sx}
     candidates.append(dict(step=record['step'],artifact=record['artifact'],thresholds=threshold,validation=v,worst_fpr=max(z['worst_window_fpr'] for z in v.values()),worst_recall=min(z['worst_window_recall'] for z in v.values()),bytes=record['bytes'],p95_ms=record['p95_ms']))
    eligible=[v for v in candidates if v['bytes']<=1572864 and v['p95_ms']<=5];chosen=max(eligible or candidates,key=lambda v:(v['worst_fpr']<=.001,-max(0,v['worst_fpr']-.001),v['worst_recall'],-v['step']))
    decision=dict(identity=identity,method=method,representation=representation,seed=seed,selected=chosen,candidates=candidates)
    target.write_text(json.dumps(decision,indent=2),encoding='utf-8');choices.append(decision);print('network calibrated selection',identity,chosen['worst_fpr'],chosen['worst_recall'],flush=True)
 (B/'selections.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
 # All choices persisted before the held-out development arrays are opened.
 holdouts={'ton_future':{k:v[ton_select_stop:] for k,v in tune.items()},'unsw_future':read(U/'development_holdout.npz'),'ton_stress':read(D/'evaluation.npz')};out=[]
 for representation in p['representations']:
  hx={e:transform(d['x'],features,representation) for e,d in holdouts.items()}
  for choice in [v for v in choices if v['representation']==representation]:
   m=joblib.load(R/choice['selected']['artifact']);held_scores={}
   for env,d in holdouts.items():
    threshold=choice['selected']['thresholds']['unsw' if env.startswith('unsw') else 'ton'];scores=m.booster_.predict(hx[env],num_threads=1);held_scores[env]=scores
    out.append(dict(identity=choice['identity'],method=choice['method'],representation=representation,seed=choice['seed'],environment=env,policy='network_specific',ranking=ranking_metrics(d['y'],scores),**metrics(d['y'],scores,threshold,d['family'],windows=20)))
   np.savez_compressed(B/(choice['identity']+'_holdout_scores.npz'),**held_scores)
   (B/'results.partial.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 (B/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2),encoding='utf-8');print('COMPLETE network calibration',len(out),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
