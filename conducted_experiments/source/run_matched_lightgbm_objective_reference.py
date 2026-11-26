"""Reuse fixed-stage LightGBM controls with original pooled FPR constraints.
Worst-window hard-cap selection is a separate stricter experiment, not the
original deployment constraint. No fits or protected-label access occur here.
"""
from pathlib import Path
import json,datetime,numpy as np,joblib
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,metrics,ranking_metrics
from multienvironment_protocol_tools import calibration_selection,development_holdouts,raw_score
from multienvironment_features import transform
from run_tolerance_threshold_audit import tolerance_threshold
ROOT=R/'results/multienvironment_2026-11-21';O=ROOT/'matched_lightgbm_objective_reference'
def main():
 O.mkdir(exist_ok=True);protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,representations=['raw46','harmonized14'],mixtures=['two_source','three_source'],seeds=[42,43,44],calibration_targets=[.001,.0005,.00025],evaluation_fpr_cap=.001,note='Exploratory development control added after earlier development outcomes. The original and stricter-window studies remain preserved; protected labels remain unused.',training_budget='One fixed stage0 fit per configuration, reused for matched XGBoost comparisons; 200000 training rows each.')
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');cal,sel,f=calibration_selection();choices=[]
 for rep in protocol['representations']:
  cx={e:transform(x,f,rep) for e,x in cal.items()};sx={e:transform(d['x'],f,rep) for e,d in sel.items()}
  for mix in protocol['mixtures']:
   for seed in protocol['seeds']:
    identity=f'{rep}_{mix}_lightgbm_logloss_balanced_seed{seed}';target=O/(identity+'_selection.json')
    if target.exists():choices.append(json.loads(target.read_text()));continue
    if mix=='two_source':folder=ROOT/'harmonization' if rep=='harmonized14' else ROOT;path=folder/f'{rep}_environment_balanced_seed{seed}_step0.joblib'
    else:path=ROOT/'source_coverage_pilot'/f'{rep}_three_source_full_seed{seed}_selected.joblib'
    model=joblib.load(path);cs={e:raw_score(model,x) for e,x in cx.items()};vs={e:raw_score(model,x) for e,x in sx.items()};candidates=[]
    for alpha in protocol['calibration_targets']:
     threshold={e:tolerance_threshold(p,alpha,.025)['threshold'] for e,p in cs.items()}
     if any(t is None for t in threshold.values()):continue
     val={e:metrics(d['y'],vs[e],threshold[e],d['family']) for e,d in sel.items()};candidates.append(dict(alpha=alpha,thresholds=threshold,validation=val,max_pooled_fpr=max(v['fpr'] for v in val.values()),min_pooled_recall=min(v['recall'] for v in val.values()),worst_fpr=max(v['worst_window_fpr'] for v in val.values()),worst_recall=min(v['worst_window_recall'] for v in val.values())))
    selected={}
    for rule in ['pooled_budget','window_budget']:
     risk='max_pooled_fpr' if rule=='pooled_budget' else 'worst_fpr';benefit='min_pooled_recall' if rule=='pooled_budget' else 'worst_recall';selected[rule]=max(candidates,key=lambda v:(v[risk]<=.001,-max(0,v[risk]-.001),v[benefit],-v['alpha']))
    choice=dict(identity=identity,representation=rep,mixture=mix,method='lightgbm_logloss_balanced',seed=seed,selected=selected,candidates=candidates,artifact=str(path.relative_to(R)))
    target.write_text(json.dumps(choice,indent=2),encoding='utf-8');choices.append(choice)
 (O/'selections.json').write_text(json.dumps(choices,indent=2),encoding='utf-8');hold=development_holdouts();out=[]
 for rep in protocol['representations']:
  hx={e:transform(d['x'],f,rep) for e,d in hold.items()}
  for c in [v for v in choices if v['representation']==rep]:
   model=joblib.load(R/c['artifact']);joblib.dump(model,O/(c['identity']+'_selected.joblib'));np.save(O/(c['identity']+'_input.npy'),hx['unsw_future'][:1000]);np.save(O/(c['identity']+'_raw_input.npy'),hold['unsw_future']['x'][:1000])
   for env,d in hold.items():
    p=raw_score(model,hx[env]);rank=ranking_metrics(d['y'],p)
    for rule,selected in c['selected'].items():
     threshold=selected['thresholds']['unsw' if env.startswith('unsw') else 'ton'];out.append(dict(identity=c['identity'],representation=rep,mixture=c['mixture'],method=c['method'],seed=c['seed'],environment=env,policy=rule,ranking=rank,**metrics(d['y'],p,threshold,d['family'],windows=20)))
 (O/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2),encoding='utf-8');print('COMPLETE matched fixed-stage LightGBM control',len(out))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
