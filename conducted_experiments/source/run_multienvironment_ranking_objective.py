"""Matched XGBoost classification vs within-environment pairwise ranking.
Established objectives only: no new algorithm or theoretical guarantee claimed.
Protected UNSW suffix is never loaded. Two retrospective source mixtures, fixed
200000-row budget, one fit/config/seed, identical features and benign label budget.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
from time import process_time,perf_counter
import json,datetime,numpy as np,joblib
from xgboost import XGBClassifier,XGBRanker
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,D,U,read,metrics,ranking_metrics,ton_temporal_boundaries
from run_auxiliary_source_coverage_pilot import balanced_pick,balanced_environment_class
from multienvironment_features import transform
from run_tolerance_threshold_audit import tolerance_threshold
ROOT=R/'results/multienvironment_2026-11-21';O=ROOT/'ranking_objective'
def score(model,x):return model.get_booster().inplace_predict(np.ascontiguousarray(x),predict_type='margin')
def main():
 O.mkdir(exist_ok=True);protocol=dict(created_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,xgboost_version='1.7.6',representations=['raw46','harmonized14'],mixtures=['two_source','three_source'],methods=['logloss_unweighted','logloss_balanced','pairwise_unweighted','pairwise_environment_balanced'],seeds=[42,43,44],training_rows=200000,model=dict(n_estimators=100,max_depth=4,min_child_weight=10,reg_lambda=10,learning_rate=.1,subsample=.8,colsample_bytree=.9,tree_method='hist',n_jobs=1,verbosity=0),calibration_targets=[.001,.0005,.00025],evaluation_fpr_cap=.001,calibration_benign_per_network=20000,score='Raw margins for classifiers and relevance scores for rankers; no probability-rounding artifacts.',selection_rules=['pooled_budget','window_budget'],weight_scope='Classifier balanced weights equalize environment/class loss mass. Ranker balanced weights are inverse query sizes. Those are distinct APIs, so compare complete training schemes, not identical weighted losses.',query_scope='One query per environment; query IDs are metadata, never model features.',chronology_note='Within-network chronological periods; UNSW2015 target with later source data is retrospective, ToN2019 target uses prior2015/2018 sources.',protected_labels_used=False)
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');features=json.loads((D/'manifest.json').read_text())['features'];t=read(D/'train.npz');u=read(U/'train.npz');c=read(R/'data/nf_cicids2018_v3/permitted_source/source.npz');tune=read(D/'tune.npz');uc=read(U/'calibration.npz');us=read(U/'selection.npz');cal_stop,sel_stop=ton_temporal_boundaries();rng=np.random.default_rng(20261121);it=np.sort(rng.choice(len(t['y']),150000,False));iu=np.sort(rng.choice(len(u['y']),50000,False));core=np.sort(np.random.default_rng(20261122).choice(it,100000,False));rr=np.random.default_rng(20261123);benign=rr.choice(np.flatnonzero(c['y']==0),50000,False);attack=balanced_pick(np.flatnonzero(c['y']==1),c['family'],25000,rr);ic=np.r_[benign[:25000],attack]
 train={'two_source':dict(x=np.r_[t['x'][it],u['x'][iu]],y=np.r_[t['y'][it],u['y'][iu]],env=np.r_[np.zeros(len(it),int),np.ones(len(iu),int)],groups=[150000,50000]),'three_source':dict(x=np.r_[t['x'][core],u['x'][iu],c['x'][ic]],y=np.r_[t['y'][core],u['y'][iu],c['y'][ic]],env=np.r_[np.zeros(len(core),int),np.ones(len(iu),int),np.full(len(ic),2)],groups=[100000,50000,50000])}
 tc={k:v[:cal_stop] for k,v in tune.items()};selection={'ton':{k:v[cal_stop:sel_stop] for k,v in tune.items()},'unsw':us};cal={}
 for name,d in [('ton',tc),('unsw',uc)]:
  ii=np.sort(rng.choice(np.flatnonzero(d['y']==0),20000,False));cal[name]=d['x'][ii]
 choices=[]
 for rep in protocol['representations']:
  cx={e:transform(v,features,rep) for e,v in cal.items()};sx={e:transform(d['x'],features,rep) for e,d in selection.items()}
  for mix,d in train.items():
   x=transform(d['x'],features,rep);y=d['y'];class_weight=balanced_environment_class(d['env'],y);query_weights=len(y)/(len(d['groups'])*np.asarray(d['groups'],float))
   for method in protocol['methods']:
    for seed in protocol['seeds']:
     identity=f'{rep}_{mix}_{method}_seed{seed}';target=O/(identity+'_selection.json')
     if target.exists():choices.append(json.loads(target.read_text()));continue
     params=dict(protocol['model'],random_state=seed);before=process_time()
     if method.startswith('pairwise'):
      model=XGBRanker(**params,objective='rank:pairwise');qw=query_weights if method.endswith('environment_balanced') else np.ones(len(d['groups']));model.fit(x,y,group=d['groups'],sample_weight=qw)
     else:
      model=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss');model.fit(x,y,sample_weight=class_weight if method.endswith('balanced') and not method.endswith('unweighted') else None)
     fit_cpu=process_time()-before;artifact=O/(identity+'_selected.joblib');joblib.dump(model,artifact);cs={e:score(model,z) for e,z in cx.items()};vs={e:score(model,z) for e,z in sx.items()};candidates=[]
     for alpha in protocol['calibration_targets']:
      threshold={e:tolerance_threshold(v,alpha,.025)['threshold'] for e,v in cs.items()}
      if any(z is None for z in threshold.values()):continue
      val={e:metrics(dv['y'],vs[e],threshold[e],dv['family']) for e,dv in selection.items()};candidates.append(dict(alpha=alpha,thresholds=threshold,validation=val,max_pooled_fpr=max(v['fpr'] for v in val.values()),min_pooled_recall=min(v['recall'] for v in val.values()),worst_fpr=max(v['worst_window_fpr'] for v in val.values()),worst_recall=min(v['worst_window_recall'] for v in val.values())))
     selected={}
     for rule in protocol['selection_rules']:
      risk='max_pooled_fpr' if rule=='pooled_budget' else 'worst_fpr';benefit='min_pooled_recall' if rule=='pooled_budget' else 'worst_recall';selected[rule]=max(candidates,key=lambda v:(v[risk]<=.001,-max(0,v[risk]-.001),v[benefit],-v['alpha']))
     record=dict(identity=identity,representation=rep,mixture=mix,method=method,seed=seed,selected=selected,candidates=candidates,artifact=str(artifact.relative_to(R)),bytes=artifact.stat().st_size,fit_cpu_s=fit_cpu)
     target.write_text(json.dumps(record,indent=2),encoding='utf-8');choices.append(record);print('ranking choice',identity,'fit CPU',round(fit_cpu,2),'pooled',selected['pooled_budget']['min_pooled_recall'],'window',selected['window_budget']['worst_recall'],flush=True)
 (O/'selections.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
 hold={'ton_future':{k:v[sel_stop:] for k,v in tune.items()},'unsw_future':read(U/'development_holdout.npz'),'ton_stress':read(D/'evaluation.npz')};out=[]
 for rep in protocol['representations']:
  hx={e:transform(d['x'],features,rep) for e,d in hold.items()}
  for choice in [v for v in choices if v['representation']==rep]:
   model=joblib.load(R/choice['artifact']);np.save(O/(choice['identity']+'_input.npy'),hx['unsw_future'][:1000]);np.save(O/(choice['identity']+'_raw_input.npy'),hold['unsw_future']['x'][:1000])
   for env,d in hold.items():
    scores=score(model,hx[env]);rank=ranking_metrics(d['y'],scores)
    for rule,selected in choice['selected'].items():
     threshold=selected['thresholds']['unsw' if env.startswith('unsw') else 'ton'];out.append(dict(identity=choice['identity'],representation=rep,mixture=choice['mixture'],method=choice['method'],seed=choice['seed'],environment=env,policy=rule,ranking=rank,**metrics(d['y'],scores,threshold,d['family'],windows=20)))
   (O/'results.partial.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 (O/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2),encoding='utf-8');print('COMPLETE ranking objective',len(out),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
