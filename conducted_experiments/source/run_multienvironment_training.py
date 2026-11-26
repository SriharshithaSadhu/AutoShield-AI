"""Fixed-capacity multi-environment training comparison, with sealed protected suffix.

All methods receive the same two-network benign calibration and validation data;
this isolates training exposure, not zero-shot deployment. Internal candidate
budgets are three fits per method/representation/seed. Extra training cost of
reweighting is measured. No protected labels are loaded by this executable.
"""
from pathlib import Path
from time import perf_counter,process_time
import json,io,hashlib,datetime
import numpy as np,joblib,psutil
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from multienvironment_features import transform,PHYSICAL_NAMES
from run_tolerance_threshold_audit import tolerance_threshold
R=Path(__file__).resolve().parent;O=R/'results/multienvironment_2026-11-21';D=R/'data/nf_ton_iot_v3/detection_foundation';U=R/'data/nf_unsw_nb15_v3/development'
METHODS=['single_ton','single_unsw','pooled','environment_balanced','family_balanced','group_dro','hard_negative'];SEEDS=[42,43,44];REPRESENTATIONS=['raw46','physical14']

def read(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def ton_temporal_boundaries():
 t=np.load(D/'tune_completion_ns.npy')
 cal=int(np.searchsorted(t,t[59999],side='right'));selection=int(np.searchsorted(t,t[129999],side='right'))
 assert t[cal-1]<t[cal] and t[selection-1]<t[selection]
 return cal,selection

def weights(groups):
 _,inv,count=np.unique(groups,return_inverse=True,return_counts=True);w=len(groups)/(len(count)*count[inv]);return w/w.mean(),inv,count

def metrics(y,score,threshold,family,windows=3):
 pred=score>threshold;tp=int(pred[y==1].sum());fp=int(pred[y==0].sum());a=int(y.sum());b=int((y==0).sum());blocks=[]
 for ix in np.array_split(np.arange(len(y)),windows):
  yy=y[ix];blocks.append(dict(benign=int((yy==0).sum()),attacks=int(yy.sum()),fp=int(pred[ix][yy==0].sum()),tp=int(pred[ix][yy==1].sum())))
 return dict(tp=tp,fp=fp,attacks=a,benign=b,recall=tp/a if a else None,fpr=fp/b if b else None,precision=tp/max(1,tp+fp),f1=2*tp/max(1,tp+fp+a),worst_window_fpr=max(v['fp']/v['benign'] for v in blocks if v['benign']),worst_window_recall=min((v['tp']/v['attacks'] for v in blocks if v['attacks']),default=0),windows=blocks,family={str(f):dict(n=int((family==f).sum()),recall=float(pred[family==f].mean())) for f in np.unique(family[y==1])})
def ranking_metrics(y,score):
 order=np.argsort(-score,kind='stable');s=score[order];yy=y[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];tp=np.cumsum(yy)[ends];fp=ends+1-tp;out={}
 for cap in (.001,.01):
  valid=fp<=np.floor(cap*(y==0).sum());out[str(cap)]=float(tp[valid].max()/y.sum()) if valid.any() else 0.
 return dict(roc_auc=float(roc_auc_score(y,score)),partial_roc_auc_001=float(roc_auc_score(y,score,max_fpr=.001)),average_precision=float(average_precision_score(y,score)),posthoc_threshold_oracle_recall=out,oracle_scope='Undeployable label-assisted ranking diagnostic; never used to choose thresholds or architecture.')

def main():
 O.mkdir(exist_ok=True);ton_cal_stop,ton_select_stop=ton_temporal_boundaries();features=json.loads((D/'manifest.json').read_text())['features']
 protocol=dict(ton_tune_sample_boundaries=[0,ton_cal_stop,ton_select_stop,200000],created_at=datetime.datetime.now().astimezone().isoformat(),methods=METHODS,seeds=SEEDS,representations=REPRESENTATIONS,rounds=3,training_rows=200000,target_budget_controls=dict(single_unsw_matched50k=50000,single_ton_matched150k=150000),pool_rows=dict(ton=150000,unsw=50000),calibration_benign_per_environment=20000,fpr_cap=.001,minimum_recall=.5,model_size_limit=1572864,inference_p95_ms_limit=5,memory_resident_limit=536870912,model=dict(n_estimators=100,num_leaves=15,min_child_samples=100,reg_lambda=10,subsample=.8,subsample_freq=1,colsample_bytree=.9),robust_eta=.5,group_mass_multiplier_cap=20,family_sample_weight_cap=20,hard_negative_multiplier=4,calibration_policy='primary=max two environment tolerance thresholds with delta .025 each; secondary=equal-size pooled tolerance threshold delta .05',protected_policy='UNSW final chronological 20% is never loaded. Only open in a separate frozen run if development joint operating-point and resources pass. NF final-period rows were inspected previously and remain a development stress test.',reference_policy='March1 corrected CSE and original grouped benchmarks unchanged; different exporter/protocol, not pooled or numerically ranked against this study.',scope=__doc__)
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 ton=read(D/'train.npz');unsw=read(U/'train.npz');tune=read(D/'tune.npz');uc=read(U/'calibration.npz');us=read(U/'selection.npz')
 # Exact row identities are shared by methods and seeds, chosen without selection/holdout labels.
 rng=np.random.default_rng(20261121);it=np.sort(rng.choice(len(ton['y']),150000,replace=False));iu=np.sort(rng.choice(len(unsw['y']),50000,replace=False))
 tc={k:v[:ton_cal_stop] for k,v in tune.items()};ts={k:v[ton_cal_stop:ton_select_stop] for k,v in tune.items()}
 raw_train=dict(single_ton=ton,single_unsw=unsw)
 pooled=dict(x=np.concatenate((ton['x'][it],unsw['x'][iu])),y=np.r_[ton['y'][it],unsw['y'][iu]],family=np.r_[ton['family'][it],unsw['family'][iu]],env=np.r_[np.zeros(len(it),dtype=int),np.ones(len(iu),dtype=int)])
 for m in METHODS[2:]:raw_train[m]=pooled
 if 'single_unsw_matched50k' in METHODS:raw_train['single_unsw_matched50k']={k:v[iu] for k,v in unsw.items()}
 if 'single_ton_matched150k' in METHODS:raw_train['single_ton_matched150k']={k:v[it] for k,v in ton.items()}
 calibrations={};selections={'ton':ts,'unsw':us}
 for name,d in [('ton',tc),('unsw',uc)]:
  ix=np.flatnonzero(d['y']==0)
  if len(ix)<20000:raise ValueError(f'{name} insufficient fixed benign calibration budget: {len(ix)}')
  ii=np.sort(rng.choice(ix,20000,replace=False));calibrations[name]=d['x'][ii]
 data_audit=dict(training={k:dict(rows=len(v['y']),attacks=int(v['y'].sum()),families=dict(zip(*[z.tolist() for z in np.unique(v['family'],return_counts=True)]))) for k,v in [('ton',ton),('unsw',unsw),('pooled',pooled)]},pool_ton_row_sha256=hashlib.sha256(ton['row'][it].tobytes()).hexdigest(),pool_unsw_row_sha256=hashlib.sha256(unsw['row'][iu].tobytes()).hexdigest(),note='NF training sample was originally balanced; per-method global or joint environment/class weights control effective class priors. Family taxonomies remain source-namespaced; no identical-name semantic merging.')
 (O/'training_audit.json').write_text(json.dumps(data_audit,indent=2),encoding='utf-8');choices=[]
 for representation in protocol['representations']:
  cc={k:transform(v,features,representation) for k,v in calibrations.items()};ss={k:transform(v['x'],features,representation) for k,v in selections.items()}
  for method in METHODS:
   d=raw_train[method];x=transform(d['x'],features,representation);y=d['y'];family=d['family'];env=d.get('env',np.zeros(len(y),dtype=int));joint=np.asarray([f'{e}:{v}' for e,v in zip(env,y)]);family_groups=np.asarray([f'{e}:{f}' for e,f in zip(env,family)])
   if len(np.unique(y))<2:raise ValueError(f'{method} lacks attack training class')
   global_weights,_,_=weights(y);joint_weights,group_index,group_counts=weights(joint);family_weights,_,_=weights(family_groups)
   for seed in SEEDS:
    identity=f'{representation}_{method}_seed{seed}';done=O/(identity+'_selection.json')
    if done.exists():choices.append(json.loads(done.read_text()));continue
    candidates=[];q=np.ones(len(group_counts))/len(group_counts);current=joint_weights.copy()
    for step in range(3):
     if method in ['single_ton','single_unsw','pooled','single_unsw_matched50k','single_ton_matched150k']:w=global_weights.copy()
     elif method=='environment_balanced':w=joint_weights.copy()
     elif method=='family_balanced':w=np.minimum(family_weights,20);w/=w.mean()
     elif method=='group_dro':w=joint_weights*np.minimum(q[group_index]*len(group_counts),20);w/=w.mean()
     else:w=current.copy()
     m=LGBMClassifier(**protocol['model'],n_jobs=1,verbosity=-1,random_state=seed+100*step)
     before=process_time();m.fit(x,y,sample_weight=w);fit_cpu=process_time()-before
     cal_scores={e:m.booster_.predict(v,num_threads=1) for e,v in cc.items()}
     max_threshold=max(tolerance_threshold(s,.001,.025)['threshold'] for s in cal_scores.values());pooled_threshold=tolerance_threshold(np.concatenate(list(cal_scores.values())),.001,.05)['threshold']
     validation={e:metrics(selections[e]['y'],m.booster_.predict(xx,num_threads=1),max_threshold,selections[e]['family']) for e,xx in ss.items()}
     b=io.BytesIO();joblib.dump(m,b);timings=[]
     for i in range(50):
      start=perf_counter();m.booster_.predict(x[:1],num_threads=1);dt=(perf_counter()-start)*1000
      if i>=10:timings.append(dt)
     rec=dict(step=step,seed=seed,method=method,representation=representation,validation=validation,thresholds=dict(max_environment=max_threshold,pooled=pooled_threshold),bytes=len(b.getvalue()),p95_ms=float(np.quantile(timings,.95)),fit_cpu_s=fit_cpu,weight_min=float(w.min()),weight_max=float(w.max()),weight_mean=float(w.mean()))
     rec['worst_fpr']=max(v['worst_window_fpr'] for v in validation.values());rec['worst_recall']=min(v['worst_window_recall'] for v in validation.values())
     artifact=O/(identity+f'_step{step}.joblib');joblib.dump(m,artifact);rec['artifact']=str(artifact.relative_to(R));candidates.append(rec)
     # Reweight using training labels/scores only, never validation/holdout labels.
     if method in ('group_dro','hard_negative'):
      p=np.clip(m.booster_.predict(x,num_threads=1),1e-7,1-1e-7)
      if method=='group_dro':
       loss=-(y*np.log(p)+(1-y)*np.log1p(-p));group_loss=np.bincount(group_index,weights=loss)/group_counts;q*=np.exp(.5*group_loss);q/=q.sum();rec['training_group_loss']=group_loss.tolist();rec['next_group_mass']=q.tolist()
      else:current=joint_weights*np.where(y==0,1+4*p,1);current/=current.mean()
     print(identity,step,'selection',rec['worst_fpr'],rec['worst_recall'],flush=True)
    eligible=[v for v in candidates if v['bytes']<=1572864 and v['p95_ms']<=5]
    chosen=max(eligible or candidates,key=lambda v:(v['worst_fpr']<=.001,-max(0,v['worst_fpr']-.001),v['worst_recall'],-v['step']))
    choice=dict(identity=identity,method=method,representation=representation,seed=seed,selected=chosen,candidates=candidates,development_operating_point_feasible=bool(eligible) and chosen['worst_fpr']<=.001 and chosen['worst_recall']>=.5,memory_measured=False)
    done.write_text(json.dumps(choice,indent=2),encoding='utf-8');choices.append(choice)
 # Freeze every choice before opening held-out development outcome arrays.
 (O/'selections.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
 holdouts={'ton_future':{k:v[ton_select_stop:] for k,v in tune.items()},'unsw_future':read(U/'development_holdout.npz'),'ton_stress':read(D/'evaluation.npz')}
 results=[]
 for representation in protocol['representations']:
  hx={k:transform(v['x'],features,representation) for k,v in holdouts.items()}
  for choice in [v for v in choices if v['representation']==representation]:
   selected=choice['selected'];model=joblib.load(R/selected['artifact']);np.save(O/(choice['identity']+'_input.npy'),hx['unsw_future'][:1000]);np.save(O/(choice['identity']+'_raw_input.npy'),holdouts['unsw_future']['x'][:1000]);joblib.dump(model,O/(choice['identity']+'_selected.joblib'))
   for env,d in holdouts.items():
    score=model.booster_.predict(hx[env],num_threads=1);rank=ranking_metrics(d['y'],score)
    for policy,threshold in selected['thresholds'].items():
     result=dict(identity=choice['identity'],method=choice['method'],representation=representation,seed=choice['seed'],environment=env,policy=policy,ranking=rank,**metrics(d['y'],score,threshold,d['family'],windows=20));results.append(result)
   (O/'results.partial.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 (O/'results.json').write_text(json.dumps(dict(protocol=protocol,results=results),indent=2),encoding='utf-8');print('COMPLETE',len(choices),'choices',len(results),'held-out operating points',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
