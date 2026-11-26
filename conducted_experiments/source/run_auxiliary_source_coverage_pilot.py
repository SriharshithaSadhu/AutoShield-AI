"""Auxiliary-source coverage pilot: fixed classifiers, matched source ablations.
CSE flows must precede 27 Feb2018; March1 reference capture remains excluded.
One fixed fit per condition/seed; two-source references reuse stage0 fits.
"""
from pathlib import Path
import json,datetime,numpy as np,joblib,pandas as pd
from time import process_time
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,D,U,read,metrics,ranking_metrics,ton_temporal_boundaries
from multienvironment_features import transform
from run_tolerance_threshold_audit import tolerance_threshold
ROOT=R/'results/multienvironment_2026-11-21';O=ROOT/'source_coverage_pilot';S=R/'data/nf_cicids2018_v3/permitted_source'
def balanced_pick(indices,family,n,rng):
 groups=[indices[family[indices]==f] for f in np.unique(family[indices])];quota=np.zeros(len(groups),dtype=int);left=n
 while left:
  active=[i for i,g in enumerate(groups) if quota[i]<len(g)]
  if not active:raise ValueError('not enough distinct permitted source rows')
  share=max(1,left//len(active))
  for i in active:
   take=min(share,len(groups[i])-quota[i],left);quota[i]+=take;left-=take
 return np.sort(np.concatenate([rng.choice(g,int(q),False) for g,q in zip(groups,quota) if q]))
def balanced_environment_class(env,y):
 w=np.zeros(len(y),float)
 for label in [0,1]:
  sources=np.unique(env[y==label])
  for source in sources:
   mask=(y==label)&(env==source);w[mask]=len(y)/(2*len(sources)*mask.sum())
 return w/w.mean()
def main():
 O.mkdir(exist_ok=True);audit=json.loads((S/'audit.json').read_text());assert audit['payload_sha1_verified'] and audit['last_ms']<audit['cutoff_ms'];assert audit['first_ms']>=int(pd.Timestamp('2018-02-01T00:00:00Z').value//1000000)
 methods=['two_source_reference','three_source_full','three_source_without_related_attacks','three_source_benign_only'];reps=['raw46','harmonized14'];seeds=[42,43,44]
 protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,methods=methods,representations=reps,seeds=seeds,training_rows=200000,source_rows=dict(ton=100000,unsw=50000,cse=50000),control='Full/restricted CSE cases share exact ToN, UNSW and 25000 CSE benign rows. Only the 25000 CSE attack-source rows differ; source-family capture/time confounding remains possible.',excluded_family_keywords=['brute','web','xss','injection','ftp','ssh'],calibration_targets=[.001,.0005,.00025],evaluation_fpr_cap=.001,model_capacity='100 trees,15 leaves,min_child100,lambda10; same stage0 seed as two-source references',protected_labels_used=False)
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');features=json.loads((D/'manifest.json').read_text())['features'];t=read(D/'train.npz');u=read(U/'train.npz');c=read(S/'source.npz');tune=read(D/'tune.npz');uc=read(U/'calibration.npz');us=read(U/'selection.npz');calstop,selstop=ton_temporal_boundaries()
 rng=np.random.default_rng(20261121);it=np.sort(rng.choice(len(t['y']),150000,False));iu=np.sort(rng.choice(len(u['y']),50000,False));core=np.sort(np.random.default_rng(20261122).choice(it,100000,False));tc={k:v[:calstop] for k,v in tune.items()};selection={'ton':{k:v[calstop:selstop] for k,v in tune.items()},'unsw':us};cal={}
 for name,d in [('ton',tc),('unsw',uc)]:
  ii=np.sort(rng.choice(np.flatnonzero(d['y']==0),20000,False));cal[name]=d['x'][ii]
 rr=np.random.default_rng(20261123);benign=rr.choice(np.flatnonzero(c['y']==0),50000,False);bf=benign[:25000];families=c['family'];attack=np.flatnonzero(c['y']==1);excluded=np.array([any(k in f.lower() for k in protocol['excluded_family_keywords']) for f in families]);restricted=attack[~excluded[attack]]
 assert excluded[attack].any(),'related-family ablation would be empty'
 full=balanced_pick(attack,families,25000,rr);limited=balanced_pick(restricted,families,25000,rr);indices={'three_source_full':np.r_[bf,full],'three_source_without_related_attacks':np.r_[bf,limited],'three_source_benign_only':benign};source_audit={}
 for name,ix in indices.items():source_audit[name]=dict(rows=len(ix),attack_rows=int(c['y'][ix].sum()),families={str(k):int(v) for k,v in zip(*np.unique(families[ix],return_counts=True))})
 (O/'source_ablation_audit.json').write_text(json.dumps(source_audit,indent=2),encoding='utf-8');choices=[]
 for rep in reps:
  cx={e:transform(x,features,rep) for e,x in cal.items()};sx={e:transform(d['x'],features,rep) for e,d in selection.items()}
  for name in methods:
   if name!='two_source_reference':
    ix=indices[name];raw=np.r_[t['x'][core],u['x'][iu],c['x'][ix]];y=np.r_[t['y'][core],u['y'][iu],c['y'][ix]];env=np.r_[np.zeros(len(core),int),np.ones(len(iu),int),np.full(len(ix),2)];x=transform(raw,features,rep);w=balanced_environment_class(env,y)
   for seed in seeds:
    identity=f'{rep}_{name}_seed{seed}';target=O/(identity+'_selection.json')
    if target.exists():choices.append(json.loads(target.read_text()));continue
    before=process_time()
    if name=='two_source_reference':
     folder=ROOT/'harmonization' if rep=='harmonized14' else ROOT;model=joblib.load(folder/f'{rep}_environment_balanced_seed{seed}_step0.joblib');fit_cpu=0
    else:model=LGBMClassifier(n_estimators=100,num_leaves=15,min_child_samples=100,reg_lambda=10,subsample=.8,subsample_freq=1,colsample_bytree=.9,n_jobs=1,verbosity=-1,random_state=seed).fit(x,y,sample_weight=w);fit_cpu=process_time()-before
    path=O/(identity+'_selected.joblib');joblib.dump(model,path);cs={e:model.booster_.predict(xx,num_threads=1) for e,xx in cx.items()};vs={e:model.booster_.predict(xx,num_threads=1) for e,xx in sx.items()};candidates=[]
    for alpha in protocol['calibration_targets']:
     threshold={e:tolerance_threshold(z,alpha,.025)['threshold'] for e,z in cs.items()}
     if any(v is None for v in threshold.values()):continue
     val={e:metrics(d['y'],vs[e],threshold[e],d['family']) for e,d in selection.items()};candidates.append(dict(alpha=alpha,thresholds=threshold,validation=val,worst_fpr=max(v['worst_window_fpr'] for v in val.values()),worst_recall=min(v['worst_window_recall'] for v in val.values())))
    chosen=max(candidates,key=lambda v:(v['worst_fpr']<=.001,-max(0,v['worst_fpr']-.001),v['worst_recall'],-v['alpha']));choice=dict(identity=identity,method=name,representation=rep,seed=seed,artifact=str(path.relative_to(R)),selected=chosen,candidates=candidates,fit_cpu_s=fit_cpu);target.write_text(json.dumps(choice,indent=2),encoding='utf-8');choices.append(choice);print('coverage choice',identity,chosen['alpha'],chosen['worst_fpr'],chosen['worst_recall'],flush=True)
 (O/'selections.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
 hold={'ton_future':{k:v[selstop:] for k,v in tune.items()},'unsw_future':read(U/'development_holdout.npz'),'ton_stress':read(D/'evaluation.npz')};out=[]
 for rep in reps:
  hx={e:transform(d['x'],features,rep) for e,d in hold.items()}
  for choice in [v for v in choices if v['representation']==rep]:
   m=joblib.load(R/choice['artifact']);np.save(O/(choice['identity']+'_input.npy'),hx['unsw_future'][:1000]);np.save(O/(choice['identity']+'_raw_input.npy'),hold['unsw_future']['x'][:1000])
   for env,d in hold.items():
    score=m.booster_.predict(hx[env],num_threads=1);threshold=choice['selected']['thresholds']['unsw' if env.startswith('unsw') else 'ton'];out.append(dict(identity=choice['identity'],method=choice['method'],representation=rep,seed=choice['seed'],environment=env,policy='coverage_pilot',ranking=ranking_metrics(d['y'],score),**metrics(d['y'],score,threshold,d['family'],windows=20)))
   (O/'results.partial.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 (O/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2),encoding='utf-8');print('COMPLETE source coverage pilot',len(out),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
