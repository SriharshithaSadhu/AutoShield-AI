"""Bounded deployment-assumption diagnostic, not another feature search.

After the frozen context ablations, test whether earlier target-period labels
help later target traffic under an explicit, expensive supervised warmup setting.
All results remain development; the parent period was previously inspected.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
import json,datetime,hashlib
from time import process_time
import numpy as np,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_network_context_study import R,D,U,O,OLD,development_inputs,part,slice_data
from run_multienvironment_training import metrics,ranking_metrics,weights
from run_tolerance_threshold_audit import tolerance_threshold
from network_context_features import transform
B=O/'prefix_coverage_control'

def main():
    B.mkdir(exist_ok=True);d=part(D,'evaluation');times=np.load(D/'evaluation_completion_ns.npy')
    bounds=[0]+[int(np.searchsorted(times,times[i-1],side='right')) for i in [100000,140000,160000]]+[len(times)]
    assert all(times[b-1]<times[b] for b in bounds[1:-1])
    a,b,c=bounds[1:4];protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
        scope=__doc__,boundaries=bounds,variants=['physical14','context29'],seeds=[42,43,44],
        rows=200000,target_train_labels=50000,
        original_sources=dict(ton=150000,unsw=50000),warmup_sources=dict(ton=100000,unsw=50000,target_prefix=50000),
        calibration_scopes=['original_shared','old_ton_matched','recent_ton_matched'],
        calibration_cap=.001,delta=.025,
        control='Same model configuration, one fixed fit per case/seed. Same fixed benign label count for old/recent ToN calibration. No tuning on the final target suffix.',
        limitation='Changed training information: 50000 earlier target labels plus full labels from the calibration prefix. Not an equal reviewer-budget improvement, independent validation, or new algorithm.')
    (B/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base,cal,_=development_inputs();rng=np.random.default_rng(20261121)
    old_ton=np.flatnonzero(base['env']==0);old_unsw=np.flatnonzero(base['env']==1)
    it=np.sort(rng.choice(old_ton,100000,False));target_ids=np.sort(rng.choice(np.arange(a),50000,False))
    warm={k:np.concatenate([base[k][it],base[k][old_unsw],d[k][target_ids]]) for k in ['x','y','ports','row','family']}
    warm['env']=np.r_[np.zeros(len(it),int),np.ones(len(old_unsw),int),np.full(len(target_ids),2)]
    recent=slice_data(d,slice(a,b));selection=slice_data(d,slice(b,c))
    benign_ids=np.flatnonzero(recent['y']==0);n=min(20000,len(benign_ids),len(cal['ton']['y']))
    recent_ids=np.sort(rng.choice(benign_ids,n,False));old_ids=np.sort(rng.choice(len(cal['ton']['y']),n,False))
    recent_cal=slice_data(recent,recent_ids);old_cal=slice_data(cal['ton'],old_ids)
    (B/'data_audit.json').write_text(json.dumps(dict(boundaries=bounds,matching_benign_calibration_rows=n,
        recent_period_total_labels=len(recent['y']),target_train_sample_row_sha256=hashlib.sha256(d['row'][target_ids].tobytes()).hexdigest(),
        chronological_strict=True,protected_rows_loaded=False,
        family_counts={name:{str(k):int(v) for k,v in zip(*np.unique(z['family'],return_counts=True))}
                       for name,z in [('warmup_training',warm),('recent_calibration_period',recent),('selection',selection)]}),indent=2))
    f=json.loads((D/'manifest.json').read_text())['features'];choices=[]
    for variant in protocol['variants']:
        cx={name:transform(z['x'],z['ports'],f,variant) for name,z in [('old_ton_matched',old_cal),('recent_ton_matched',recent_cal)]}
        for arm,train in [('source_only',base),('supervised_target_warmup',warm)]:
            x=transform(train['x'],train['ports'],f,variant);w,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(train['env'],train['y'])]))
            for seed in protocol['seeds']:
                identity=f'{variant}_{arm}_seed{seed}';path=B/(identity+'_selected.joblib')
                original=json.loads((O/f'{variant}_seed{seed}_selection.json').read_text())['candidates'][0]
                if arm=='source_only':m=joblib.load(R/original['artifact']);cpu=0.
                elif path.exists():m=joblib.load(path);cpu=None
                else:
                    params=json.loads((OLD/'protocol.json').read_text())['model'];m=LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=seed)
                    t=process_time();m.fit(x,train['y'],sample_weight=w);cpu=process_time()-t
                joblib.dump(m,path)
                thresholds={name:tolerance_threshold(m.booster_.predict(z,num_threads=1),.001,.025)['threshold'] for name,z in cx.items()}
                if arm=='source_only':thresholds['original_shared']=original['thresholds']['shared']
                else:
                    thresholds['original_shared']=max(tolerance_threshold(m.booster_.predict(transform(z['x'],z['ports'],f,variant),num_threads=1),.001,.025)['threshold'] for z in cal.values())
                choice=dict(identity=identity,variant=variant,arm=arm,seed=seed,artifact=str(path.relative_to(R)),fit_cpu_s=cpu,thresholds=thresholds)
                choices.append(choice)
                print('prefix diagnostic fitted',identity,'recent benign n',n,flush=True)
    (B/'selections.json').write_text(json.dumps(choices,indent=2));out=[]
    final=slice_data(d,slice(c,None))
    for choice in choices:
        model=joblib.load(R/choice['artifact']);x=transform(final['x'],final['ports'],f,choice['variant']);score=model.booster_.predict(x,num_threads=1)
        rank=ranking_metrics(final['y'],score)
        for policy,threshold in choice['thresholds'].items():
            if threshold is None:
                out.append(dict(identity=choice['identity'],policy=policy,feasible_calibration=False));continue
            out.append(dict(identity=choice['identity'],variant=choice['variant'],arm=choice['arm'],seed=choice['seed'],policy=policy,
                feasible_calibration=True,ranking=rank,**metrics(final['y'],score,threshold,final['family'],windows=20)))
    (B/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2))
    print('COMPLETE prefix coverage diagnostic',len(out),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
