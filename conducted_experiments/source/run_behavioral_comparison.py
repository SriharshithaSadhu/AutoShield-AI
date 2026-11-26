"""Matched-budget conventional behavioral comparison; protected labels excluded."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
from pathlib import Path
from time import process_time
import json,datetime,hashlib
import numpy as np,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_network_context_study import R,D,U,OLD,part,slice_data,choose
from run_multienvironment_training import ton_temporal_boundaries,weights,metrics,ranking_metrics
from multienvironment_features import transform
from behavioral_models import QuantileTailAnomaly,scores
from run_tolerance_threshold_audit import tolerance_threshold
O=R/'results/behavioral_observation_2026-11-22'

def load_part(folder,name,source):
    d=part(folder,name);d['behavior']=np.load(O/f'{source}_{name}_behavior.npy');d['metadata']=np.load(O/f'{source}_{name}_metadata.npy')
    assert np.array_equal(d['row'],d['metadata']['row'])
    if 'completion_ms' in d:assert np.array_equal(d['completion_ms'],d['metadata']['t'])
    else:assert np.array_equal(np.load(D/f'{name}_completion_ns.npy'),d['metadata']['t']*1000000)
    return d

def inputs():
    ton=load_part(D,'train','ton');unsw=load_part(U,'train','unsw');tune=load_part(D,'tune','ton');uc=load_part(U,'calibration','unsw');us=load_part(U,'selection','unsw')
    a,b=ton_temporal_boundaries();rng=np.random.default_rng(20261121);it=np.sort(rng.choice(len(ton['y']),150000,False));iu=np.sort(rng.choice(len(unsw['y']),50000,False))
    train={k:np.concatenate([ton[k][it],unsw[k][iu]]) for k in ['x','y','family','row','behavior']};train['env']=np.r_[np.zeros(len(it),int),np.ones(len(iu),int)]
    cal={}
    for env,d in [('ton',slice_data(tune,slice(0,a))),('unsw',uc)]:
        ix=np.sort(rng.choice(np.flatnonzero(d['y']==0),20000,False));cal[env]=slice_data(d,ix)
    selection={'ton':slice_data(tune,slice(a,b)),'unsw':us}
    old=json.loads((OLD/'training_audit.json').read_text())
    for env,d,ids in [('ton',ton,it),('unsw',unsw,iu)]:assert hashlib.sha256(d['row'][ids].tobytes()).hexdigest()==old[f'pool_{env}_row_sha256']
    return train,cal,selection

def representation(d,method,features):
    if method=='flow14':return transform(d['x'],features,'physical14')
    if method in ['behavior24','quantile_tail_anomaly']:return np.ascontiguousarray(d['behavior'][:,:24])
    if method=='behavior26_global_ablation':return np.ascontiguousarray(d['behavior'])
    if method=='hybrid38':return np.column_stack([transform(d['x'],features,'physical14'),d['behavior'][:,:24]])
    raise ValueError(method)

def evaluate_holdouts(choices,hold,outfile):
    features=json.loads((D/'manifest.json').read_text())['features'];result=[]
    for choice in choices:
        method=choice['method'];hx={e:representation(d,method,features) for e,d in hold.items()};cache={}
        for policy,c in choice['selected'].items():
            path=c['artifact']
            if path not in cache:
                model=joblib.load(R/path);cache[path]={e:scores(model,x) for e,x in hx.items()}
            for env,d in hold.items():
                s=cache[path][env];scope=policy.split('_')[0];threshold=c['thresholds']['shared' if scope=='shared' else 'unsw' if env.startswith('unsw') else 'ton']
                result.append(dict(identity=choice['identity'],method=method,seed=choice['seed'],environment=env,policy=policy,step=c['step'],
                    ranking=ranking_metrics(d['y'],s),**metrics(d['y'],s,threshold,d['family'],windows=20)))
        for policy,c in choice['selected'].items():
            np.savez_compressed(O/(choice['identity']+'_'+policy+'_scores.npz'),**cache[c['artifact']])
    outfile.write_text(json.dumps(dict(scope='Development outcomes only; all choices frozen before scoring.',results=result),indent=2));print('COMPLETE behavioral scorecards',len(result),flush=True)

def main():
    protocol=json.loads((O/'protocol.json').read_text());train,cal,selection=inputs();features=json.loads((D/'manifest.json').read_text())['features'];choices=[]
    w,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(train['env'],train['y'])]));params=json.loads((OLD/'protocol.json').read_text())['model']
    for method in protocol['methods']:
        x=representation(train,method,features);cx={e:representation(d,method,features) for e,d in cal.items()};sx={e:representation(d,method,features) for e,d in selection.items()}
        seeds=[42] if method=='quantile_tail_anomaly' else protocol['seeds']
        for seed in seeds:
            identity=f'{method}_seed{seed}';target=O/(identity+'_selection.json')
            if target.exists():choices.append(json.loads(target.read_text()));continue
            candidates=[]
            for step in range(1 if method=='quantile_tail_anomaly' else 3):
                path=O/(identity+f'_step{step}.joblib');cpu=0.;reused=method=='flow14'
                if reused:path=OLD/f'physical14_environment_balanced_seed{seed}_step{step}.joblib';m=joblib.load(path)
                elif path.exists():m=joblib.load(path);cpu=None
                elif method=='quantile_tail_anomaly':
                    start=process_time();m=QuantileTailAnomaly().fit(x[train['y']==0]);cpu=process_time()-start;joblib.dump(m,path)
                else:
                    m=LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=seed+100*step)
                    start=process_time();m.fit(x,train['y'],sample_weight=w);cpu=process_time()-start;joblib.dump(m,path)
                thresholds={e:tolerance_threshold(scores(m,z),.001,.025)['threshold'] for e,z in cx.items()}
                assert all(t is not None for t in thresholds.values());thresholds['shared']=max(thresholds.values())
                ps={e:scores(m,z) for e,z in sx.items()}
                validation={scope:{e:metrics(d['y'],ps[e],thresholds['shared'] if scope=='shared' else thresholds[e],d['family']) for e,d in selection.items()} for scope in ['shared','network']}
                candidates.append(dict(step=step,artifact=str(path.relative_to(R)),bytes=path.stat().st_size,
                    reused=reused,fit_cpu_s=cpu,thresholds=thresholds,validation=validation))
                print('behavioral fit',identity,step,'CPU',cpu,flush=True)
            selected={p:choose(candidates,*p.split('_')) for p in protocol['policies']}
            if method=='flow14':
                old=json.loads((OLD/f'physical14_environment_balanced_seed{seed}_selection.json').read_text())['selected']
                assert selected['shared_window']['step']==old['step'] and abs(selected['shared_window']['thresholds']['shared']-old['thresholds']['max_environment'])<1e-12
            c=dict(identity=identity,method=method,seed=seed,candidates=candidates,selected=selected)
            target.write_text(json.dumps(c,indent=2));choices.append(c)
    (O/'selections.json').write_text(json.dumps(choices,indent=2))
    # No stress cache exists until full replay; nearby holdout labels enter here.
    tune=load_part(D,'tune','ton');_,b=ton_temporal_boundaries()
    hold={'ton_future':slice_data(tune,slice(b,None)),'unsw_future':load_part(U,'development_holdout','unsw')}
    evaluate_holdouts(choices,hold,O/'nearby_results.json')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
