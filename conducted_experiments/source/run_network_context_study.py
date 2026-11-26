"""Frozen contextual-feature ablations with identical training and label budgets.

Exploratory development study. No protected-period loader is provided.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
from pathlib import Path
from time import process_time
import json,datetime,hashlib
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,D,U,read,ton_temporal_boundaries,metrics,ranking_metrics,weights
from run_tolerance_threshold_audit import tolerance_threshold
from network_context_features import transform,VARIANTS,SERVICES,SERVICE_NAMES
O=R/'results/network_context_2026-11-21'
OLD=R/'results/multienvironment_2026-11-21'

def part(folder,name):
    d=read(folder/(name+'.npz'));d['ports']=np.load(folder/(name+'_ports.npy'))
    assert len(d['ports'])==len(d['y'])
    return d

def slice_data(d,ids):return {k:v[ids] for k,v in d.items()}

def development_inputs():
    ton=part(D,'train');unsw=part(U,'train');tune=part(D,'tune');uc=part(U,'calibration')
    us=part(U,'selection');a,b=ton_temporal_boundaries();rng=np.random.default_rng(20261121)
    it=np.sort(rng.choice(len(ton['y']),150000,False));iu=np.sort(rng.choice(len(unsw['y']),50000,False))
    train={k:np.concatenate([ton[k][it],unsw[k][iu]]) for k in ['x','y','ports','row','family']}
    train['env']=np.r_[np.zeros(len(it),int),np.ones(len(iu),int)]
    cal={}
    for name,d in [('ton',slice_data(tune,slice(0,a))),('unsw',uc)]:
        ii=np.sort(rng.choice(np.flatnonzero(d['y']==0),20000,False));cal[name]=slice_data(d,ii)
    selection={'ton':slice_data(tune,slice(a,b)),'unsw':us}
    audit=json.loads((OLD/'training_audit.json').read_text())
    for env,d,ids in [('ton',ton,it),('unsw',unsw,iu)]:
        assert hashlib.sha256(d['row'][ids].tobytes()).hexdigest()==audit[f'pool_{env}_row_sha256']
    return train,cal,selection

def holdouts():
    tune=part(D,'tune');_,b=ton_temporal_boundaries()
    return {'ton_future':slice_data(tune,slice(b,None)),
            'unsw_future':part(U,'development_holdout'),'ton_stress':part(D,'evaluation')}

def choose(candidates,scope,criterion):
    def key(c):
        v=c['validation'][scope]
        risk=max(z['fpr'] if criterion=='pooled' else z['worst_window_fpr'] for z in v.values())
        benefit=min(z['recall'] if criterion=='pooled' else z['worst_window_recall'] for z in v.values())
        return (risk<=.001,-max(0,risk-.001),benefit,-c['step'])
    return max(candidates,key=key)

def main():
    O.mkdir(exist_ok=True)
    old_protocol=json.loads((OLD/'protocol.json').read_text())
    protocol=dict(created_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,
        variants=VARIANTS,seeds=[42,43,44],rounds=3,model=old_protocol['model'],
        training_rows=200000,source_rows=dict(ton=150000,unsw=50000),
        weighting='Exact same environment-by-class inverse-frequency weights as original strongest baseline.',
        cap=.001,minimum_recall=.5,calibration_benign_per_network=20000,calibration_delta=.025,
        policies=['shared_pooled','shared_window','network_pooled','network_window'],
        services=SERVICES,service_names=SERVICE_NAMES,
        interpretation='Port-derived categories are fallible service hints, not verified application identity or labels.',
        input_scope='Transport ports and existing exporter fields only. No endpoint identity, raw port feature, clock, history or protected row IDs.',
        model_selection='Identical three fits per representation/seed. Original pooled and stricter window constraints reported separately; thresholds calibrated once on separate fixed benign data.',
        baseline='Reuse exact three fitted physical14/harmonized14 environment-balanced artifacts; new variants use same rows, weights and fit seeds.',
        frozen_code={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in ['network_context_features.py','run_network_context_study.py']})
    saved=O/'protocol.json'
    if saved.exists():
        previous=json.loads(saved.read_text());assert previous['frozen_code']==protocol['frozen_code'],'Feature/runner code changed after freeze; use a separate experiment folder.'
    else:saved.write_text(json.dumps(protocol,indent=2))
    train,cal,selection=development_inputs();features=json.loads((D/'manifest.json').read_text())['features']
    w,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(train['env'],train['y'])]))
    choices=[]
    for variant in VARIANTS:
        x=transform(train['x'],train['ports'],features,variant)
        cx={e:transform(d['x'],d['ports'],features,variant) for e,d in cal.items()}
        sx={e:transform(d['x'],d['ports'],features,variant) for e,d in selection.items()}
        for seed in protocol['seeds']:
            identity=f'{variant}_seed{seed}';target=O/(identity+'_selection.json')
            if target.exists():choices.append(json.loads(target.read_text()));continue
            candidates=[]
            for step in range(3):
                path=O/(identity+f'_step{step}.joblib');reused=variant in ['physical14','harmonized14']
                if reused:
                    folder=OLD/'harmonization' if variant=='harmonized14' else OLD
                    path=folder/f'{variant}_environment_balanced_seed{seed}_step{step}.joblib';m=joblib.load(path);cpu=0.
                elif path.exists():m=joblib.load(path);cpu=None
                else:
                    m=LGBMClassifier(**protocol['model'],n_jobs=1,verbosity=-1,random_state=seed+100*step)
                    before=process_time();m.fit(x,train['y'],sample_weight=w);cpu=process_time()-before;joblib.dump(m,path)
                cs={e:m.booster_.predict(z,num_threads=1) for e,z in cx.items()}
                thresholds={e:tolerance_threshold(z,.001,.025)['threshold'] for e,z in cs.items()}
                assert all(t is not None for t in thresholds.values())
                thresholds['shared']=max(thresholds.values())
                scores={e:m.booster_.predict(z,num_threads=1) for e,z in sx.items()}
                validation={scope:{e:metrics(d['y'],scores[e],thresholds['shared'] if scope=='shared' else thresholds[e],d['family']) for e,d in selection.items()} for scope in ['shared','network']}
                record=dict(step=step,artifact=str(path.relative_to(R)),fit_cpu_s=cpu,reused=reused,
                    bytes=path.stat().st_size,thresholds=thresholds,validation=validation)
                assert record['bytes']<=1572864,'Model size limit exceeded'
                candidates.append(record)
                print('context fit',identity,step,'CPU',cpu,flush=True)
            selected={p:choose(candidates,*p.split('_')) for p in protocol['policies']}
            if variant=='physical14':
                original=json.loads((OLD/f'physical14_environment_balanced_seed{seed}_selection.json').read_text())['selected']
                assert selected['shared_window']['step']==original['step']
                assert abs(selected['shared_window']['thresholds']['shared']-original['thresholds']['max_environment'])<1e-12
            choice=dict(identity=identity,variant=variant,seed=seed,selected=selected,candidates=candidates)
            target.write_text(json.dumps(choice,indent=2));choices.append(choice)
    # All feature/training/threshold decisions persist before outcome labels load.
    (O/'selections.json').write_text(json.dumps(choices,indent=2))
    hold=holdouts();out=[]
    for variant in VARIANTS:
        hx={e:transform(d['x'],d['ports'],features,variant) for e,d in hold.items()}
        for c in [z for z in choices if z['variant']==variant]:
            models={};scorecache={}
            for policy,selected in c['selected'].items():
                path=selected['artifact']
                if path not in models:models[path]=joblib.load(R/path)
                for env,d in hold.items():
                    key=(path,env)
                    if key not in scorecache:scorecache[key]=models[path].booster_.predict(hx[env],num_threads=1)
                    s=scorecache[key];scope=policy.split('_')[0];threshold=selected['thresholds']['shared' if scope=='shared' else 'unsw' if env.startswith('unsw') else 'ton']
                    out.append(dict(identity=c['identity'],variant=variant,seed=c['seed'],environment=env,policy=policy,selected_step=selected['step'],threshold=threshold,
                        ranking=ranking_metrics(d['y'],s),**metrics(d['y'],s,threshold,d['family'],windows=20)))
            for policy,selected in c['selected'].items():
                scores={e:scorecache[(selected['artifact'],e)] for e in hold}
                np.savez_compressed(O/(c['identity']+'_'+policy+'_scores.npz'),**scores)
            (O/'results.partial.json').write_text(json.dumps(out,indent=2))
    (O/'results.json').write_text(json.dumps(dict(protocol=protocol,results=out),indent=2))
    print('COMPLETE frozen context study',len(out),'operating points',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
