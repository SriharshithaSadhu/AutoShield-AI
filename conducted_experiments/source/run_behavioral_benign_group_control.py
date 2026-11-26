"""Exploratory benign-service loss weighting; fixed data, architecture and calibration.

Motivation: false alarms cluster in legitimate service-field groups. Coarse
ports are training-weight strata only, never new classifier features. This is
conventional group balancing, not a new algorithm or independent validation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import datetime,json
from time import process_time
import numpy as np,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_behavioral_comparison import R,D,O,OLD,inputs,load_part,representation
from run_multienvironment_training import weights,metrics,ranking_metrics
from run_tolerance_threshold_audit import tolerance_threshold
from behavioral_models import scores
B=O/'benign_group_control'

def groups(protocol,port):
    g=np.full(len(port),'other_protocol',dtype='<U20');g[protocol==6]='tcp_other';g[protocol==17]='udp_other'
    for p in [20,21,22,23,25,53,80,110,143,443,445,3389]:g[(protocol==6)&(port==p)]=f'tcp_{p}'
    for p in [53,67,68,123,161,1900,5353]:g[(protocol==17)&(port==p)]=f'udp_{p}'
    return g

def main():
    B.mkdir(exist_ok=True);protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,seeds=[42,43,44],methods=['flow14','hybrid38'],
        groups='Fixed standard TCP/UDP service-port proxies, otherTCP, otherUDP, otherprotocol; rare groups below20 examples merged within each network.',
        weighting='Reciprocal benign group count, capped20; normalize mean1 per benign network. Original environment/class mass preserved.',
        fits_per_case=1,calibration='Exact original20k benign per network; max tolerance threshold, alpha .001/delta .025.',
        timing='Post-stress development response to observed failures; no protected labels or holdout tuning claims.')
    if not (B/'protocol.json').exists():(B/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train,cal,_=inputs();f=json.loads((D/'manifest.json').read_text())['features'];pi=f.index('PROTOCOL')
    ton=load_part(D,'train','ton');unsw=load_part(__import__('run_behavioral_comparison').U,'train','unsw');rng=np.random.default_rng(20261121)
    it=np.sort(rng.choice(len(ton['y']),150000,False));iu=np.sort(rng.choice(len(unsw['y']),50000,False));ports=np.r_[ton['metadata']['port'][it],unsw['metadata']['port'][iu]]
    assert np.array_equal(train['row'],np.r_[ton['row'][it],unsw['row'][iu]])
    g=groups(train['x'][:,pi],ports);base,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(train['env'],train['y'])]));multiplier=np.ones(len(base));audit=[]
    for env in [0,1]:
        ix=np.flatnonzero((train['env']==env)&(train['y']==0));labels=g[ix].copy();u,c=np.unique(labels,return_counts=True);rare=u[c<20];labels[np.isin(labels,rare)]='rare_combined'
        u,inv,count=np.unique(labels,return_inverse=True,return_counts=True);values=np.minimum(20.,len(ix)/(len(u)*count[inv]));values/=values.mean();multiplier[ix]=values
        audit.append(dict(environment=env,groups={str(k):int(v) for k,v in zip(u,count)},maximum_multiplier=float(values.max()),mean_multiplier=float(values.mean())))
    w=base*multiplier
    for env in [0,1]:
        for y in [0,1]:
            ix=(train['env']==env)&(train['y']==y);assert np.isclose(w[ix].sum(),base[ix].sum())
    (B/'training_audit.json').write_text(json.dumps(audit,indent=2));params=json.loads((OLD/'protocol.json').read_text())['model'];choices=[]
    for method in protocol['methods']:
        x=representation(train,method,f);cx={e:representation(d,method,f) for e,d in cal.items()}
        for seed in protocol['seeds']:
            identity=f'{method}_seed{seed}';path=B/(identity+'.joblib');cpu=None
            if path.exists():model=joblib.load(path)
            else:
                model=LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=seed);t=process_time();model.fit(x,train['y'],sample_weight=w);cpu=process_time()-t;joblib.dump(model,path)
            ts={e:tolerance_threshold(scores(model,z),.001,.025)['threshold'] for e,z in cx.items()};ts['shared']=max(ts.values())
            choices.append(dict(identity=identity,method=method,seed=seed,artifact=str(path.relative_to(R)),fit_cpu_s=cpu,thresholds=ts));print('BENIGN GROUP fitted',identity,flush=True)
    (B/'selections.json').write_text(json.dumps(choices,indent=2));stress=load_part(D,'evaluation','ton');out=[]
    for c in choices:
        model=joblib.load(R/c['artifact']);s=scores(model,representation(stress,c['method'],f));out.append(dict(**c,
            **metrics(stress['y'],s,c['thresholds']['shared'],stress['family'],windows=20),ranking=ranking_metrics(stress['y'],s)))
    (B/'stress_results.json').write_text(json.dumps(dict(scope=__doc__,results=out),indent=2));print('COMPLETE benign group control',len(out),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
