"""Exploratory conventional multivariate anomaly control, not frozen primary.

The elementary upper-tail profile failed on nearby development data. Test a
compact Isolation Forest without attack training labels, before stress scoring.
No protected rows. Equal three-fit candidate budget and identical calibration.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,json,datetime
from time import process_time
import numpy as np,joblib
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits
from run_behavioral_comparison import R,D,U,O,load_part,inputs
from run_network_context_study import choose,slice_data
from run_multienvironment_training import metrics,ranking_metrics,ton_temporal_boundaries
from run_tolerance_threshold_audit import tolerance_threshold
B=O/'isolation_forest_control'

def score(m,x):return -m.score_samples(np.ascontiguousarray(x,dtype=np.float32))

def evaluate(stress=False):
    choices=json.loads((B/'selections.json').read_text());tune=load_part(D,'tune','ton');_,b=ton_temporal_boundaries()
    hold={'ton_stress':load_part(D,'evaluation','ton')} if stress else {'ton_future':slice_data(tune,slice(b,None)),'unsw_future':load_part(U,'development_holdout','unsw')}
    result=[]
    for c in choices:
        cache={}
        for policy,s in c['selected'].items():
            if s['artifact'] not in cache:
                m=joblib.load(R/s['artifact']);cache[s['artifact']]={e:score(m,d['behavior'][:,:24]) for e,d in hold.items()}
            ps=cache[s['artifact']]
            for env,d in hold.items():
                threshold=s['thresholds']['shared' if policy.startswith('shared') else 'unsw' if env.startswith('unsw') else 'ton']
                result.append(dict(identity=c['identity'],seed=c['seed'],method='isolation_forest_behavior24',environment=env,policy=policy,
                    **metrics(d['y'],ps[env],threshold,d['family'],windows=20),ranking=ranking_metrics(d['y'],ps[env])))
    (B/('stress_results.json' if stress else 'nearby_results.json')).write_text(json.dumps(dict(results=result,scope=__doc__),indent=2))
    print('COMPLETE isolation scorecards',len(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args();B.mkdir(exist_ok=True)
    if args.stress:evaluate(True);return
    protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,
        design='50trees,128samples per tree;24frozen causal behavioral features. No new features.',seeds=[42,43,44],fits_per_seed=3,
        training='Benign subset of exactly the primary200k pooled training rows; not fully unsupervised because benign labels are required.',
        thresholds='Same20k benign per network, tolerance alpha .001/delta .025; max-network shared control.',
        reason='Upper-tail profile failed nearby. Multivariate partitioning is a conventional alternative, not a new algorithm.',
        development_status='Added after nearby outcomes, before this control has any evaluation results; all three networks/periods are development, protected untouched.')
    if not (B/'protocol.json').exists():(B/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train,cal,selection=inputs();x=train['behavior'][train['y']==0,:24];choices=[]
    for seed in protocol['seeds']:
        identity=f'isolation_seed{seed}';record=B/(identity+'_selection.json')
        if record.exists():choices.append(json.loads(record.read_text()));continue
        candidates=[]
        for step in range(3):
            path=B/(identity+f'_step{step}.joblib');cpu=None
            if path.exists():m=joblib.load(path)
            else:
                m=IsolationForest(n_estimators=50,max_samples=128,n_jobs=1,random_state=seed+100*step)
                t=process_time();m.fit(x);cpu=process_time()-t;joblib.dump(m,path)
            ts={e:tolerance_threshold(score(m,d['behavior'][:,:24]),.001,.025)['threshold'] for e,d in cal.items()};ts['shared']=max(ts.values())
            ps={e:score(m,d['behavior'][:,:24]) for e,d in selection.items()}
            validation={scope:{e:metrics(d['y'],ps[e],ts['shared'] if scope=='shared' else ts[e],d['family']) for e,d in selection.items()} for scope in ['shared','network']}
            candidates.append(dict(step=step,artifact=str(path.relative_to(R)),bytes=path.stat().st_size,fit_cpu_s=cpu,thresholds=ts,validation=validation))
            print('isolation fit',seed,step,'CPU',cpu,flush=True)
        choices.append(dict(identity=identity,seed=seed,candidates=candidates,selected={p:choose(candidates,*p.split('_')) for p in ['shared_pooled','shared_window','network_pooled','network_window']}))
        record.write_text(json.dumps(choices[-1],indent=2))
    (B/'selections.json').write_text(json.dumps(choices,indent=2));evaluate()
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
