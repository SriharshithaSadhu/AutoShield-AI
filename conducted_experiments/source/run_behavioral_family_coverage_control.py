"""Development-only DDoS-family fitting exclusion, matched sample-count control.

Tests coverage effects, not a zero-day guarantee. Other network attack names
are not harmonized and can include related techniques. All observation history
is label-free and unchanged; removing fitting labels does not erase traffic.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,datetime,json,hashlib
from time import process_time
import numpy as np,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_behavioral_comparison import R,D,U,O,OLD,inputs,load_part,representation
from run_network_context_study import slice_data
from run_multienvironment_training import weights,metrics,ranking_metrics,ton_temporal_boundaries
from run_tolerance_threshold_audit import tolerance_threshold
from behavioral_models import scores
B=O/'family_coverage_control'

def evaluate(stress=False):
    f=json.loads((D/'manifest.json').read_text())['features'];tune=load_part(D,'tune','ton');_,b=ton_temporal_boundaries()
    hold={'ton_stress':load_part(D,'evaluation','ton')} if stress else {'ton_future':slice_data(tune,slice(b,None)),'unsw_future':load_part(U,'development_holdout','unsw')};out=[]
    for c in json.loads((B/'selections.json').read_text()):
        m=joblib.load(R/c['artifact'])
        for env,d in hold.items():
            s=scores(m,representation(d,c['method'],f));out.append(dict(identity=c['identity'],condition=c['condition'],method=c['method'],seed=c['seed'],environment=env,
                **metrics(d['y'],s,c['thresholds']['shared'],d['family'],windows=20),ranking=ranking_metrics(d['y'],s)))
    (B/('stress_results.json' if stress else 'nearby_results.json')).write_text(json.dumps(dict(scope=__doc__,results=out),indent=2));print('COMPLETE family-coverage controls',len(out),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args();B.mkdir(exist_ok=True)
    if args.stress:evaluate(True);return
    train,cal,_=inputs();withheld=(train['env']==0)&(train['y']==1)&(train['family']=='ddos');count=int(withheld.sum());assert count==30212
    keep=np.flatnonzero(~withheld);rng=np.random.default_rng(20261123);drop=rng.choice(np.flatnonzero((train['env']==0)&(train['y']==1)),count,False)
    mask=np.ones(len(withheld),bool);mask[drop]=False;control=np.flatnonzero(mask)
    conditions={'withhold_ton_ddos':keep,'matched_coverage_control':control}
    protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,seeds=[42,43,44],methods=['flow14','hybrid38'],
        rationale='Separate family coverage from feature effects and training quantity; matched comparator removes the same number of ToN attacks at random.',
        fitted_rows=int(len(keep)),removed_rows=count,identical_env_class_counts=True,fits_per_configuration=1,
        calibration='Exact primary20k benign per network; max tolerance threshold; no holdout threshold tuning.',
        status='Exploratory control added after nearby primary outcomes. Not a preregistered primary superiority test.')
    if not (B/'protocol.json').exists():(B/'protocol.json').write_text(json.dumps(protocol,indent=2))
    f=json.loads((D/'manifest.json').read_text())['features'];params=json.loads((OLD/'protocol.json').read_text())['model'];choices=[]
    audits=[]
    for condition,ix in conditions.items():
        d=slice_data(train,ix);w,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(d['env'],d['y'])]));counts={str((e,y)):int(((d['env']==e)&(d['y']==y)).sum()) for e in [0,1] for y in [0,1]}
        audits.append(dict(condition=condition,row_index_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),env_class_counts=counts,
            ton_attack_families={str(fam):int(((d['env']==0)&(d['family']==fam)&(d['y']==1)).sum()) for fam in np.unique(d['family'][(d['env']==0)&(d['y']==1)])}))
        for method in protocol['methods']:
            x=representation(d,method,f);cx={e:representation(z,method,f) for e,z in cal.items()}
            for seed in protocol['seeds']:
                identity=f'{condition}_{method}_seed{seed}';path=B/(identity+'.joblib');record=B/(identity+'_selection.json');cpu=None
                if record.exists():choices.append(json.loads(record.read_text()));continue
                if path.exists():m=joblib.load(path)
                else:
                    m=LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=seed);t=process_time();m.fit(x,d['y'],sample_weight=w);cpu=process_time()-t;joblib.dump(m,path)
                ts={e:tolerance_threshold(scores(m,z),.001,.025)['threshold'] for e,z in cx.items()};ts['shared']=max(ts.values())
                c=dict(identity=identity,condition=condition,method=method,seed=seed,artifact=str(path.relative_to(R)),fit_cpu_s=cpu,thresholds=ts)
                record.write_text(json.dumps(c,indent=2));choices.append(c);print('family-coverage fit',identity,'CPU',cpu,flush=True)
    assert audits[0]['env_class_counts']==audits[1]['env_class_counts']
    (B/'training_audit.json').write_text(json.dumps(audits,indent=2));(B/'selections.json').write_text(json.dumps(choices,indent=2));evaluate()
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
