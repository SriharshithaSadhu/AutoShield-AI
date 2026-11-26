"""Exploratory source-only control:200k fitting rows, target-benign calibration.

This is transfer with labeled target benign calibration, not zero-shot IDS.
Fixed one fit per source/method/seed, no target attack-label model selection.
Added after nearby exploratory outcomes; protected data remain unused.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
from pathlib import Path
from time import process_time
import json,datetime,argparse
import numpy as np,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_behavioral_comparison import R,D,U,O,OLD,load_part,inputs,representation
from run_multienvironment_training import weights,metrics,ranking_metrics,ton_temporal_boundaries
from run_network_context_study import slice_data
from run_tolerance_threshold_audit import tolerance_threshold
from behavioral_models import scores
B=O/'source_transfer_control'

def evaluate(choices,stress=False):
    _,cal,_=inputs();f=json.loads((D/'manifest.json').read_text())['features'];tune=load_part(D,'tune','ton');_,b=ton_temporal_boundaries()
    hold={'ton_future':slice_data(tune,slice(b,None)),'unsw_future':load_part(U,'development_holdout','unsw')}
    if stress:hold={'ton_stress':load_part(D,'evaluation','ton')}
    out=[]
    for c in choices:
        model=joblib.load(R/c['artifact'])
        for env,d in hold.items():
            s=scores(model,representation(d,c['method'],f));threshold=c['thresholds']['unsw' if env.startswith('unsw') else 'ton']
            out.append(dict(identity=c['identity'],method=c['method'],training_source=c['source'],seed=c['seed'],environment=env,
                deployment_network_seen_in_fit=(c['source']=='ton')==env.startswith('ton'),
                calibration='20000 labeled target benign flows; fixed tolerance rule',
                ranking=ranking_metrics(d['y'],s),**metrics(d['y'],s,threshold,d['family'],windows=20)))
    (B/('stress_results.json' if stress else 'nearby_results.json')).write_text(json.dumps(dict(scope=__doc__,results=out),indent=2))
    print('COMPLETE source-transfer controls',len(out),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args();B.mkdir(exist_ok=True)
    if args.stress:evaluate(json.loads((B/'selections.json').read_text()),True);return
    protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,methods=['flow14','behavior24','hybrid38'],
        sources=['ton','unsw'],seeds=[42,43,44],fit_rows=200000,fits_per_case=1,calibration_benign_per_target=20000,
        warning='Additional development control; compare fixed-fit source exposure, not a fully matched selection-budget superiority claim.')
    (B/'protocol.json').write_text(json.dumps(protocol,indent=2));_,cal,_=inputs();f=json.loads((D/'manifest.json').read_text())['features'];params=json.loads((OLD/'protocol.json').read_text())['model'];choices=[]
    for source,folder in [('ton',D),('unsw',U)]:
        train=load_part(folder,'train',source);w,_,_=weights(train['y'])
        for method in protocol['methods']:
            x=representation(train,method,f);cx={e:representation(d,method,f) for e,d in cal.items()}
            for seed in protocol['seeds']:
                identity=f'{source}_{method}_seed{seed}';record=B/(identity+'_selection.json');path=B/(identity+'_selected.joblib')
                if record.exists():choices.append(json.loads(record.read_text()));continue
                if method=='flow14':path=OLD/f'physical14_single_{source}_seed{seed}_step0.joblib';m=joblib.load(path);cpu=0.
                elif path.exists():m=joblib.load(path);cpu=None
                else:
                    m=LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=seed);t=process_time();m.fit(x,train['y'],sample_weight=w);cpu=process_time()-t;joblib.dump(m,path)
                thresholds={e:tolerance_threshold(scores(m,z),.001,.025)['threshold'] for e,z in cx.items()};assert all(t is not None for t in thresholds.values())
                c=dict(identity=identity,source=source,method=method,seed=seed,artifact=str(path.relative_to(R)),fit_cpu_s=cpu,thresholds=thresholds)
                record.write_text(json.dumps(c,indent=2));choices.append(c);print('source-control fitted',identity,flush=True)
    (B/'selections.json').write_text(json.dumps(choices,indent=2));evaluate(choices)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
