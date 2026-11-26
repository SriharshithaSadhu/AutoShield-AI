"""Exploratory recent-benign coverage control, after primary stress failures.

Original frozen results untouched. Fixed fit count/size and equal calibration
isolate fitting exposure from recalibration. Not independent confirmation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import datetime,json,pickle,hashlib
from time import perf_counter,process_time
import numpy as np,pandas as pd,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from behavioral_data import META_COLUMNS,META_DTYPE,metadata,event_columns
from run_behavioral_comparison import R,D,O,OLD,inputs,load_part,representation
from run_multienvironment_training import weights,metrics,ranking_metrics
from run_tolerance_threshold_audit import tolerance_threshold
from behavioral_models import scores
B=O/'recent_benign_control'

def prepare():
    if (B/'cache_audit.json').exists():return
    with np.load(D/'calibration.npz') as z:ids=z['row'].copy();assert (z['y']==0).all()
    wanted={int(r):i for i,r in enumerate(ids)};v=np.zeros((len(ids),26),np.float32);meta=np.empty(len(ids),META_DTYPE);seen=np.zeros(len(ids),bool)
    with (O/'nf_prefix_state.pkl').open('rb') as f:engine=pickle.load(f)
    boundaries=json.loads((D/'manifest.json').read_text())['boundaries'];start,stop=boundaries[2:4];offset=0;wall=perf_counter();processed=0
    for frame in pd.read_csv(D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv',usecols=META_COLUMNS,chunksize=100000):
        end=offset+len(frame)
        if end<=start:offset=end;continue
        take=frame.iloc[max(0,start-offset):min(len(frame),stop-offset)];base=max(start,offset);m=metadata(take,base)
        for i,e in enumerate(zip(*event_columns(m))):
            j=wanted.get(base+i);value=engine.observe(*e,query=j is not None)
            if j is not None:v[j]=(*value,*engine.global_features());meta[j]=m[i];seen[j]=True
        processed+=len(m);offset=end
        if end>=stop:break
    assert seen.all() and processed==stop-start
    np.save(B/'behavior.npy',v);np.save(B/'metadata.npy',meta)
    (B/'cache_audit.json').write_text(json.dumps(dict(processed_rows=processed,probes=len(ids),wall_s=perf_counter()-wall,state=engine.diagnostics(),
        labels_used_for_state=False,protected_network_processed=False),indent=2));print('RECENT BENIGN cache complete',round(perf_counter()-wall,2),flush=True)

def main():
    B.mkdir(exist_ok=True)
    protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,methods=['flow14','hybrid38'],seeds=[42,43,44],
        training='Exactly200k original pool; treatment replaces25k ToN benign rows with25k benign records from earlier half of the pre-stress gap. Same env/class counts.',
        calibration='20k benign records from later gap half plus the exact original20k UNSW benign calibration records; shared max tolerance threshold.',
        control='Original fixed step0 model, same new calibration. No model selection; one fixed fit per method/seed.',
        chronology='All added fitting observations strictly before new ToN calibration; calibration strictly before stress. Do not evaluate preceding ToN nearby period as prospective.',
        caveat='Post-stress development experiment; tests benign data exposure, not algorithmic novelty or independent validation.')
    if not (B/'protocol.json').exists():(B/'protocol.json').write_text(json.dumps(protocol,indent=2))
    prepare();train,oldcal,_=inputs()
    with np.load(D/'calibration.npz') as z:gap={k:z[k] for k in z.files}
    gap['behavior']=np.load(B/'behavior.npy');m=np.load(B/'metadata.npy');assert np.array_equal(gap['row'],m['row'])
    times=m['t'];middle=int(np.searchsorted(times,times[len(times)//2],side='right'));rng=np.random.default_rng(20261123)
    add=np.sort(rng.choice(middle,25000,False));ci=np.sort(rng.choice(np.arange(middle,len(times)),20000,False));drop=rng.choice(np.flatnonzero((train['env']==0)&(train['y']==0)),25000,False)
    assert times[add].max()<times[ci].min();assert m['row'][ci].max()<json.loads((D/'manifest.json').read_text())['boundaries'][-2]
    # Preserve every row position and therefore the per-seed subsampling map.
    # Only the selected benign observations change, not fitting row order.
    updated={k:v.copy() for k,v in train.items()}
    for k in updated:updated[k][drop]=0 if k=='env' else gap[k][add]
    assert len(updated['y'])==len(train['y'])==200000
    for e in [0,1]:
        for y in [0,1]:assert ((updated['env']==e)&(updated['y']==y)).sum()==((train['env']==e)&(train['y']==y)).sum()
    cal={'ton':{k:v[ci] for k,v in gap.items()},'unsw':oldcal['unsw']};f=json.loads((D/'manifest.json').read_text())['features'];params=json.loads((OLD/'protocol.json').read_text())['model'];choices=[]
    audit=dict(training_added_row_sha256=hashlib.sha256(gap['row'][add].tobytes()).hexdigest(),calibration_row_sha256=hashlib.sha256(gap['row'][ci].tobytes()).hexdigest(),
        latest_added_training_ms=int(times[add].max()),earliest_calibration_ms=int(times[ci].min()),latest_calibration_ms=int(times[ci].max()),
        gap_benign_destination_ports={str(int(p)):int((m['port'][add]==p).sum()) for p in [21,53,80,443]},fitted_rows=200000,env_class_counts_matched=True,
        replacement_in_place=True,training_row_position_and_subsampling_map_preserved=True)
    (B/'training_audit.json').write_text(json.dumps(audit,indent=2))
    for method in protocol['methods']:
        x=representation(updated,method,f);cx={e:representation(d,method,f) for e,d in cal.items()};w,_,_=weights(np.array([f'{e}:{y}' for e,y in zip(updated['env'],updated['y'])]))
        for seed in protocol['seeds']:
            for condition in ['original_fit_recent_calibration','recent_benign_replacement']:
                identity=f'{condition}_{method}_seed{seed}';path=OLD/f'physical14_environment_balanced_seed{seed}_step0.joblib' if condition.startswith('original') and method=='flow14' else O/f'hybrid38_seed{seed}_step0.joblib' if condition.startswith('original') else B/(identity+'.joblib');cpu=0.
                if path.exists():model=joblib.load(path)
                else:
                    model=LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=seed);t=process_time();model.fit(x,updated['y'],sample_weight=w);cpu=process_time()-t;joblib.dump(model,path)
                ts={e:tolerance_threshold(scores(model,z),.001,.025)['threshold'] for e,z in cx.items()};ts['shared']=max(ts.values())
                choices.append(dict(identity=identity,condition=condition,method=method,seed=seed,artifact=str(path.relative_to(R)),fit_cpu_s=cpu,thresholds=ts));print('RECENT BENIGN fitted',identity,flush=True)
    (B/'selections.json').write_text(json.dumps(choices,indent=2));stress=load_part(D,'evaluation','ton');out=[]
    for c in choices:
        model=joblib.load(R/c['artifact']);s=scores(model,representation(stress,c['method'],f));out.append(dict(**c,
            **metrics(stress['y'],s,c['thresholds']['shared'],stress['family'],windows=20),ranking=ranking_metrics(stress['y'],s)))
    (B/'stress_results.json').write_text(json.dumps(dict(results=out,scope=__doc__),indent=2));print('COMPLETE recent benign control',len(out),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
