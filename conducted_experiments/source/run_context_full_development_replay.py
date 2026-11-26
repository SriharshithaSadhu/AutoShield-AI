"""Full existing ToN development-tail replay; never an independent validation.

Two predeclared representations, three seeds, frozen shared-window decisions.
Real batches of at most1000 flows; metrics include all rows. No adaptation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
from pathlib import Path
from time import perf_counter,process_time
from collections import defaultdict
import json,datetime,hashlib
import numpy as np,pandas as pd,joblib,psutil
from threadpoolctl import threadpool_limits
from run_network_context_study import R,D,O
from network_context_features import transform
B=O/'full_development_replay'

def empty():return dict(benign=0,attacks=0,fp=0,tp=0)

def update(counts,y,pred):
    counts['benign']+=int((y==0).sum());counts['attacks']+=int((y==1).sum())
    counts['fp']+=int(pred[y==0].sum());counts['tp']+=int(pred[y==1].sum())

def finish(c):
    a,b=c['attacks'],c['benign'];tp,fp=c['tp'],c['fp']
    return dict(**c,recall=tp/a if a else None,fpr=fp/b if b else None,
                precision=tp/max(1,tp+fp),f1=2*tp/max(1,tp+fp+a))

def main():
    B.mkdir(exist_ok=True);manifest=json.loads((D/'manifest.json').read_text());features=manifest['features'];start,stop=manifest['boundaries'][-2:]
    choices=[]
    for variant in ['physical14','context29']:
        for seed in [42,43,44]:
            c=json.loads((O/f'{variant}_seed{seed}_selection.json').read_text());s=c['selected']['shared_window']
            choices.append(dict(identity=c['identity'],variant=variant,seed=seed,
                artifact=s['artifact'],artifact_sha256=hashlib.sha256((R/s['artifact']).read_bytes()).hexdigest(),threshold=s['thresholds']['shared']))
    protocol=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,source_rows=[start,stop],
        batch_size=1000,flow_window=10000,time_window_ms=60000,minimum_benign_for_supported_window=100,
        policy='Previously frozen shared_window; representations chosen as original baseline versus complete context, not stress-period winners.',
        deadline_scope='Offline batch compute including metrics. No network delivery or live arrival-time guarantee.',choices=choices)
    (B/'protocol.json').write_text(json.dumps(protocol,indent=2))
    models={c['identity']:joblib.load(R/c['artifact']) for c in choices}
    total={c['identity']:empty() for c in choices}
    windows={c['identity']:defaultdict(empty) for c in choices}
    time_windows={c['identity']:defaultdict(empty) for c in choices}
    families={c['identity']:defaultdict(lambda:dict(n=0,tp=0)) for c in choices}
    times=[];cpu_before=process_time();wall_before=perf_counter();offset=0;processed=0;previous=None;first_ms=None
    cols=features+['L4_SRC_PORT','L4_DST_PORT','FLOW_END_MILLISECONDS','Label','Attack']
    source=D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv'
    for frame in pd.read_csv(source,usecols=cols,chunksize=100000):
        end=offset+len(frame)
        if end<=start:offset=end;continue
        frame=frame.iloc[max(0,start-offset):max(0,min(len(frame),stop-offset))]
        frame_start=max(start,offset)
        for a in range(0,len(frame),1000):
            d=frame.iloc[a:a+1000];n=len(d);row_ids=np.arange(frame_start+a,frame_start+a+n)
            t=d.FLOW_END_MILLISECONDS.to_numpy(dtype=np.int64)
            assert (np.diff(t)>=0).all() and (previous is None or previous<=t[0]);previous=int(t[-1])
            if first_ms is None:first_ms=int(t[0])
            tick=perf_counter();x=d[features].to_numpy(dtype=np.float32);ports=d[['L4_SRC_PORT','L4_DST_PORT']].to_numpy(dtype=np.uint16)
            x[~np.isfinite(x)]=np.nan;y=d.Label.to_numpy(dtype=np.uint8);family=d.Attack.to_numpy(dtype=str)
            assert np.isin(y,[0,1]).all()
            fw=(row_ids-start)//10000;tw=(t-first_ms)//60000
            representations={v:transform(x,ports,features,v) for v in ['physical14','context29']}
            for c in choices:
                identity=c['identity'];score=models[identity].booster_.predict(representations[c['variant']],num_threads=1);pred=score>c['threshold']
                update(total[identity],y,pred)
                for w in np.unique(fw):
                    ix=fw==w;update(windows[identity][int(w)],y[ix],pred[ix])
                for w in np.unique(tw):
                    ix=tw==w;update(time_windows[identity][int(w)],y[ix],pred[ix])
                for f in np.unique(family[y==1]):
                    ix=(family==f)&(y==1);families[identity][str(f)]['n']+=int(ix.sum());families[identity][str(f)]['tp']+=int(pred[ix].sum())
            times.append((perf_counter()-tick)*1000);processed+=n
        offset=end
        if processed and processed//1000000!=(processed-len(frame))//1000000:
            print('full context replay',processed,'rows',round(perf_counter()-wall_before,1),'s',flush=True)
        if end>=stop:break
    assert processed==stop-start
    resources=dict(cpu_s=process_time()-cpu_before,wall_s=perf_counter()-wall_before,batches=len(times),
        combined_six_model_batch_compute_p50_ms=float(np.quantile(times,.5)),
        combined_six_model_batch_compute_p95_ms=float(np.quantile(times,.95)),
        combined_six_model_batch_compute_p99_ms=float(np.quantile(times,.99)),
        peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),
        scope='CSV parse contributes to wall/CPU and memory, but not batch compute timing. Batch timing includes preprocessing, six model scores and all evaluation bookkeeping. Offline desktop only.')
    results=[]
    for c in choices:
        identity=c['identity'];out=dict(**c,**finish(total[identity]));out['family']={f:dict(**z,recall=z['tp']/z['n']) for f,z in families[identity].items()}
        for name,table in [('flow',windows[identity]),('time',time_windows[identity])]:
            records=[dict(window=w,**finish(z)) for w,z in sorted(table.items())]
            assert all(sum(r[k] for r in records)==total[identity][k] for k in ['benign','attacks','fp','tp'])
            out[name+'_windows']=records
            out['worst_'+name+'_window_fpr']=max(r['fpr'] for r in records if r['benign'])
            supported=[r for r in records if r['benign']>=100]
            out['worst_supported_'+name+'_window_fpr']=max(r['fpr'] for r in supported)
        assert sum(z['n'] for z in out['family'].values())==out['attacks']
        results.append(out)
    (B/'results.json').write_text(json.dumps(dict(protocol=protocol,resources=resources,results=results),indent=2))
    print('COMPLETE full existing development replay',processed,'rows',resources,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
