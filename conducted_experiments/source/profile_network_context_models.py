"""Separate-process desktop feature+inference cost on matched varied flow traces."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,json,sys,subprocess
from pathlib import Path
from time import perf_counter,process_time
import numpy as np,joblib,psutil
from threadpoolctl import threadpool_limits
from network_context_features import transform
R=Path(__file__).resolve().parent;O=R/'results/network_context_2026-11-21'

def worker(task):
    with np.load(O/'resource_trace.npz') as z:raw=z['x'];ports=z['ports'];features=z['features'].tolist()
    model=joblib.load(R/task['artifact']);timings=[];feature_times=[];model_times=[];before=process_time()
    for i in range(3300):
        j=i%len(raw);a=perf_counter();x=transform(raw[j:j+1],ports[j:j+1],features,task['variant']);b=perf_counter()
        model.booster_.predict(x,num_threads=1);c=perf_counter()
        if i>=300:timings.append((c-a)*1000);feature_times.append((b-a)*1000);model_times.append((c-b)*1000)
    cpu=process_time()-before;batch=[]
    for i in range(75):
        j=(i%3)*1000;a=perf_counter();x=transform(raw[j:j+1000],ports[j:j+1000],features,task['variant']);model.booster_.predict(x,num_threads=1)
        if i>=15:batch.append((perf_counter()-a)*1000)
    memory=psutil.Process().memory_info()._asdict()
    return dict(**task,bytes=(R/task['artifact']).stat().st_size,samples=3000,warmup=300,
        one_row_p50_ms=float(np.quantile(timings,.5)),one_row_p95_ms=float(np.quantile(timings,.95)),one_row_p99_ms=float(np.quantile(timings,.99)),
        feature_p95_ms=float(np.quantile(feature_times,.95)),native_inference_p95_ms=float(np.quantile(model_times,.95)),
        one_row_cpu_s=cpu,batch1000_p95_ms=float(np.quantile(batch,.95)),
        peak_resident_bytes=memory.get('peak_wset',memory['rss']),memory=memory,
        scope='Desktop completed-flow preprocessing plus inference, varied label-independent traces, one thread. Excludes exporter, capture, queues and alert transport. No Pi measurements.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');args=ap.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):print(json.dumps(worker(json.loads(args.worker))))
        return
    from run_network_context_study import holdouts,D
    h=holdouts();xx=[];pp=[]
    for d in h.values():
        ids=np.linspace(0,len(d['x'])-1,1000,dtype=int);xx.append(d['x'][ids]);pp.append(d['ports'][ids])
    features=json.loads((D/'manifest.json').read_text())['features']
    np.savez_compressed(O/'resource_trace.npz',x=np.concatenate(xx),ports=np.concatenate(pp),features=np.array(features))
    choices=json.loads((O/'selections.json').read_text());out=[]
    for c in choices:
        s=c['selected']['shared_window'];task=dict(identity=c['identity'],variant=c['variant'],seed=c['seed'],artifact=s['artifact'],policy='shared_window')
        result=subprocess.run([sys.executable,__file__,'--worker',json.dumps(task)],check=True,capture_output=True,text=True)
        r=json.loads(result.stdout);out.append(r)
        print('profile',c['identity'],'p95',round(r['one_row_p95_ms'],4),'ms',flush=True)
        (O/'resource_profiles.partial.json').write_text(json.dumps(out,indent=2))
    (O/'resource_profiles.json').write_text(json.dumps(out,indent=2));print('COMPLETE desktop context profiles',len(out),flush=True)
if __name__=='__main__':main()
