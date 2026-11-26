"""Isolated desktop, real contiguous prefix, single-flow deployment service cost.

No accuracy estimation or Pi claim. Excludes capture/export/network transport;
includes feature state, native inference, threshold and an in-memory alert sink.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
from pathlib import Path
from time import perf_counter,process_time
import argparse,json,subprocess,sys,platform
import numpy as np,pandas as pd,joblib,psutil
from threadpoolctl import threadpool_limits
from behavioral_observation import BehavioralObservation
from behavioral_data import META_COLUMNS,metadata,event_columns
from behavioral_models import scores
from multienvironment_features import transform
from run_behavioral_full_replay import RAW_FIELDS
R=Path(__file__).resolve().parent
O=R/'results/behavioral_observation_2026-11-22'

def worker(method):
    selected=next(c for c in json.loads((O/'selections.json').read_text()) if c['method']==method and c['seed']==42)['selected']['shared_pooled']
    path=R/selected['artifact'];model=joblib.load(path);engine=BehavioralObservation()
    columns=list(dict.fromkeys(META_COLUMNS+RAW_FIELDS))
    # Real records only; no labels loaded. Equal contiguous trace for all methods.
    frame=pd.read_csv(R/'data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=columns,nrows=6000)
    m=metadata(frame,0);raw=frame[RAW_FIELDS].to_numpy(dtype=np.float32);events=list(zip(*event_columns(m)))
    proc=psutil.Process();times=[];feature_times=[];native_times=[];positive=0;peak=proc.memory_info().rss
    start=perf_counter();cpu=process_time()
    for i,event in enumerate(events):
        t=perf_counter();f=t
        b=engine.observe(*event) if method!='flow14' else None
        if method=='flow14':x=transform(raw[i:i+1],RAW_FIELDS,'physical14')
        elif method=='hybrid38':x=np.column_stack([transform(raw[i:i+1],RAW_FIELDS,'physical14'),np.asarray(b,dtype=np.float32).reshape(1,-1)])
        elif method=='behavior26_global_ablation':x=np.asarray([(*b,*engine.global_features())],dtype=np.float32)
        else:x=np.asarray(b,dtype=np.float32).reshape(1,-1)
        feature_ms=(perf_counter()-f)*1000;n=perf_counter();s=scores(model,x);native_ms=(perf_counter()-n)*1000
        # Actual delivery target here is a constant-memory in-process consumer.
        if bool(s[0]>selected['thresholds']['shared']):positive+=1
        elapsed=(perf_counter()-t)*1000
        if i>=1000:times.append(elapsed);feature_times.append(feature_ms);native_times.append(native_ms)
        if i%500==0:peak=max(peak,proc.memory_info().rss)
    peak=max(peak,proc.memory_info().rss)
    def stats(x):return dict(mean_ms=float(np.mean(x)),p50_ms=float(np.quantile(x,.5)),p95_ms=float(np.quantile(x,.95)),p99_ms=float(np.quantile(x,.99)),max_ms=float(np.max(x)))
    return dict(method=method,model_bytes=path.stat().st_size,warmup_records=1000,measured_records=5000,
        service=stats(times),feature=stats(feature_times),native=stats(native_times),peak_sampled_rss_bytes=peak,
        process_peak_working_set_bytes=proc.memory_info().peak_wset,process_cpu_s=process_time()-cpu,wall_s=perf_counter()-start,
        state=engine.diagnostics() if method!='flow14' else None,
        resource_pass=bool(path.stat().st_size<=1572864 and np.quantile(times,.95)<=5 and proc.memory_info().peak_wset<=536870912))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');args=ap.parse_args()
    if args.worker:
        with threadpool_limits(limits=1):print(json.dumps(worker(args.worker)))
        return
    records=[]
    for method in ['flow14','behavior24','quantile_tail_anomaly','hybrid38','behavior26_global_ablation']:
        done=subprocess.run([sys.executable,__file__,'--worker',method],cwd=R,capture_output=True,text=True,check=True)
        records.append(json.loads(done.stdout));print('BENCHMARK',method,records[-1]['service'],flush=True)
    (O/'deployment_benchmark.json').write_text(json.dumps(dict(records=records,platform=platform.platform(),cpu=platform.processor(),
        scope=__doc__,trace='First6000 genuine ordered ToN records;1000 warmup,5000 measured; early prefix workload, not worst-case capacity.',
        caveats=['Desktop only; no Raspberry Pi measurements.','Service cost excludes CSV loading, capture, exporter receipt, network transport and queue wait.',
                 'Independent process per method; arrays for6000 records retained. Peak working set includes imports and trace loading.',
                 'Combined service includes state update even at tied timestamps; current event has no access to same-time peers.']),indent=2))
if __name__=='__main__':main()
