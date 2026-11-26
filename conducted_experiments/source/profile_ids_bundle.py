"""Run on the deployment device; never interprets desktop timing as Pi timing."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from five_track_common import probabilities


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',required=True)
    p.add_argument('--fixture',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--repetitions',type=int,default=200)
    args=p.parse_args()
    if args.repetitions<20: p.error('use at least 20 timing repetitions')
    process=psutil.Process();rss_before=process.memory_info().rss
    bundle=joblib.load(args.bundle)
    with np.load(args.fixture) as z:
        x=z['x'];expected=z['expected']
    assert x.shape[1]==len(bundle['features'])
    got=probabilities(bundle,x)
    error=float(np.max(np.abs(got-expected)))
    assert np.allclose(got,expected,atol=1e-8,rtol=1e-6),error
    measured=[];observed_rss=[process.memory_info().rss]
    for batch in (1,min(32,len(x))):
        for _ in range(10):probabilities(bundle,x[:batch])
        timings=[]
        for i in range(args.repetitions):
            start=perf_counter();probabilities(bundle,x[:batch]);timings.append((perf_counter()-start)*1000)
            observed_rss.append(process.memory_info().rss)
        measured.append(dict(batch=batch,p50_batch_ms=float(np.median(timings)),p95_batch_ms=float(np.quantile(timings,.95)),
                             median_rows_per_second=batch/(np.median(timings)/1000)))
    result=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
                versions={n:importlib.metadata.version(n) for n in ('numpy','scikit-learn','lightgbm','xgboost','joblib','psutil')},
                bundle_sha256=hashlib.sha256(Path(args.bundle).read_bytes()).hexdigest(),artifact_bytes=Path(args.bundle).stat().st_size,
                probability_parity_max_error=error,measurements=measured,rss_before_load=rss_before,
                maximum_observed_rss_bytes=max(observed_rss),energy_joules=None,
                limitations=['RSS sampled after inference, not guaranteed peak allocation','No energy instrumentation',
                             'Fixed repeated fixture; cache-hot measurements, not a live workload','Model-feature input; flow extraction is excluded',
                             'Record device power mode, cooling, OS and competing workloads separately'])
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
