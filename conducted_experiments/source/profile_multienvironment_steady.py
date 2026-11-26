"""Steady-state resource profile on diverse real development flows.
Preserves earlier measurements; no training jobs should run concurrently.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
from time import perf_counter,process_time
import argparse,json,sys,subprocess,hashlib
import numpy as np,joblib,psutil
from threadpoolctl import threadpool_limits
from multienvironment_features import transform
from multienvironment_protocol_tools import raw_score
R=Path(__file__).resolve().parent;ROOT=R/'results/multienvironment_2026-11-21'
def prepare_inputs():
 from multienvironment_protocol_tools import development_holdouts
 d=development_holdouts();raw=np.concatenate([d[e]['x'][:1000] for e in ['ton_future','unsw_future','ton_stress']]);path=ROOT/'resource_trace_raw.npy';np.save(path,raw);return path

def worker(path):
 model=joblib.load(path);raw=np.load(ROOT/'resource_trace_raw.npy');mode='harmonized14' if path.name.startswith('harmonized14') else 'physical14' if path.name.startswith('physical14') else 'raw46';features=json.loads((R/'data/nf_ton_iot_v3/detection_foundation/manifest.json').read_text())['features'];cpu_load=psutil.cpu_percent(interval=.1);timings=[];pre=[];inference=[];start_cpu=process_time()
 for i in range(3300):
  idx=i%len(raw);a=perf_counter();x=transform(raw[idx:idx+1],features,mode);b=perf_counter();raw_score(model,x);c=perf_counter()
  if i>=300:timings.append((c-a)*1000);pre.append((b-a)*1000);inference.append((c-b)*1000)
 cpu=process_time()-start_cpu;batch=[]
 for i in range(75):
  idx=(i%3)*1000;a=perf_counter();raw_score(model,transform(raw[idx:idx+1000],features,mode));d=(perf_counter()-a)*1000
  if i>=15:batch.append(d)
 mem=psutil.Process().memory_info()._asdict();return dict(identity=path.name.removesuffix('_selected.joblib'),artifact=str(path),bytes=path.stat().st_size,representation=mode,host_cpu_percent_before=cpu_load,warmup=300,samples=3000,real_input_rows=len(raw),one_row_p50_ms=float(np.quantile(timings,.5)),one_row_p95_ms=float(np.quantile(timings,.95)),one_row_p99_ms=float(np.quantile(timings,.99)),preprocess_p95_ms=float(np.quantile(pre,.95)),native_inference_p95_ms=float(np.quantile(inference,.95)),one_row_component_cpu_s=cpu,batch1000_p95_ms=float(np.quantile(batch,.95)),peak_resident_bytes=mem.get('peak_wset',mem['rss']),memory=mem,scope='Separate desktop process, 3000 varied real inputs drawn equally from three development periods, transformation plus native score. Includes warmup before timing. Excludes capture/export, CSV parse, queueing and alert transport. No Pi measurement or guarantee under arbitrary host contention.')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--directory');ap.add_argument('--pattern',default='*_selected.joblib');args=ap.parse_args()
 if args.worker:
  with threadpool_limits(limits=1):print(json.dumps(worker(Path(args.worker))))
 else:
  directory=Path(args.directory)
  if not (ROOT/'resource_trace_raw.npy').exists():prepare_inputs()
  results=[]
  for path in sorted(directory.glob(args.pattern)):
   c=subprocess.run([sys.executable,__file__,'--worker',str(path)],capture_output=True,text=True,check=True);r=json.loads(c.stdout);results.append(r);print(r['identity'],r['one_row_p95_ms'],flush=True)
  (directory/'steady_resource_profiles.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if __name__=='__main__':main()
