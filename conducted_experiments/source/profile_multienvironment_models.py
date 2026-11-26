"""Isolated process resource profile with actual representation transformation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(key,'1')
from pathlib import Path
from time import perf_counter,process_time
import argparse,json,sys,subprocess
import numpy as np,joblib,psutil
from threadpoolctl import threadpool_limits
from multienvironment_features import transform
R=Path(__file__).resolve().parent;O=R/'results/multienvironment_2026-11-21'
def worker(path):
 folder=path.parent;identity=path.name.removesuffix('_selected.joblib');mode='harmonized14' if identity.startswith('harmonized14') else 'physical14' if identity.startswith('physical14') else 'raw46';m=joblib.load(path);raw=np.load(folder/(identity+'_raw_input.npy'));cached=np.load(folder/(identity+'_input.npy'));features=json.loads((R/'data/nf_ton_iot_v3/detection_foundation/manifest.json').read_text())['features']
 assert np.allclose(transform(raw,features,mode),cached,equal_nan=True)
 times={}
 for n in (1,1000):
  elapsed=[];pre=[];inf=[];cpu=process_time()
  for i in range(100):
   t=perf_counter();xx=transform(raw[:n],features,mode);mid=perf_counter();m.booster_.predict(xx,num_threads=1);end=perf_counter()
   if i>=10:elapsed.append((end-t)*1000);pre.append((mid-t)*1000);inf.append((end-mid)*1000)
  times[str(n)]=dict(p50_ms=float(np.quantile(elapsed,.5)),p95_ms=float(np.quantile(elapsed,.95)),p99_ms=float(np.quantile(elapsed,.99)),preprocess_p95_ms=float(np.quantile(pre,.95)),inference_p95_ms=float(np.quantile(inf,.95)),cpu_s=process_time()-cpu)
 mem=psutil.Process().memory_info()._asdict();return dict(identity=identity,bytes=path.stat().st_size,batches=times,peak_resident_bytes=mem.get('peak_wset',mem['rss']),memory=mem,scope='Desktop isolated process, representation plus native inference. Cached 1000 real UNSW development inputs, no labels. Excludes CSV parsing, network capture, queues, alert delivery. Resident memory is process-wide, not model-only.')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--directory',default=str(O));a=ap.parse_args()
 if a.worker:
  with threadpool_limits(limits=1):print(json.dumps(worker(Path(a.worker))))
 else:
  directory=Path(a.directory);records=[]
  for p in sorted(directory.glob('*_selected.joblib')):
   out=subprocess.run([sys.executable,__file__,'--worker',str(p)],check=True,capture_output=True,text=True);r=json.loads(out.stdout);records.append(r);print(r['identity'],r['batches']['1']['p95_ms'],r['peak_resident_bytes'],flush=True)
   (directory/'resource_profiles.partial.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
  (directory/'resource_profiles.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
if __name__=='__main__':main()
