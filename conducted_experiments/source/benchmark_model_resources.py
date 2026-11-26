"""Isolated desktop process benchmark for serialized LightGBM candidates."""
import argparse,json,subprocess,sys
from pathlib import Path
from time import perf_counter,process_time
import joblib,numpy as np,psutil
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parent

def worker(path):
 p=psutil.Process();before=perf_counter();m=joblib.load(path);load_seconds=perf_counter()-before
 with np.load(R/'data/nf_ton_iot_v3/detection_foundation/train.npz') as z:x=np.ascontiguousarray(z['x'][:1000])
 if m.n_features_in_==50:x=np.column_stack((x,np.load(R/'data/nf_ton_iot_v3/detection_foundation/train_causal_context.npy')[:1000]))
 times=[]
 for i in range(120):
  t=perf_counter();m.predict_proba(x[:1]);dt=(perf_counter()-t)*1000
  if i>=20:times.append(dt)
 cpu=process_time();wall=perf_counter()
 for _ in range(20):m.predict_proba(x)
 elapsed=perf_counter()-wall;cpu=process_time()-cpu
 mem=p.memory_info()._asdict()
 return dict(path=str(path),bytes=Path(path).stat().st_size,load_seconds=load_seconds,p95_ms=float(np.quantile(times,.95)),batch_1000_mean_ms=elapsed/20*1000,scoring_cpu_seconds=cpu,memory=mem,peak_memory_bytes=mem.get('peak_wset'),scope='Isolated desktop process; peak OS working set includes imports, NPZ training-array decompression, model and input. Conservative process-level memory, not incremental model memory. No Pi measurement.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory');ap.add_argument('--worker');args=ap.parse_args()
 if args.worker:
  with threadpool_limits(limits=1):print(json.dumps(worker(args.worker)))
 else:
  directory=Path(args.directory);rows=[]
  for path in sorted(directory.glob('*.joblib')):
   run=subprocess.run([sys.executable,__file__,'--worker',str(path)],capture_output=True,text=True,check=True)
   row=json.loads(run.stdout);rows.append(row);print(path.name,row['p95_ms'],row['peak_memory_bytes'],flush=True)
  (directory/'isolated_desktop_resources.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
