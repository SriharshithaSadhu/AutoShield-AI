"""Benign-only anomaly baseline on the fixed detection-foundation partitions."""
import io,json
from pathlib import Path
from time import process_time,perf_counter
import joblib,numpy as np
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/nf_ton_iot_v3/detection_foundation"
OUT=ROOT/"results/detection_foundation"

def main():
 with np.load(DATA/"train.npz") as z:train=z["x"][z["y"]==0]
 with np.load(DATA/"calibration.npz") as z:cal=z["x"]
 with np.load(DATA/"evaluation.npz") as z:x,y=z["x"],z["y"]
 records=[]
 for samples in (256,2048):
  model=IsolationForest(n_estimators=100,max_samples=samples,n_jobs=1,random_state=42)
  start=process_time();model.fit(train);cpu=process_time()-start
  cal_scores=-model.score_samples(cal)
  start=perf_counter();scores=-model.score_samples(x);scoring=perf_counter()-start
  stream=io.BytesIO();joblib.dump(model,stream)
  for cap in (.001,.01):
   t=tolerance_threshold(cal_scores,cap,.05)["threshold"];p=scores>t
   fp=int(p[y==0].sum());tp=int(p[y==1].sum())
   records.append(dict(max_samples=samples,cap=cap,fp=fp,tp=tp,
                       benign=int((y==0).sum()),attacks=int(y.sum()),fpr=float(p[y==0].mean()),
                       recall=float(p[y==1].mean()),fit_cpu_s=cpu,scoring_seconds=scoring,bytes=len(stream.getvalue())))
  joblib.dump(model,OUT/f"isolation_{samples}.joblib")
 (OUT/"benign_anomaly_baseline.json").write_text(json.dumps(records,indent=2))
 print(records)

if __name__=="__main__":
 with threadpool_limits(limits=1):main()
