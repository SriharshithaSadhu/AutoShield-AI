"""Retrospective score separability diagnostic, never used for selection."""
import json
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/detection_foundation"
DATA=ROOT/"data/nf_ton_iot_v3/detection_foundation"

def main():
 with np.load(DATA/"evaluation.npz") as z:x,y,family=z["x"],z["y"],z["family"]
 manifest=json.loads((DATA/"manifest.json").read_text())
 records=[]
 for path in sorted(OUT.glob("*.joblib")):
  bundle=joblib.load(path)
  if not isinstance(bundle,dict):
   continue
  ids=[manifest["features"].index(f) for f in bundle["features"]]
  scores=bundle["model"].predict_proba(x[:,ids])[:,1]
  fp,recall,threshold=roc_curve(y,scores,drop_intermediate=False)
  for cap in (.001,.01):
   i=np.flatnonzero(fp<=cap)[-1]
   records.append(dict(model=path.stem,cap=cap,oracle_recall=float(recall[i]),
                       oracle_fpr=float(fp[i]),threshold=float(threshold[i])))
 report=dict(scope="Evaluation-label oracle diagnostic; cannot select or deploy these thresholds.",
             results=records)
 (OUT/"score_separability_oracle.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(records,indent=2))

if __name__=="__main__":
 with threadpool_limits(limits=1):main()
