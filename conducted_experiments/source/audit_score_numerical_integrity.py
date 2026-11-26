"""Check whether probability rounding hides useful score ranking on development data."""
from pathlib import Path
import json,numpy as np,joblib
from threadpoolctl import threadpool_limits
from run_multienvironment_training import ranking_metrics
R=Path(__file__).resolve().parent;O=R/'results/multienvironment_2026-11-21'
def main():
 with np.load(R/'data/nf_ton_iot_v3/detection_foundation/evaluation.npz') as z:x=z['x'];y=z['y']
 cases=[]
 for name in ['rich_small','rich_large','rich_xgb','rich_logistic']:
  bundle=joblib.load(R/'results/detection_foundation'/(name+'.joblib'));m=bundle['model'];prob=m.predict_proba(x)[:,1]
  if hasattr(m,'booster_'):raw=m.booster_.predict(x,raw_score=True,num_threads=1)
  elif name=='rich_xgb':raw=m.predict(x,output_margin=True)
  else:raw=m.decision_function(x)
  cases.append(dict(name=name,probability_distinct=len(np.unique(prob)),raw_distinct=len(np.unique(raw)),probability_exact_zero=int((prob==0).sum()),probability_exact_one=int((prob==1).sum()),raw_min=float(raw.min()),raw_max=float(raw.max()),probability_ranking=ranking_metrics(y,prob),raw_ranking=ranking_metrics(y,raw)))
 (O/'score_numerical_integrity.json').write_text(json.dumps(dict(scope='Previously inspected NF development period only. Restores internal raw margins as a diagnostic; no protected data, no model or feature redesign.',results=cases),indent=2),encoding='utf-8');print(json.dumps(cases,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
