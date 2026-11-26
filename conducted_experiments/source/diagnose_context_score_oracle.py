"""Post-hoc discrimination diagnostic; evaluation-label oracle is undeployable."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation'
def oracle(scores,y,cap):
 # Evaluate thresholds only after entire equal-score groups; no favorable tie breaks.
 order=np.argsort(-scores,kind='stable');s=scores[order];yy=y[order]
 ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
 tp=np.cumsum(yy)[ends];fp=(ends+1)-tp;valid=fp<=np.floor(cap*(y==0).sum())
 return dict(maximum_recall=float(tp[valid].max()/y.sum()) if valid.any() else 0,cap=cap)
def main():
 with np.load(D/'evaluation.npz') as z:y=z['y']
 rows=[]
 for folder in ('causal_context_ablation','causal_context_components'):
  for p in sorted((R/'results'/folder).glob('scores*.npz')):
   with np.load(p) as z:s=z['evaluation']
   rows.append(dict(file=str(p),oracle=[oracle(s,y,c) for c in (.001,.01)]))
 (R/'results/detection_foundation/context_score_oracle_diagnostic.json').write_text(json.dumps(dict(scope='Evaluation-label oracle only, no threshold selected for deployment, no feature redesign based on this diagnostic.',results=rows),indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
