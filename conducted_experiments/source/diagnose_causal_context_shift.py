"""Post-evaluation shift diagnosis, never used to redesign/select features."""
from pathlib import Path
import numpy as np,json,joblib
from scipy.stats import ks_2samp
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation';out={};reference=None
for name in ('train','tune','calibration','evaluation'):
 c=np.load(D/(name+'_causal_context.npy'))
 with np.load(D/(name+'.npz')) as z:y=z['y']
 if name=='train':reference=c[y==0]
 out[name]={}
 for label in (0,1):
  cc=c[y==label]
  if not len(cc):continue
  out[name][str(label)]=dict(n=len(cc),quantiles=np.quantile(cc,[0,.5,.9,.99,1],axis=0).tolist(),ks_vs_training_benign=[float(ks_2samp(reference[:,i],cc[:,i]).statistic) for i in range(4)])
p=R/'results/causal_context_ablation/model_contextTrue_balancedFalse_seed42.joblib'
if p.exists():
 m=joblib.load(p);gain=m.booster_.feature_importance(importance_type='gain');out['context_feature_gain_fraction']=(gain[-4:]/gain.sum()).tolist()
(R/'results/detection_foundation/causal_context_shift_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
