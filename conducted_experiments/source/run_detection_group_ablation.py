"""Train once per consistent feature group, using only training labels."""
import json
from pathlib import Path
from time import process_time
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
from run_tolerance_threshold_audit import tolerance_threshold
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/nf_ton_iot_v3/detection_foundation'
OUT=ROOT/'results/detection_foundation'
with np.load(DATA/'train.npz') as z:x,y=z['x'],z['y']
h=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
frame=pd.DataFrame(dict(hash=h,y=y,row=np.arange(len(y))))
groups=frame.groupby('hash').agg(min_label=('y','min'),max_label=('y','max'),first=('row','first'),rows=('y','size'))
valid=groups.min_label==groups.max_label
idx=groups.loc[valid,'first'].to_numpy()
result=dict(train_rows=len(y),unique_vectors=len(groups),conflicting_vectors=int((~valid).sum()),
            conflict_rows=int(groups.loc[~valid,'rows'].sum()),consistent_group_training_rows=len(idx),models=[])
with np.load(DATA/'calibration.npz') as z:cal=z['x']
with np.load(DATA/'evaluation.npz') as z:ev,ey=z['x'],z['y']
with threadpool_limits(limits=1):
 for balance in (None,'balanced'):
  model=LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,random_state=42,verbosity=-1,class_weight=balance)
  start=process_time();model.fit(x[idx],y[idx]);cpu=process_time()-start
  cs=model.predict_proba(cal)[:,1];s=model.predict_proba(ev)[:,1]
  for cap in (.001,.01):
   t=tolerance_threshold(cs,cap,.05)['threshold'];pred=s>t
   fpr,recall,_=roc_curve(ey,s,drop_intermediate=False)
   result['models'].append(dict(class_weight=balance,cap=cap,fit_cpu_s=cpu,
      fp=int(pred[ey==0].sum()),tp=int(pred[ey==1].sum()),
      fpr=float(pred[ey==0].mean()),recall=float(pred[ey==1].mean()),
      diagnostic_oracle_recall=float(recall[np.flatnonzero(fpr<=cap)[-1]])))
  joblib.dump(model,OUT/f'dedup_lgb_{balance}.joblib')
(OUT/'group_balanced_training.json').write_text(json.dumps(result,indent=2))
print(result)
