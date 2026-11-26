"""Training-only benign domain fingerprint and repeated-vector diagnostic."""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from multienvironment_features import transform
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation';U=R/'data/nf_unsw_nb15_v3/development';O=R/'results/multienvironment_2026-11-21'
def load(p):
 with np.load(p) as z:return z['x'],z['y']
def main():
 f=json.loads((D/'manifest.json').read_text())['features'];rng=np.random.default_rng(20261121);xx=[]
 for p in [D/'train.npz',U/'train.npz']:
  x,y=load(p);ix=rng.choice(np.flatnonzero(y==0),20000,replace=False);xx.append(x[ix])
 raw=np.concatenate(xx);env=np.r_[np.zeros(20000),np.ones(20000)];records=[]
 for mode in ['raw46','physical14']:
  x=transform(raw,f,mode);groups=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy();train,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=42).split(x,env,groups))
  assert not np.intersect1d(groups[train],groups[test]).size
  m=LGBMClassifier(n_estimators=80,num_leaves=7,reg_lambda=10,n_jobs=1,verbosity=-1,random_state=42).fit(x[train],env[train]);p=m.predict_proba(x[test])[:,1]
  records.append(dict(representation=mode,benign_rows_per_environment=20000,unique_feature_vectors=len(np.unique(groups)),train_rows=len(train),test_rows=len(test),domain_auc=float(roc_auc_score(env[test],p)),missing_cells=int(np.isnan(x).sum())))
 result=dict(scope='Dataset-origin classification using BENIGN TRAINING rows only. Feature groups kept disjoint. This is a nuisance/domain-separability diagnostic, not IDS accuracy, chronological evaluation or an independent network result.',results=records)
 (O/'training_domain_fingerprint.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
