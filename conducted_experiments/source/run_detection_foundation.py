"""Finite-bank chronological detection study with untouched final calibration."""
import io,json
from pathlib import Path
from time import perf_counter,process_time
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/nf_ton_iot_v3/detection_foundation"
OUT=ROOT/"results/detection_foundation"
CORE=["PROTOCOL","FLOW_DURATION_MILLISECONDS","IN_PKTS","OUT_PKTS","IN_BYTES","OUT_BYTES"]

def main():
 OUT.mkdir(exist_ok=True)
 manifest=json.loads((DATA/"manifest.json").read_text())
 data={}
 for name in ("train","tune","calibration","evaluation"):
  with np.load(DATA/(name+".npz")) as z:data[name]={k:z[k] for k in z.files}
 # Track feature-vector overlap without conflating float32 projection with raw identity.
 hashes={name:pd.util.hash_pandas_object(pd.DataFrame(rows["x"]),index=False).to_numpy()
         for name,rows in data.items()}
 overlap={name:int(np.isin(hashes[name],hashes["train"]).sum()) for name in data if name!="train"}
 candidates=[
 ("core_small",CORE,LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,random_state=42,verbosity=-1)),
 ("rich_small",manifest["features"],LGBMClassifier(n_estimators=60,num_leaves=15,n_jobs=1,random_state=42,verbosity=-1)),
 ("rich_large",manifest["features"],LGBMClassifier(n_estimators=200,num_leaves=31,n_jobs=1,random_state=42,verbosity=-1)),
 ("rich_regularized",manifest["features"],LGBMClassifier(n_estimators=200,num_leaves=15,min_child_samples=200,reg_lambda=10,n_jobs=1,random_state=42,verbosity=-1)),
 ("rich_xgb",manifest["features"],XGBClassifier(n_estimators=100,max_depth=5,n_jobs=1,random_state=42,tree_method="hist")),
 ("rich_extra",manifest["features"],ExtraTreesClassifier(n_estimators=60,max_depth=16,min_samples_leaf=5,n_jobs=1,random_state=42)),
 ("rich_logistic",manifest["features"],make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=300,random_state=42)))]
 tune=[];models={}
 for name,features,model in candidates:
  ids=[manifest["features"].index(f) for f in features]
  train=data["train"];start=process_time()
  model.fit(np.nan_to_num(train["x"][:,ids],nan=0,posinf=0,neginf=0),train["y"])
  fit_cpu=process_time()-start
  scores=model.predict_proba(np.nan_to_num(data["tune"]["x"][:,ids],nan=0,posinf=0,neginf=0))[:,1]
  threshold=tolerance_threshold(scores[data["tune"]["y"]==0],.001,.05)["threshold"]
  y=data["tune"]["y"];pred=scores>threshold
  stream=io.BytesIO();joblib.dump(model,stream);size=len(stream.getvalue())
  measure=np.nan_to_num(data["tune"]["x"][:1,ids],nan=0,posinf=0,neginf=0)
  timings=[]
  for _ in range(60):
   start=perf_counter();model.predict_proba(measure);timings.append((perf_counter()-start)*1000)
  record=dict(name=name,features=features,bytes=size,fit_cpu_s=fit_cpu,
              batch_one_p95_ms=float(np.quantile(timings,.95)),
              tuning_recall=float(pred[y==1].mean()),tuning_fpr=float(pred[y==0].mean()),
              tuning_threshold=threshold)
  tune.append(record);models[name]=(model,ids)
  joblib.dump(dict(model=model,features=features),OUT/(name+".joblib"))
  print("Fit",name,record["tuning_recall"],size,flush=True)
 # Write selections before final calibration or evaluation scoring.
 eligible=[r for r in tune if r["bytes"]<=1.5*1024*1024 and r["batch_one_p95_ms"]<=5]
 selection=max(eligible,key=lambda r:r["tuning_recall"])["name"] if eligible else None
 (OUT/"selection.json").write_text(json.dumps(dict(primary_cap=.001,minimum_recall=.5,selected=selection,candidates=tune),indent=2))
 records=[]
 for record in tune:
  name=record["name"];model,ids=models[name]
  cal=data["calibration"]
  cal_scores=model.predict_proba(np.nan_to_num(cal["x"][:,ids],nan=0,posinf=0,neginf=0))[:,1]
  # Robust comparator calibrates on five consecutive benign blocks; union correction.
  for cap in (.001,.01):
   pooled=tolerance_threshold(cal_scores,cap,.05)["threshold"]
   block_thresholds=[tolerance_threshold(s,cap,.05/5)["threshold"] for s in np.array_split(cal_scores,5)]
   for policy,threshold in (("pooled",pooled),("worst_block",max(block_thresholds))):
    ev=data["evaluation"];start=perf_counter()
    scores=model.predict_proba(np.nan_to_num(ev["x"][:,ids],nan=0,posinf=0,neginf=0))[:,1]
    elapsed=perf_counter()-start;pred=scores>threshold;y=ev["y"]
    fp=int(pred[y==0].sum());tp=int(pred[y==1].sum());benign=int((y==0).sum());attack=int(y.sum())
    family={f:dict(n=int(((ev["family"]==f)&(y==1)).sum()),
                   tp=int((pred&(ev["family"]==f)&(y==1)).sum())) for f in np.unique(ev["family"][y==1])}
    blocks=[]
    for ix in np.array_split(np.arange(len(y)),20):
     b=int((y[ix]==0).sum())
     if b:blocks.append(float(pred[ix][y[ix]==0].mean()))
    # Same scored sample, restricted to vectors absent from training.
    clean=~np.isin(hashes["evaluation"],hashes["train"])
    records.append(dict(name=name,cap=cap,calibration_policy=policy,threshold=threshold,
        fp=fp,tp=tp,benign=benign,attacks=attack,fpr=fp/benign,recall=tp/attack,
        precision=tp/max(1,tp+fp),f1=2*tp/max(1,tp+fp+attack),worst_sample_block_fpr=max(blocks),
        family=family,evaluation_scoring_seconds=elapsed,
        evaluation_without_train_vectors=dict(rows=int(clean.sum()),
         benign=int((clean&(y==0)).sum()),attacks=int((clean&(y==1)).sum()),
         fp=int((clean&pred&(y==0)).sum()),tp=int((clean&pred&(y==1)).sum()))))
 result=dict(scope="Finite bank development comparison, no optimizer superiority or independent validation.",
  manifest=manifest,train_float32_vector_overlap=overlap,candidates=tune,selected=selection,operating_points=records)
 (OUT/"results.json").write_text(json.dumps(result,indent=2))
 print("Completed",len(records),"operating points",flush=True)

if __name__=="__main__":
 with threadpool_limits(limits=1):main()
