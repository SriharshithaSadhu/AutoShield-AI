"""Shared candidate bank for pooled, temporal and family-aware selection."""
import io,json
from pathlib import Path
from time import process_time,perf_counter
import joblib,numpy as np,psutil
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/nf_ton_iot_v3/detection_foundation"
OUT=ROOT/"results/generalization_evaluator_v3"
BANK=[dict(n_estimators=60,num_leaves=15,min_child_samples=20,reg_lambda=0),
      dict(n_estimators=150,num_leaves=31,min_child_samples=20,reg_lambda=0),
      dict(n_estimators=150,num_leaves=15,min_child_samples=200,reg_lambda=10),
      dict(n_estimators=100,num_leaves=7,min_child_samples=500,reg_lambda=30)]

def fit(x,y,config,seed):
 return LGBMClassifier(**config,n_jobs=1,random_state=seed,verbosity=-1,subsample=.8,subsample_freq=1,colsample_bytree=.9).fit(np.ascontiguousarray(x),y)

def measure(model,x,y):
 p=model.predict_proba(np.ascontiguousarray(x))[:,1]
 return p

def main():
 OUT.mkdir(exist_ok=True)
 with np.load(DATA/"train.npz") as z:x,y,f=z["x"],z["y"],z["family"]
 # Windows refer only to the chronological training-development prefix.
 n=len(y);fit_end=9*n//10;cal_end=n
 fit_idx=np.arange(fit_end)
 proxy_cal=x[fit_end:cal_end][y[fit_end:cal_end]==0]
 with np.load(DATA/"tune.npz") as z:tx,ty=z["x"],z["y"]
 windows=np.array_split(np.arange(len(ty)),3)
 rows=[]
 for seed in (42,43,44):
  candidates=[]
  for ci,config in enumerate(BANK):
   start=process_time()
   model=fit(x[fit_idx],y[fit_idx],config,seed)
   artifact=io.BytesIO();joblib.dump(model,artifact)
   timings=[]
   single=np.ascontiguousarray(x[:1])
   for _ in range(120):
    before=perf_counter();model.booster_.predict(single,num_threads=1);timings.append((perf_counter()-before)*1000)
   resources=dict(bytes=len(artifact.getvalue()),batch_one_p95_ms=float(np.quantile(timings[20:],.95)),
                  observed_process_rss=psutil.Process().memory_info().rss)
   t=tolerance_threshold(measure(model,proxy_cal,None),.001,.05)["threshold"]
   if t is None:raise ValueError("insufficient proxy benign")
   counts=[];family_recalls=[]
   for ix in windows:
    pred=measure(model,tx[ix],None)>t;yy=ty[ix]
    counts.append(dict(fp=int(pred[yy==0].sum()),benign=int((yy==0).sum()),
                       tp=int(pred[yy==1].sum()),attacks=int(yy.sum())))
   pooled=sum(r["tp"] for r in counts)/max(1,sum(r["attacks"] for r in counts))
   worst_recall=min(r["tp"]/r["attacks"] for r in counts if r["attacks"])
   worst_fpr=max(r["fp"]/r["benign"] for r in counts if r["benign"])
   # Family-transfer diagnostic: leave one entire family out within known development prefix.
   # These folds are not chronological forecasts; tune windows above are chronological.
   for family in np.unique(f[y==1]):
    training=fit_idx[(f[fit_idx]!=family)]
    if len(np.unique(y[training]))<2:continue
    validation=fit_idx.copy()
    validation=validation[(f[validation]==family)&(y[validation]==1)]
    if not len(validation):continue
    held=fit(x[training],y[training],config,seed)
    cutoff=tolerance_threshold(measure(held,proxy_cal,None),.001,.05)["threshold"]
    family_recalls.append(dict(family=str(family),n=len(validation),
                                recall=float((measure(held,x[validation],None)>cutoff).mean())))
   candidates.append(dict(config_id=ci,resources=resources,pooled_recall=pooled,worst_recall=worst_recall,
                           worst_fpr=worst_fpr,held_family=family_recalls,
                           held_family_min=min([r["recall"] for r in family_recalls],default=0),
                           development_counts=counts,fit_cpu_s=process_time()-start))
   print(seed,ci,pooled,worst_fpr,flush=True)
  chosen={}
  resource_feasible=[r for r in candidates if r["resources"]["bytes"]<=1.5*1024**2
                     and r["resources"]["batch_one_p95_ms"]<=5]
  (OUT/f"candidates_seed{seed}.json").write_text(json.dumps(candidates,indent=2))
  for rule,key in (("pooled","pooled_recall"),("worst_window","worst_recall"),("family_aware","held_family_min")):
   # Same feasibility-first ordering for every rule.
   resource_feasible=[r for r in candidates if r["resources"]["bytes"]<=1.5*1024**2
                      and r["resources"]["batch_one_p95_ms"]<=5]
   pool=resource_feasible if resource_feasible else candidates
   def ranking(r):
    counts=r["development_counts"]
    pooled_fpr=sum(c["fp"] for c in counts)/max(1,sum(c["benign"] for c in counts))
    risk=pooled_fpr if rule=="pooled" else r["worst_fpr"]
    return (risk<=.001,-max(0,risk-.001),r[key],
            np.mean([v["recall"] for v in r["held_family"]]) if rule=="family_aware" else 0,
            r["worst_recall"] if rule=="family_aware" else 0,-r["config_id"])
   chosen[rule]=max(pool,key=ranking)["config_id"]
  (OUT/f"decisions_seed{seed}.json").write_text(json.dumps(dict(candidates=candidates,selected=chosen,
      diagnostic_only=not bool(resource_feasible)),indent=2))
  with np.load(DATA/"calibration.npz") as z:final_cal=z["x"]
  with np.load(DATA/"evaluation.npz") as z:ex,ey=z["x"],z["y"]
  evaluated={}
  for ci in sorted(set(chosen.values())):
   model=fit(x,y,BANK[ci],seed)
   cutoff=tolerance_threshold(measure(model,final_cal,None),.001,.05)["threshold"]
   pred=measure(model,ex,None)>cutoff
   artifact=OUT/f"model_seed{seed}_config{ci}.joblib";joblib.dump(model,artifact)
   evaluated[ci]=dict(fp=int(pred[ey==0].sum()),tp=int(pred[ey==1].sum()),
    fpr=float(pred[ey==0].mean()),recall=float(pred[ey==1].mean()),bytes=artifact.stat().st_size)
  rows.append(dict(seed=seed,diagnostic_only=not bool(resource_feasible),selected=chosen,
                   outcomes={k:evaluated[v] for k,v in chosen.items()}))
 (OUT/"results.json").write_text(json.dumps(dict(scope="Development comparison on previously inspected NF-v3. Four shared configurations per seed, identical internal folds, final calibration excluded from selection. Family transfer folds are not chronological forecasts.",bank=BANK,version="v3: chronological tune windows; training-prefix leave-family-out transfer diagnostic; stochastic seeds; fixed bank",runs=rows),indent=2))
 print(rows)

if __name__=="__main__":
 with threadpool_limits(limits=1):main()
