"""Matched causal-context ablation. All outcomes are development diagnostics."""
import json,io
from pathlib import Path
from time import perf_counter,process_time
import numpy as np,joblib,psutil
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from run_tolerance_threshold_audit import tolerance_threshold
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/nf_ton_iot_v3/detection_foundation'; OUT=ROOT/'results/causal_context_ablation'
def main():
 OUT.mkdir(exist_ok=True)
 data={}
 for name in ('train','calibration','evaluation'):
  with np.load(DATA/(name+'.npz')) as z:data[name]={k:z[k] for k in z.files}
 rows=[]
 for context,balanced in ((False,False),(True,False),(False,True),(True,True)):
  xs={name:np.ascontiguousarray(np.column_stack((d['x'],np.load(DATA/(name+'_causal_context.npy')))) if context else d['x']) for name,d in data.items()}
  for seed in (42,43,44):
   # Subsampling supplies real stochastic variation; identical settings in both arms.
   model=LGBMClassifier(n_estimators=100,num_leaves=15,min_child_samples=100,reg_lambda=10,
      subsample=.8,subsample_freq=1,colsample_bytree=.9,n_jobs=1,random_state=seed,verbosity=-1)
   weights=np.ones(len(data['train']['y']))
   if balanced:
    fam=data['train']['family'];attack=data['train']['y']==1;families=np.unique(fam[attack])
    for f in families:weights[fam==f]=attack.sum()/(len(families)*(fam==f).sum())
   start=process_time();model.fit(xs['train'],data['train']['y'],sample_weight=weights);cpu=process_time()-start
   cal=model.predict_proba(xs['calibration'])[:,1];scores=model.predict_proba(xs['evaluation'])[:,1]
   np.savez_compressed(OUT/f'scores_context{context}_balanced{balanced}_seed{seed}.npz',cal=cal,evaluation=scores)
   artifact=io.BytesIO();joblib.dump(model,artifact)
   native=model.booster_;single=xs['evaluation'][:1].copy();timings=[]
   for i in range(110):
    t=perf_counter();native.predict(single,num_threads=1);elapsed=(perf_counter()-t)*1000
    if i>=10:timings.append(elapsed)
   t=perf_counter();native.predict(xs['evaluation'],num_threads=1);batch_seconds=perf_counter()-t
   for cap in (.001,.01):
    for policy in ('pooled','worst_block'):
     threshold=tolerance_threshold(cal,cap,.05)['threshold'] if policy=='pooled' else max(tolerance_threshold(s,cap,.01)['threshold'] for s in np.array_split(cal,5))
     y=data['evaluation']['y'];pred=scores>threshold;fp=int(pred[y==0].sum());tp=int(pred[y==1].sum())
     windows=[]
     for ix in np.array_split(np.arange(len(y)),20):
      yy=y[ix];windows.append(dict(n=len(ix),benign=int((yy==0).sum()),attacks=int(yy.sum()),fp=int(pred[ix][yy==0].sum()),tp=int(pred[ix][yy==1].sum())))
     family={str(f):dict(n=int((data['evaluation']['family']==f).sum()),recall=float(pred[data['evaluation']['family']==f].mean())) for f in np.unique(data['evaluation']['family'][y==1])}
     r=dict(context=context,family_balanced=balanced,seed=seed,cap=cap,policy=policy,threshold=threshold,fp=fp,tp=tp,benign=int((y==0).sum()),attacks=int(y.sum()),fpr=float(pred[y==0].mean()),recall=float(pred[y==1].mean()),precision=tp/max(1,tp+fp),f1=2*tp/max(1,tp+fp+int(y.sum())),worst_window_fpr=max(w['fp']/w['benign'] for w in windows if w['benign']),windows=windows,family=family,bytes=len(artifact.getvalue()),fit_cpu_s=cpu,native_one_p95_ms=float(np.quantile(timings,.95)),native_batch_seconds=batch_seconds,observed_rss=psutil.Process().memory_info().rss)
     rows.append(r);print(context,balanced,seed,cap,policy,r['fpr'],r['recall'],flush=True)
   joblib.dump(model,OUT/f'model_context{context}_balanced{balanced}_seed{seed}.joblib')
   (OUT/'results.partial.json').write_text(json.dumps(rows,indent=2))
 (OUT/'results.json').write_text(json.dumps(dict(scope='Previously inspected NF-v3 development data. Matched features-only ablation, three stochastic seeds, fixed training/calibration/evaluation rows. No blind validation.',results=rows),indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
