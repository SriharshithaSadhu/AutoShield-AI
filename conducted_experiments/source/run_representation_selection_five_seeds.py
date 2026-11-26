"""Shared four-candidate representation bank with causal temporal validation.

The family-transfer diagnostic is internal group holdout, not a chronological
forecast. All its labels precede candidate selection and final calibration.
"""
from pathlib import Path
from time import perf_counter,process_time
import json,io
import numpy as np,joblib
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from generalization_metrics import score_windows,binomial_bounds
from run_tolerance_threshold_audit import tolerance_threshold
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation';O=R/'results/representation_selection'
BANK=[dict(context=c,n_estimators=n,num_leaves=15,min_child_samples=child,reg_lambda=reg) for c in (False,True) for n,child,reg in ((60,20,0),(150,200,10))]
def fit(x,y,cfg,seed):
 cfg={k:v for k,v in cfg.items() if k!='context'}
 return LGBMClassifier(**cfg,n_jobs=1,verbosity=-1,random_state=seed,subsample=.8,subsample_freq=1,colsample_bytree=.9).fit(np.ascontiguousarray(x),y)
def predict(m,x):return m.booster_.predict(np.ascontiguousarray(x),num_threads=1)
def load(name,context):
 with np.load(D/(name+'.npz')) as z:d={k:z[k] for k in z.files}
 if context:d['x']=np.column_stack((d['x'],np.load(D/(name+'_causal_context.npy'))))
 return d

def main():
 O.mkdir(exist_ok=True);protocol=dict(bank=BANK,seeds=[42,43,44,45,46],fit_sample_end=180000,proxy_calibration_sample_start=180000,
  validation='Three chronological tune windows; four internal leave-family-out transfer diagnostics.',fpr_cap=.001,recall_target=.5,size_cap=1572864,latency_cap_ms=5,
  scope='Development only: previously inspected NF-v3 capture. Final calibration/evaluation are loaded after each seed selection. No optimizer comparison.')
 (O/'protocol.json').write_text(json.dumps(protocol,indent=2));runs=[]
 for seed in protocol['seeds']:
  candidates=[]
  for ci,cfg in enumerate(BANK):
   cache=O/f'candidate_seed{seed}_id{ci}.json'
   if cache.exists():candidates.append(json.loads(cache.read_text()));continue
   tr=load('train',cfg['context']);va=load('tune',cfg['context']);ix=np.arange(180000);x=tr['x'];y=tr['y'];f=tr['family']
   cp=x[180000:][y[180000:]==0];start=process_time();m=fit(x[ix],y[ix],cfg,seed)
   cutoff=tolerance_threshold(predict(m,cp),.001,.05)['threshold'];assert cutoff is not None
   vp=predict(m,va['x'])>cutoff;windows=np.array_split(np.arange(len(vp)),3)
   score=score_windows(va['y'],vp,windows,bank_size=len(BANK),alpha=.025)
   counts=score['windows'];pooled_fp=sum(v['fp'] for v in counts);pooled_b=sum(v['benign'] for v in counts)
   pooled_tp=sum(v['tp'] for v in counts);pooled_a=sum(v['attacks'] for v in counts)
   families=[]
   for family in np.unique(f[ix][y[ix]==1]):
    fi=ix[f[ix]!=family];vi=ix[f[ix]==family]
    if len(np.unique(y[fi]))<2:raise ValueError('missing training class')
    held=fit(x[fi],y[fi],cfg,seed);threshold=tolerance_threshold(predict(held,cp),.001,.05)['threshold']
    tp=int((predict(held,x[vi])>threshold).sum());n=len(vi)
    families.append(dict(family=str(family),n=n,tp=tp,recall=tp/n,recall_lower=binomial_bounds(tp,n,.025/(len(BANK)*4))[0]))
   b=io.BytesIO();joblib.dump(m,b);times=[]
   for i in range(70):
    t=perf_counter();predict(m,x[:1]);dt=(perf_counter()-t)*1000
    if i>=20:times.append(dt)
   record=dict(seed=seed,id=ci,config=cfg,window_score=score,pooled_fpr=pooled_fp/pooled_b,pooled_recall=pooled_tp/pooled_a,
    worst_recall=min(v['recall'] for v in counts if v['recall'] is not None),family= families,
    minimum_family_recall=min(v['recall'] for v in families),mean_family_recall=float(np.mean([v['recall'] for v in families])),
    robust_recall_lower=min([score['worst_recall_lower']]+[v['recall_lower'] for v in families]),
    bytes=len(b.getvalue()),p95_ms=float(np.quantile(times,.95)),fit_family_cpu_s=process_time()-start,proxy_threshold=cutoff)
   cache.write_text(json.dumps(record,indent=2));joblib.dump(m,O/f'candidate_seed{seed}_id{ci}.joblib');candidates.append(record)
   print('candidate',seed,ci,record['pooled_recall'],score['worst_fpr'],record['minimum_family_recall'],flush=True)
  decisions={}
  resources=[c for c in candidates if c['bytes']<=1572864 and c['p95_ms']<=5]
  for rule in ('pooled','worst_window','family_empirical','generalization_bound'):
   def metrics(c):
    risk=c['pooled_fpr'] if rule=='pooled' else c['window_score']['worst_fpr_upper'] if rule=='generalization_bound' else c['window_score']['worst_fpr']
    value=c['pooled_recall'] if rule=='pooled' else c['worst_recall'] if rule=='worst_window' else c['minimum_family_recall'] if rule=='family_empirical' else c['robust_recall_lower']
    return risk,value
   def key(c):
    risk,value=metrics(c)
    return (risk<=.001,-max(0,risk-.001),value,c['mean_family_recall'] if rule.startswith(('family','generalization')) else 0,-c['id'])
   chosen=max(resources or candidates,key=key);risk,value=metrics(chosen)
   decisions[rule]=dict(id=chosen['id'],selection_fpr_feasible=bool(resources) and risk<=.001,selection_recall_feasible=value>=.5,
      memory_certified=False,selection_value=value,selection_risk=risk)
  (O/f'decisions_seed{seed}.json').write_text(json.dumps(decisions,indent=2))
  # Decisions are durable before final calibration/evaluation is opened.
  outcomes={}
  for ci in sorted({v['id'] for v in decisions.values()}):
   cfg=BANK[ci];m=joblib.load(O/f'candidate_seed{seed}_id{ci}.joblib');cal=load('calibration',cfg['context']);ev=load('evaluation',cfg['context'])
   threshold=tolerance_threshold(predict(m,cal['x']),.001,.05)['threshold'];ep=predict(m,ev['x'])>threshold;y=ev['y']
   ws=score_windows(y,ep,np.array_split(np.arange(len(y)),20),bank_size=len(BANK))
   tp=int(ep[y==1].sum());fp=int(ep[y==0].sum());a=int(y.sum());b=int((y==0).sum())
   family={str(f):dict(n=int((ev['family']==f).sum()),recall=float(ep[ev['family']==f].mean())) for f in np.unique(ev['family'][y==1])}
   outcomes[ci]=dict(tp=tp,fp=fp,attacks=a,benign=b,recall=tp/a,fpr=fp/b,precision=tp/max(1,tp+fp),f1=2*tp/max(1,tp+fp+a),window_score=ws,family=family,threshold=threshold)
  runs.append(dict(seed=seed,decisions=decisions,outcomes=outcomes));(O/'results.partial.json').write_text(json.dumps(runs,indent=2))
 (O/'results.json').write_text(json.dumps(dict(protocol=protocol,runs=runs),indent=2));print('complete',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
