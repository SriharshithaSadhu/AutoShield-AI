"""Replay one frozen bank configuration through the reusable evaluator."""
from pathlib import Path
import json,numpy as np
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from candidate_evaluation import DevelopmentData,GeneralizationEvaluator
from run_representation_selection import load,predict,BANK
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation'
def main():
 tr=load('train',False);va=load('tune',False);t=np.load(D/'train_completion_ns.npy');vt=np.load(D/'tune_completion_ns.npy');benign=tr['y'][180000:]==0
 data=DevelopmentData(tr['x'][:180000],tr['y'][:180000],tr['family'][:180000],tr['x'][180000:][benign],tr['y'][180000:][benign],va['x'],va['y'],list(np.array_split(np.arange(len(va['y'])),3)),int(t[179999]),int(t[180000:][benign][0]),int(t[180000:][benign][-1]),int(vt[0]))
 params={k:v for k,v in BANK[0].items() if k!='context'}
 factory=lambda:LGBMClassifier(**params,n_jobs=1,verbosity=-1,random_state=42,subsample=.8,subsample_freq=1,colsample_bytree=.9)
 _,record=GeneralizationEvaluator(data,4,score_fn=predict).evaluate(factory)
 old=json.loads((R/'results/representation_selection/candidate_seed42_id0.json').read_text())
 for k in ('pooled_fpr','pooled_recall','worst_recall','minimum_family_recall','mean_family_recall','robust_recall_lower','proxy_threshold'):assert abs(record[k]-old[k])<1e-12,(k,record[k],old[k])
 assert record['family']==old['family'] and record['window_score']==old['window_score']
 for field,value in [('proxy_benign_y',np.ones(len(data.proxy_benign_y))),('fit_last_ns',data.proxy_first_ns)]:
  previous=getattr(data,field);setattr(data,field,value)
  try:data.validate()
  except ValueError:pass
  else:raise AssertionError('invalid information boundary accepted')
  setattr(data,field,previous)
 (R/'results/representation_selection/reusable_evaluator_parity.json').write_text(json.dumps(dict(status='pass',record=record,checks='Exact scorecard parity against frozen seed42 candidate0; mixed-label proxy and overlapping chronology rejected'),indent=2));print('PASS reusable evaluator exact scorecard parity and information-boundary rejection')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
