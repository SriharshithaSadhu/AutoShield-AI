"""Dataset-independent candidate evaluation with explicit information boundaries."""
from dataclasses import dataclass
from time import process_time
import numpy as np
from generalization_metrics import score_windows,binomial_bounds
from run_tolerance_threshold_audit import tolerance_threshold
@dataclass
class DevelopmentData:
 train_x:np.ndarray
 train_y:np.ndarray
 train_family:np.ndarray
 proxy_benign_x:np.ndarray
 proxy_benign_y:np.ndarray
 validation_x:np.ndarray
 validation_y:np.ndarray
 validation_windows:list
 fit_last_ns:int
 proxy_first_ns:int
 proxy_last_ns:int
 validation_first_ns:int
 def validate(self):
  if not self.fit_last_ns<self.proxy_first_ns<=self.proxy_last_ns<self.validation_first_ns:raise ValueError('chronological boundaries overlap')
  if len(self.proxy_benign_y)==0 or np.any(self.proxy_benign_y!=0):raise ValueError('proxy calibration must contain benign labels only')
  if len(self.train_x)!=len(self.train_y) or len(self.train_family)!=len(self.train_y):raise ValueError('training lengths differ')
  if len(self.proxy_benign_x)!=len(self.proxy_benign_y) or len(self.validation_x)!=len(self.validation_y):raise ValueError('calibration/validation lengths differ')
  if not self.validation_windows or len(np.unique(self.validation_y))!=2:raise ValueError('validation windows and both classes required')
  ix=np.concatenate(self.validation_windows)
  if len(ix)!=len(self.validation_y) or not np.array_equal(np.sort(ix),np.arange(len(self.validation_y))):raise ValueError('validation windows must partition validation rows exactly')
  if len(np.unique(self.train_y))!=2:raise ValueError('both training classes required')

class GeneralizationEvaluator:
 def __init__(self,data,bank_size,cap=.001,delta=.05,score_fn=None):
  data.validate()
  if not isinstance(bank_size,int) or bank_size<1:raise ValueError('positive fixed bank size required')
  self.data=data;self.bank_size=bank_size;self.cap=cap;self.delta=delta
  self.score_fn=score_fn or (lambda model,x:model.predict_proba(np.ascontiguousarray(x))[:,1])
 def evaluate(self,estimator_factory):
  d=self.data;start=process_time();model=estimator_factory().fit(np.ascontiguousarray(d.train_x),d.train_y)
  def cutoff(m):
   result=tolerance_threshold(self.score_fn(m,d.proxy_benign_x),self.cap,self.delta)
   if result['threshold'] is None:raise ValueError('insufficient proxy calibration support')
   return result['threshold']
  threshold=cutoff(model);pred=self.score_fn(model,d.validation_x)>threshold
  score=score_windows(d.validation_y,pred,d.validation_windows,self.bank_size,alpha=self.delta/2)
  families=[];labels=np.unique(d.train_family[d.train_y==1])
  for family in labels:
   keep=d.train_family!=family;held=(d.train_family==family)&(d.train_y==1)
   if len(np.unique(d.train_y[keep]))<2:raise ValueError('family holdout leaves no positive training class')
   transfer=estimator_factory().fit(np.ascontiguousarray(d.train_x[keep]),d.train_y[keep]);t=cutoff(transfer)
   tp=int((self.score_fn(transfer,d.train_x[held])>t).sum());n=int(held.sum())
   families.append(dict(family=str(family),n=n,tp=tp,recall=tp/n,recall_lower=binomial_bounds(tp,n,self.delta/(2*self.bank_size*len(labels)))[0]))
  counts=score['windows'];record=dict(proxy_threshold=threshold,proxy_benign_rows=len(d.proxy_benign_y),window_score=score,family=families,
   pooled_fpr=sum(v['fp'] for v in counts)/sum(v['benign'] for v in counts),pooled_recall=sum(v['tp'] for v in counts)/sum(v['attacks'] for v in counts),
   worst_recall=min(v['recall'] for v in counts if v['recall'] is not None),minimum_family_recall=min(v['recall'] for v in families),
   mean_family_recall=float(np.mean([v['recall'] for v in families])),robust_recall_lower=min([score['worst_recall_lower']]+[v['recall_lower'] for v in families]),fit_family_cpu_s=process_time()-start,
   scope='Development-only chronological validation plus internal nonchronological family-transfer diagnostics. Resources require separate measurement. No final calibration/evaluation accepted by this API.')
  return model,record
