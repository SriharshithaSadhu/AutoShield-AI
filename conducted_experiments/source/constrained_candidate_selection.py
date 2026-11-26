"""Strict candidate admission for the fixed generalization bank.

Inputs contain development validation scorecards and label-independent resource
profiles. Final calibration/evaluation scores are not accepted by this API.
No feasible candidate returns None; diagnostic best-effort fits are not approvals.
"""
from dataclasses import dataclass
@dataclass(frozen=True)
class Limits:
 fpr:float=.001
 recall:float=.5
 model_bytes:int=1572864
 memory_bytes:int=536870912
 p95_ms:float=5.

def select(records,profiles,rule,limits=Limits()):
 admitted=[];audit=[]
 for c in records:
  p=profiles.get(c['id']);ws=c['window_score']
  risk=c['pooled_fpr'] if rule=='pooled' else ws['worst_fpr_upper'] if rule=='generalization_bound' else ws['worst_fpr']
  value=c['pooled_recall'] if rule=='pooled' else c['worst_recall'] if rule=='worst_window' else c['minimum_family_recall'] if rule=='family_empirical' else c['robust_recall_lower']
  tests=dict(fpr=risk<=limits.fpr,recall=value>=limits.recall,model_size=c['bytes']<=limits.model_bytes,
   latency=p is not None and p.get('native_p95_ms',float('inf'))<=limits.p95_ms,
   memory=p is not None and p.get('peak_memory_bytes') is not None and p['peak_memory_bytes']<=limits.memory_bytes)
  audit.append(dict(id=c['id'],checks=tests,value=value,risk=risk))
  if all(tests.values()):admitted.append((value,c['mean_family_recall'],-c['bytes'],-c['id'],c['id']))
 return dict(selected=max(admitted)[-1] if admitted else None,admitted=len(admitted),audit=audit,
  scope='Feasibility under observed development scorecards and desktop component resources; no future-distribution guarantee.')
