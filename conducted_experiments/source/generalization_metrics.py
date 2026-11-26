"""Reusable finite-window scorecards for resource-constrained model selection.

Intervals assume independent Bernoulli outcomes within each declared window.
Chronological drift/dependence can invalidate population interpretations. Bonferroni
covers a fixed candidate bank and fixed groups, never unspecified future streams.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
@dataclass(frozen=True)
class DeploymentLimits:
 fpr_cap:float=.001
 minimum_recall:float=.5
 model_bytes:int=1572864
 p95_ms:float=5.
 memory_bytes:int=536870912

def binomial_bounds(successes,n,alpha):
 if not (0<alpha<1) or not (0<=successes<=n):raise ValueError('invalid binomial inputs')
 if n==0:return (0.,1.)
 low=0. if successes==0 else float(beta.ppf(alpha,successes,n-successes+1))
 high=1. if successes==n else float(beta.ppf(1-alpha,successes+1,n-successes))
 return low,high

def score_windows(y,pred,windows,bank_size=1,alpha=.05):
 y=np.asarray(y);pred=np.asarray(pred,dtype=bool)
 if y.shape!=pred.shape or not np.isin(y,[0,1]).all():raise ValueError('invalid labels/predictions')
 # Two one-sided quantities for each group and each fixed candidate.
 local_alpha=alpha/(2*bank_size*len(windows));rows=[]
 for ix in windows:
  ix=np.asarray(ix);yy=y[ix];pp=pred[ix]
  benign=int((yy==0).sum());attacks=int((yy==1).sum())
  fp=int(pp[yy==0].sum());tp=int(pp[yy==1].sum())
  rows.append(dict(benign=benign,attacks=attacks,fp=fp,tp=tp,
    fpr=fp/benign if benign else None,recall=tp/attacks if attacks else None,
    fpr_upper=binomial_bounds(fp,benign,local_alpha)[1],
    recall_lower=binomial_bounds(tp,attacks,local_alpha)[0]))
 return dict(windows=rows,worst_fpr=max((r['fpr'] for r in rows if r['fpr'] is not None),default=None),
  worst_fpr_upper=max(r['fpr_upper'] for r in rows),
  worst_recall_lower=min((r['recall_lower'] for r in rows if r['attacks']),default=0),
  local_alpha=local_alpha,missing_benign_windows=sum(r['benign']==0 for r in rows),
  missing_attack_windows=sum(r['attacks']==0 for r in rows))

def joint_feasibility(score,resources,limits=DeploymentLimits()):
 return dict(empirical_fpr=score['worst_fpr'] is not None and score['worst_fpr']<=limits.fpr_cap,
  bounded_fpr=score['worst_fpr_upper']<=limits.fpr_cap,
  recall=score['worst_recall_lower']>=limits.minimum_recall,
  model_size=resources['bytes']<=limits.model_bytes,
  latency=resources['p95_ms']<=limits.p95_ms,
  memory=None if resources.get('peak_memory_bytes') is None else resources['peak_memory_bytes']<=limits.memory_bytes)
