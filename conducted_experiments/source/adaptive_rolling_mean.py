"""Original implementation of Algorithm 1 / Eqs 3.3,3.5 in Han et al., ICML24.
https://arxiv.org/html/2402.08672v2 . Mean estimator, not their tournament.
Assumptions of independent within-period samples are not established for flows.
"""
import numpy as np

def adaptive_rolling_mean(groups,delta=.05,width=1.):
 if not groups or not (0<delta<1) or width<=0:raise ValueError('invalid inputs')
 groups=[np.asarray(g,dtype=float) for g in groups]
 if any(len(g)==0 for g in groups):raise ValueError('empty period: cannot apply published setting')
 t=len(groups);local_delta=delta/(3*t);log=np.log(2/local_delta);means=[];psi=[];counts=[]
 n=0;total=0.;squares=0.
 for g in reversed(groups):
  n+=len(g);total+=g.sum();squares+=np.square(g).sum();mean=total/n
  variance=max(0,(squares-total*total/n)/(n-1)) if n>1 else 0
  uncertainty=width if n==1 else np.sqrt(variance*2*log/n)+8*width*log/(3*(n-1))
  counts.append(n);means.append(mean);psi.append(uncertainty)
 bias=[max(0,max(abs(means[k]-means[i])-psi[k]-psi[i] for i in range(k+1))) for k in range(t)]
 chosen=int(np.argmin(np.asarray(bias)+psi))
 return dict(estimate=float(means[chosen]),lookback=chosen+1,counts=counts,means=means,uncertainty=psi,bias_proxy=bias,delta_prime=local_delta)
