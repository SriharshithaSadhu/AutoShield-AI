import numpy as np
from generalization_metrics import binomial_bounds,score_windows
assert binomial_bounds(0,0,.05)==(0.,1.)
assert abs(binomial_bounds(0,100,.05)[1]-(1-.05**.01))<1e-12
assert abs(binomial_bounds(100,100,.05)[0]-.05**.01)<1e-12
y=np.array([0,0,1,1,0,1]);p=np.array([0,1,1,0,0,1]);s=score_windows(y,p,[np.arange(4),np.arange(4,6)],4)
assert s['worst_fpr']==.5 and s['windows'][0]['tp']==1
assert s['worst_fpr_upper']>.5 and s['worst_recall_lower']<.5
print('PASS exact binomial endpoints and worst-window aggregation')
