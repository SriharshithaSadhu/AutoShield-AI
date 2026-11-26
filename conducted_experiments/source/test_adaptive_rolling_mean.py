import numpy as np
from adaptive_rolling_mean import adaptive_rolling_mean
r=adaptive_rolling_mean([np.zeros(100),np.zeros(200),np.zeros(300)])
assert r['lookback']==3 and r['estimate']==0 and r['counts']==[300,500,600]
r=adaptive_rolling_mean([np.ones(10000),np.ones(10000),np.zeros(10000)])
assert r['lookback']==1 and r['estimate']==0
r=adaptive_rolling_mean([np.array([.2])]);assert r['estimate']==.2 and r['uncertainty']==[1.]
print('PASS stationary pooling, shifted-history exclusion, singleton case')
