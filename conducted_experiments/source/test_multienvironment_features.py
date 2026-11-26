import numpy as np
from multienvironment_features import transform,PHYSICAL_NAMES
features=['PROTOCOL','FLOW_DURATION_MILLISECONDS','IN_PKTS','OUT_PKTS','IN_BYTES','OUT_BYTES','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS','SHORTEST_FLOW_PKT','LONGEST_FLOW_PKT','TCP_FLAGS']
x=np.array([[6,10,2,1,100,50,1,0,40,60,18],[17,0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)
a=transform(x,features,'physical14');assert a.shape==(2,14) and np.isfinite(a).all();assert np.isclose(a[0,6],1/3) and a[0,12]==1 and a[0,13]==1 and a[1,6]==0
xx=x.copy();xx[0,4]=np.nan;b=transform(xx,features,'physical14');assert np.isnan(b[0,3]) and np.isnan(b[0,4]);assert np.isfinite(b[1]).all()
print('PASS physical feature ratios, zero denominators, TCP flags, explicit missing propagation')

h=transform(x,features,'harmonized14');assert np.isclose(h[1,1],np.log(2));assert np.allclose(h[:,np.arange(14)!=1],a[:,np.arange(14)!=1]);assert np.allclose(h[0],a[0]);print('PASS isolated one-ms harmonization; all other features preserved')
