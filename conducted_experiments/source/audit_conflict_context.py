import json
from pathlib import Path
import numpy as np,pandas as pd
p=Path('data/nf_ton_iot_v3/detection_foundation')
with np.load(p/'train.npz') as z:
 x,y,idx=z['x'],z['y'],z['row']
h=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
g=pd.DataFrame(dict(h=h,y=y)).groupby('h').y.agg(['min','max'])
mask=np.isin(h,g.index[g['min']!=g['max']])
selected=idx[mask];buf=[];offset=0
cols=['FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT','Attack']
for frame in pd.read_csv('data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=cols,chunksize=250000):
 stop=offset+len(frame);lo,hi=np.searchsorted(selected,[offset,stop])
 if hi>lo:buf.append(frame.iloc[selected[lo:hi]-offset])
 offset=stop
 if stop>selected[-1]:break
meta=pd.concat(buf,ignore_index=True)
meta['feature_hash']=h[mask];meta['label']=y[mask]
out={}
for col in cols:
 out[col]=int((meta.groupby('feature_hash')[col].nunique()>1).sum())
# Audit context dependence only. Addresses, ports and time remain excluded predictors.
groups=meta.groupby(['feature_hash','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT']).label.agg(['min','max','size'])
bad=groups['min']!=groups['max']
report=dict(scope='Training conflict context audit; metadata used only for provenance analysis',
            varying_context_vectors=out,full_context_conflicting_groups=int(bad.sum()),
            full_context_conflict_rows=int(groups.loc[bad,'size'].sum()))
Path('results/detection_foundation/conflict_context_audit.json').write_text(json.dumps(report,indent=2))
print(report)
