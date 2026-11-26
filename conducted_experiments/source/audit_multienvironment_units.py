"""Training-only exporter/units audit; protected labels and examples are excluded."""
from pathlib import Path
import json,numpy as np,pandas as pd,hashlib
R=Path(__file__).resolve().parent;S=R/'data/nf_unsw_nb15_v3'
with np.load(S/'development/train.npz') as z:ids=z['row']
cols=['FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS','FLOW_DURATION_MILLISECONDS','SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS']
parts=[];offset=0
for f in pd.read_csv(S/'NF-UNSW-NB15-v3.csv',usecols=cols,chunksize=200000):
 lo,hi=np.searchsorted(np.sort(ids),[offset,offset+len(f)])
 if hi>lo:parts.append(f.iloc[np.sort(ids)[lo:hi]-offset])
 offset+=len(f)
f=pd.concat(parts,ignore_index=True);elapsed=f.FLOW_END_MILLISECONDS-f.FLOW_START_MILLISECONDS;duration=f.FLOW_DURATION_MILLISECONDS
r=dict(rows=len(f),dictionary_sha1=hashlib.sha1((S/'NetFlow_v3_Features.csv').read_bytes()).hexdigest(),duration_equal_end_minus_start=int((duration==elapsed).sum()),duration_equal_one_ms_floor=int((duration==np.maximum(1,elapsed)).sum()),duration_difference_quantiles=np.quantile(duration-elapsed,[0,.5,.99,1]).tolist(),nonfinite={c:dict(nan=int(f[c].isna().sum()),positive_inf=int(np.isposinf(f[c]).sum()),negative_inf=int(np.isneginf(f[c]).sum())) for c in cols if (~np.isfinite(f[c])).any()},negative_counts={c:int((f[c]<0).sum()) for c in cols if c not in ['FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS']},scope='Only 200000 fixed UNSW training rows; no labels read, no scale factors estimated from protected traffic.')
(S/'training_units_audit.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
