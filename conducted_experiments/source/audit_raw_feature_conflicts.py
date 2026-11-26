"""Audit cached float32 training conflicts against raw numeric exporter values."""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
D=ROOT/"data/nf_ton_iot_v3/detection_foundation"
manifest=json.loads((D/"manifest.json").read_text())
with np.load(D/"train.npz") as z:idx,x,y,family=z["row"],z["x"],z["y"],z["family"]
h=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
g=pd.DataFrame(dict(h=h,y=y)).groupby("h").y.agg(["min","max"])
bad=g.index[g["min"]!=g["max"]].to_numpy()
mask=np.isin(h,bad);selected=idx[mask]
buffers=[];offset=0
source=ROOT/"data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv"
for frame in pd.read_csv(source,usecols=manifest["features"],chunksize=250000,dtype=str):
 stop=offset+len(frame);lo,hi=np.searchsorted(selected,[offset,stop])
 if hi>lo:buffers.append(frame.iloc[selected[lo:hi]-offset].copy())
 offset=stop
raw=pd.concat(buffers,ignore_index=True)
numeric=raw.apply(pd.to_numeric,errors="raise")
assert np.array_equal(numeric.to_numpy(dtype=np.float32),x[mask])
raw_hash=pd.util.hash_pandas_object(numeric,index=False).to_numpy()
r=pd.DataFrame(dict(h=raw_hash,y=y[mask]))
groups=r.groupby("h").y.agg(["min","max","size"])
conf=groups["min"]!=groups["max"]
exact_text_hash=pd.util.hash_pandas_object(raw,index=False).to_numpy()
textgroups=pd.DataFrame(dict(h=exact_text_hash,y=y[mask])).groupby("h").y.agg(["min","max","size"])
textconf=textgroups["min"]!=textgroups["max"]
result=dict(cached_conflicting_vectors=len(bad),cached_conflict_rows=int(mask.sum()),
 raw_numeric_groups=len(groups),raw_numeric_conflicting_groups=int(conf.sum()),
 raw_numeric_conflicting_rows=int(groups.loc[conf,"size"].sum()),
 exact_text_conflicting_groups=int(textconf.sum()),exact_text_conflicting_rows=int(textgroups.loc[textconf,"size"].sum()),
 families={str(k):int((family[mask]==k).sum()) for k in np.unique(family[mask])},
 scope="Raw string and numeric audits of conflicting training rows only. Evaluation labels unused.")
Path("results/detection_foundation/raw_conflict_audit.json").write_text(json.dumps(result,indent=2))
np.savez_compressed(D/"raw_conflict_training.npz",x=numeric.to_numpy(dtype=np.float64),row=selected,y=y[mask],family=family[mask])
print(result)
