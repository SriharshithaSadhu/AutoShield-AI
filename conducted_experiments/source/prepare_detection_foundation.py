"""Prepare fixed development partitions for resource-constrained detection comparisons."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
folder=ROOT/"data/nf_ton_iot_v3/detection_foundation"
folder.mkdir(exist_ok=True)
with np.load(ROOT/"data/nf_ton_iot_v3/core6_development/stream.npz") as z:
 offsets,labels=z["chunk_offsets"],z["labels"]
n=len(offsets)-1
boundaries=[0,int(offsets[n//2]),int(offsets[7*n//12]),int(offsets[2*n//3]),len(labels)]
rng=np.random.default_rng(20261118)
indices={}
for name,left,right in zip(("train","tune","calibration","evaluation"),boundaries,boundaries[1:]):
 candidates=np.arange(left,right)
 if name=="train":
  selected=np.concatenate([rng.choice(candidates[labels[left:right]==c],100000,replace=False) for c in (0,1)])
 elif name=="calibration":
  selected=rng.choice(candidates[labels[left:right]==0],200000,replace=False)
 else:
  selected=rng.choice(candidates,min(200000,len(candidates)),replace=False)
 indices[name]=np.sort(selected)
exclude={"FLOW_START_MILLISECONDS","FLOW_END_MILLISECONDS","IPV4_SRC_ADDR","IPV4_DST_ADDR","L4_SRC_PORT","L4_DST_PORT","Label","Attack","DNS_QUERY_ID"}
source=ROOT/"data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv"
features=[c for c in pd.read_csv(source,nrows=0) if c not in exclude]
buffers={k:[] for k in indices};families={k:[] for k in indices}
offset=0
for frame in pd.read_csv(source,usecols=features+["Label","Attack"],chunksize=200000):
 stop=offset+len(frame)
 assert np.array_equal(frame.Label.to_numpy(),labels[offset:stop])
 for name,idx in indices.items():
  lo,hi=np.searchsorted(idx,[offset,stop])
  if hi>lo:
   rows=idx[lo:hi]-offset
   buffers[name].append(frame.iloc[rows][features].to_numpy(dtype=np.float32))
   families[name].append(frame.iloc[rows].Attack.to_numpy(dtype=str))
 offset=stop
assert offset==len(labels)
for name,idx in indices.items():
 np.savez_compressed(folder/(name+".npz"),x=np.concatenate(buffers[name]),y=labels[idx],
                     family=np.concatenate(families[name]),row=idx)
manifest=dict(boundaries=boundaries,features=features,seed=20261118,
 partitions={k:dict(rows=len(v),indices_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in indices.items()},
 scope="Chronological sampled development protocol on previously explored NF-v3; evaluation sample unused for configuration selection.")
(folder/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(manifest)
