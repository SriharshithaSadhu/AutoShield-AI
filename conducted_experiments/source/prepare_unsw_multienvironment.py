"""Audit and prepare chronological UNSW development data; protected suffix stays unscored."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
R=Path(__file__).resolve().parent;S=R/'data/nf_unsw_nb15_v3';O=S/'development';O.mkdir(exist_ok=True)
features=json.loads((R/'data/nf_ton_iot_v3/detection_foundation/manifest.json').read_text())['features']
raw=S/'NF-UNSW-NB15-v3.csv';header=pd.read_csv(raw,nrows=0).columns.tolist();assert set(features)<=set(header)
assert (S/'NetFlow_v3_Features.csv').read_bytes()==(R/'data/nf_ton_iot_v3/NetFlow_v3_Features.csv').read_bytes()
times=pd.read_csv(raw,usecols=['FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS']);t=times.FLOW_END_MILLISECONDS.to_numpy(dtype=np.int64)
assert np.isfinite(t).all() and (t>=times.FLOW_START_MILLISECONDS.to_numpy()).all()
order=np.argsort(t,kind='stable');sorted_t=t[order];n=len(t);fractions=[0,.5,.55,.65,.8,1]
bounds=[0]+[int(np.searchsorted(sorted_t,sorted_t[int(n*f)],side='left')) for f in fractions[1:-1]]+[n]
names=['train','calibration','selection','development_holdout','protected'];rng=np.random.default_rng(20261121);indices={}
for name,a,b in zip(names,bounds[:-1],bounds[1:]):
 ids=order[a:b]
 if name=='protected':
  np.save(O/'protected_row_ids.npy',ids);continue
 cap=200000 if name=='train' else 30000 if name=='calibration' else 80000
 # Sampling is independent of labels; calibration benign filtering occurs later.
 selected=np.sort(rng.choice(ids,min(cap,len(ids)),replace=False));indices[name]=selected
buffers={k:[] for k in indices};offset=0
for frame in pd.read_csv(raw,usecols=features+['Label','Attack'],chunksize=200000):
 for name,ids in indices.items():
  lo,hi=np.searchsorted(ids,[offset,offset+len(frame)])
  if hi>lo:
   local=ids[lo:hi]-offset;f=frame.iloc[local].copy();f['_row']=ids[lo:hi];buffers[name].append(f)
 offset+=len(frame)
report=dict(rows=n,features=features,feature_dictionary_identical=True,completion_inversions_in_source=int((np.diff(t)<0).sum()),sort='stable completion timestamp',bounds=bounds,fractions=fractions,protected_rows=bounds[-1]-bounds[-2],protected_labels_summarized=False,partitions={},source_sha1='2cba4879ead17c51e3268a01c06af41186cd90e4')
for name in indices:
 f=pd.concat(buffers[name],ignore_index=True);ids=f['_row'].to_numpy();ii=np.argsort(t[ids],kind='stable');f=f.iloc[ii];ids=ids[ii];x=f[features].to_numpy(dtype=np.float32);y=f.Label.to_numpy(dtype=np.uint8);family=f.Attack.to_numpy(dtype=str)
 assert np.isin(y,[0,1]).all(),np.unique(y)
 nonfinite={features[i]:int((~np.isfinite(x[:,i])).sum()) for i in range(len(features)) if (~np.isfinite(x[:,i])).any()}
 # Retain every selected row. Native missing-value handling; explicit infinity conversion.
 x[~np.isfinite(x)]=np.nan
 np.savez_compressed(O/(name+'.npz'),x=x,y=y,family=family,row=ids,completion_ms=t[ids])
 report['partitions'][name]=dict(nonfinite_before_native_missing=nonfinite,rows=len(y),attacks=int(y.sum()),families={str(k):int(v) for k,v in zip(*np.unique(family,return_counts=True))},first_ms=int(t[ids][0]),last_ms=int(t[ids][-1]),row_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
(S/'development_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
