"""Prepare auxiliary CSE-v3 source using flows completed before 27 Feb 2018 UTC.
March 1 and all later flows are excluded by time before label statistics/sampling.
This cache is an auxiliary historical source for ToN2019; no CSE test is scored.
"""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,datetime,zipfile,io
R=Path(__file__).resolve().parent;S=R/'data/nf_cicids2018_v3';O=S/'permitted_source';FEATURES=json.loads((R/'data/nf_ton_iot_v3/detection_foundation/manifest.json').read_text())['features']
def main():
 O.mkdir(exist_ok=True);archive=S/'NF-CICIDS2018-v3_UQ.zip';z=zipfile.ZipFile(archive)
 names=z.namelist();member=next(n for n in names if n.endswith('/data/NF-CICIDS2018-v3.csv'));dictionary=next(n for n in names if n.endswith('/data/NetFlow_v3_Features.csv'))
 assert z.read(dictionary)==(R/'data/nf_ton_iot_v3/NetFlow_v3_Features.csv').read_bytes()
 bag=next(n for n in names if n.endswith('/manifest-sha1.txt'));expected=next(line.split()[0] for line in z.read(bag).decode().splitlines() if line.endswith('NF-CICIDS2018-v3.csv'))
 digest=hashlib.sha1()
 class HashReader(io.RawIOBase):
  def __init__(self,source):self.source=source
  def readable(self):return True
  def readinto(self,b):
   data=self.source.read(len(b));digest.update(data);b[:len(data)]=data;return len(data)
 source=io.BufferedReader(HashReader(z.open(member)),buffer_size=1024**2)
 cutoff=int(pd.Timestamp('2018-02-27T00:00:00Z').value//1000000);rng=np.random.default_rng(20261121);states={};counts={};offset=0;excluded=0
 cols=FEATURES+['FLOW_END_MILLISECONDS','Label','Attack']
 for frame in pd.read_csv(source,usecols=cols,chunksize=200000):
  row=np.arange(offset,offset+len(frame));eligible=frame.FLOW_END_MILLISECONDS.to_numpy()<cutoff;excluded+=int((~eligible).sum());f=frame.loc[eligible].copy();f['_row']=row[eligible];f['_priority']=rng.random(len(f))
  assert f.Label.isin([0,1]).all()
  for family in np.unique(f.Attack.to_numpy(dtype=str)):
   new=f[f.Attack==family];counts[str(family)]=counts.get(str(family),0)+len(new);cap=100000 if bool((new.Label==0).all()) else 40000
   merged=pd.concat([states[family],new],ignore_index=True) if family in states else new.reset_index(drop=True)
   if len(merged)>cap:merged=merged.iloc[np.argpartition(merged['_priority'].to_numpy(),cap-1)[:cap]].copy()
   states[family]=merged
  offset+=len(frame)
  if offset%2000000==0:print('CSE source rows audited',offset,'permitted families',counts,flush=True)
 assert digest.hexdigest()==expected,'Publisher BagIt SHA1 mismatch'
 source.close();z.close()
 selected=pd.concat(states.values(),ignore_index=True).sort_values('FLOW_END_MILLISECONDS',kind='stable');x=selected[FEATURES].to_numpy(dtype=np.float32);nonfinite={FEATURES[i]:int((~np.isfinite(x[:,i])).sum()) for i in range(len(FEATURES)) if (~np.isfinite(x[:,i])).any()};x[~np.isfinite(x)]=np.nan;y=selected.Label.to_numpy(dtype=np.uint8);family=selected.Attack.to_numpy(dtype=str);t=selected.FLOW_END_MILLISECONDS.to_numpy(dtype=np.int64)
 assert len(t) and t.max()<cutoff
 np.savez_compressed(O/'source.npz',x=x,y=y,family=family,row=selected['_row'].to_numpy(),completion_ms=t)
 report=dict(publisher_sha1=expected,payload_sha1_verified=True,source_read_mode='Direct ZIP stream with CRC and BagIt SHA1 verification; avoids another multi-gigabyte raw CSV read.',created_at=datetime.datetime.now().astimezone().isoformat(),scope=__doc__,raw_rows=offset,cutoff_ms=cutoff,excluded_rows_by_time=excluded,permitted_family_population=counts,cached_family_counts={str(k):int(v) for k,v in zip(*np.unique(family,return_counts=True))},cache_rows=len(y),cache_attacks=int(y.sum()),nonfinite_to_native_missing=nonfinite,feature_dictionary_identical=True,first_ms=int(t.min()),last_ms=int(t.max()),protected_march1_labels_summarized=False,row_sha256=hashlib.sha256(selected['_row'].to_numpy().tobytes()).hexdigest())
 (O/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
