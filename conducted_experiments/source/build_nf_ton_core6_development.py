"""Prepare a separately trained NF-ToN-IoT six-field development replay."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["PROTOCOL", "FLOW_DURATION_MILLISECONDS", "IN_PKTS", "OUT_PKTS",
            "IN_BYTES", "OUT_BYTES"]

def main():
    n = 27520260
    x = np.empty((n,len(FEATURES)), dtype=np.int64)
    y = np.empty(n,dtype=np.uint8)
    t = np.empty(n,dtype=np.int64)
    source = ROOT/"data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv"
    start=0
    for chunk in pd.read_csv(source,usecols=FEATURES+["Label","FLOW_END_MILLISECONDS"],chunksize=250000):
        end=start+len(chunk)
        vals=chunk[FEATURES].to_numpy(dtype=np.int64)
        assert (vals>=0).all() and chunk["Label"].isin([0,1]).all()
        x[start:end]=vals
        y[start:end]=chunk["Label"]
        t[start:end]=chunk["FLOW_END_MILLISECONDS"].to_numpy(dtype=np.int64)*10**6
        start=end
    assert start==n and (t[1:]>=t[:-1]).all()
    boundaries=np.unique(np.r_[0,np.searchsorted(t,t[19999::20000],side="right"),n])
    assert (t[boundaries[1:-1]-1]<t[boundaries[1:-1]]).all()
    folder=ROOT/"data/nf_ton_iot_v3/core6_development"
    folder.mkdir(exist_ok=True)
    np.savez(folder/"stream.npz",features=x,labels=y,completion_ns=t,chunk_offsets=boundaries,
             source_row=np.arange(n,dtype=np.int32))
    report=dict(rows=n,attacks=int(y.sum()),chunks=len(boundaries)-1,features=FEATURES,
                scope="Previously explored NetFlow collection. Fit separate models; native units. Development only.")
    (folder/"manifest.json").write_text(json.dumps(report,indent=2))
    print(report)

if __name__=="__main__":
    main()
