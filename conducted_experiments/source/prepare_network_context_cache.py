"""Extract only transport ports for already-permitted development row IDs.

Does not parse labels, IP addresses, or protected row identifiers. Source row IDs
must match the cached raw exporter order, not a newly sorted/reindexed CSV.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
O=R/'results/network_context_2026-11-21'

def extract(csv, folders):
    targets={}
    for folder,name in folders:
        with np.load(folder/(name+'.npz')) as z:ids=z['row'].copy()
        order=np.argsort(ids);sorted_ids=ids[order]
        assert len(np.unique(ids))==len(ids)
        targets[str(folder/name)]=dict(ids=ids,order=order,sorted_ids=sorted_ids,
            ports=np.zeros((len(ids),2),dtype=np.uint16),found=np.zeros(len(ids),dtype=bool))
    offset=0
    for frame in pd.read_csv(csv,usecols=['L4_SRC_PORT','L4_DST_PORT'],chunksize=250000):
        # Explicit column order; never rely on CSV header order.
        v=frame[['L4_SRC_PORT','L4_DST_PORT']].to_numpy()
        for target in targets.values():
            a,b=np.searchsorted(target['sorted_ids'],[offset,offset+len(frame)])
            if a==b:continue
            dest=target['order'][a:b];values=v[target['sorted_ids'][a:b]-offset]
            assert np.isfinite(values).all() and ((values>=0)&(values<=65535)&(values==np.floor(values))).all()
            target['ports'][dest]=values.astype(np.uint16);target['found'][dest]=True
        offset+=len(frame)
        if all(t['found'].all() for t in targets.values()):break
        if offset%2500000==0:print(csv.name,offset,'rows scanned',flush=True)
    audit={}
    for stem,t in targets.items():
        assert t['found'].all(),stem
        path=Path(stem+'_ports.npy');np.save(path,t['ports'])
        audit[str(path.relative_to(R))]=dict(rows=len(t['ids']),
            row_sha256=hashlib.sha256(t['ids'].tobytes()).hexdigest(),
            ports_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return audit

def main():
    O.mkdir(exist_ok=True);a=R/'data/nf_ton_iot_v3/detection_foundation';b=R/'data/nf_unsw_nb15_v3/development'
    audit=dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        scope=__doc__,protected_ids_loaded=False,columns=['L4_SRC_PORT','L4_DST_PORT'],sources={})
    for csv,parts in [(a.parent/'NF-ToN-IoT-v3_endtime_ordered.csv',[(a,n) for n in ['train','tune','evaluation']]),
                       (b.parent/'NF-UNSW-NB15-v3.csv',[(b,n) for n in ['train','calibration','selection','development_holdout']])]:
        audit['sources'][str(csv.relative_to(R))]=extract(csv,parts)
    (O/'port_cache_audit.json').write_text(json.dumps(audit,indent=2))
    print('COMPLETE allowed development transport-port cache',flush=True)
if __name__=='__main__':main()
