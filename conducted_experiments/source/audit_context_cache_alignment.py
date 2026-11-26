"""Independent raw-row alignment check of permitted port caches and four fields."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'results/network_context_2026-11-21'
F=json.loads((R/'data/nf_ton_iot_v3/detection_foundation/manifest.json').read_text())['features']
CHECK=['PROTOCOL','IN_BYTES','OUT_BYTES','FLOW_DURATION_MILLISECONDS']

def main():
    groups=[('nf_ton_iot_v3','detection_foundation','NF-ToN-IoT-v3_endtime_ordered.csv',['train','tune','evaluation']),
            ('nf_unsw_nb15_v3','development','NF-UNSW-NB15-v3.csv',['train','calibration','selection','development_holdout'])]
    report={}
    for dataset,sub,csv,names in groups:
        folder=R/'data'/dataset/sub;targets={}
        for name in names:
            with np.load(folder/(name+'.npz')) as z:
                ids=z['row'];order=np.argsort(ids);expected=z['x'][:,[F.index(c) for c in CHECK]]
            targets[name]=dict(ids=ids[order],order=order,expected=expected,
                ports=np.load(folder/(name+'_ports.npy')),checked=0)
        offset=0
        for frame in pd.read_csv(folder.parent/csv,usecols=['L4_SRC_PORT','L4_DST_PORT']+CHECK,chunksize=250000):
            for name,t in targets.items():
                a,b=np.searchsorted(t['ids'],[offset,offset+len(frame)])
                if a==b:continue
                dest=t['order'][a:b];rows=frame.iloc[t['ids'][a:b]-offset]
                np.testing.assert_equal(rows[['L4_SRC_PORT','L4_DST_PORT']].to_numpy().astype(np.uint16),t['ports'][dest])
                np.testing.assert_equal(rows[CHECK].to_numpy(dtype=np.float32),t['expected'][dest])
                t['checked']+=len(dest)
            offset+=len(frame)
            if all(t['checked']==len(t['ids']) for t in targets.values()):break
        report[dataset]={name:dict(rows=t['checked'],four_original_fields_exact=True,ports_exact=True) for name,t in targets.items()}
        assert all(t['checked']==len(t['ids']) for t in targets.values())
        print('PASS independent source alignment',dataset,flush=True)
    (O/'cache_alignment_audit.json').write_text(json.dumps(dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        protected_inputs_loaded=False,checked_raw_columns=CHECK,results=report),indent=2))
if __name__=='__main__':main()
