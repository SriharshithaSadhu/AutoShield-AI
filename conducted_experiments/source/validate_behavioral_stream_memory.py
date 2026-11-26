"""Bounded traversal validation; preserves initial offline loader measurements."""
from time import perf_counter,process_time
from pathlib import Path
import json
import numpy as np,psutil
from behavioral_observation import BehavioralObservation
from behavioral_data import event_columns
from run_network_context_study import R,U
O=R/'results/behavioral_observation_2026-11-22'

def main():
    m=np.load(O/'unsw_allowed_metadata.npy',mmap_mode='r');wanted={};data={}
    for name in ['train','calibration','selection','development_holdout']:
        with np.load(U/(name+'.npz')) as z:ids=z['row'].copy()
        data[name]=np.zeros((len(ids),26),np.float32)
        for j,row in enumerate(ids):wanted[int(row)]=(name,j)
    engine=BehavioralObservation();wall=perf_counter();cpu=process_time()
    for a in range(0,len(m),100000):
        chunk=m[a:a+100000]
        for i,event in enumerate(zip(*event_columns(chunk))):
            row=int(chunk['row'][i]);query=row in wanted;v=engine.observe(*event,query=query)
            if query:
                name,j=wanted[row];data[name][j]=(*v,*engine.global_features())
    checked=0
    for name,x in data.items():
        np.testing.assert_equal(x,np.load(O/f'unsw_{name}_behavior.npy'));checked+=len(x)
    report=dict(rows=len(m),probe_rows_bit_exact=checked,state=engine.diagnostics(),
        wall_s=perf_counter()-wall,cpu_s=process_time()-cpu,
        peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),
        initial_offline_loader_peak_bytes=json.loads((O/'unsw_cache_audit.json').read_text())['peak_resident_bytes'],
        scope='Chunked traversal of already-sorted permitted metadata, not raw CSV sorting or an on-device benchmark. Original offline loader and all caches remain unchanged.')
    (O/'bounded_stream_memory_validation.json').write_text(json.dumps(report,indent=2));print('COMPLETE chunked traversal',report,flush=True)
if __name__=='__main__':main()
