"""Build causal histories from every prior permitted record, not sampled rows.

NF prefix stops at its old tune boundary. Later replay resumes the saved state.
UNSW protected row IDs are filtered before state updates or cache extraction.
"""
from pathlib import Path
from time import perf_counter,process_time
import json,pickle,datetime,hashlib,argparse
import numpy as np,pandas as pd,psutil
from behavioral_observation import BehavioralObservation,FEATURE_NAMES
from behavioral_data import META_COLUMNS,META_DTYPE,metadata,event_columns
from run_network_context_study import R,D,U
O=R/'results/behavioral_observation_2026-11-22'

def targets(folder,names):
    result={}
    for name in names:
        with np.load(folder/(name+'.npz')) as z:ids=z['row'].copy()
        order=np.argsort(ids);result[name]=dict(ids=ids,sorted_ids=ids[order],order=order,
            values=np.zeros((len(ids),26),np.float32),meta=np.empty(len(ids),dtype=META_DTYPE),seen=np.zeros(len(ids),bool))
    return result

def save_targets(data,source):
    audit={}
    for name,d in data.items():
        assert d['seen'].all(),(source,name,int(d['seen'].sum()))
        np.save(O/f'{source}_{name}_behavior.npy',d['values']);np.save(O/f'{source}_{name}_metadata.npy',d['meta'])
        audit[name]=dict(rows=len(d['ids']),row_sha256=hashlib.sha256(d['ids'].tobytes()).hexdigest())
    return audit

def replay_rows(engine,m,data,ordered_rows=True):
    # Only probe queries are expensive; updates include every observed record.
    wanted={}
    for name,d in data.items():
        if ordered_rows:
            a,b=np.searchsorted(d['sorted_ids'],[int(m['row'][0]),int(m['row'][-1])+1]);ix=d['order'][a:b]
            lookup={int(row):int(j) for row,j in zip(d['ids'][ix],ix)}
        else:lookup={int(row):int(j) for j,row in enumerate(d['ids'])}
        for row,j in lookup.items():wanted.setdefault(row,[]).append((name,j))
    for i,event in enumerate(zip(*event_columns(m))):
        row=int(m['row'][i]);query=row in wanted;v=engine.observe(*event,query=query)
        if query:
            feature=(*v,*engine.global_features())
            for name,j in wanted[row]:data[name]['values'][j]=feature;data[name]['meta'][j]=m[i];data[name]['seen'][j]=True

def nf():
    out=O/'nf_cache_audit.json'
    if out.exists():print('NF caches already complete');return
    engine=BehavioralObservation();data=targets(D,['train','tune']);stop=json.loads((D/'manifest.json').read_text())['boundaries'][2]
    source=D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv';offset=0;wall=perf_counter();cpu=process_time()
    for frame in pd.read_csv(source,usecols=META_COLUMNS,chunksize=100000):
        n=min(len(frame),stop-offset);m=metadata(frame.iloc[:n],offset)
        assert (np.diff(m['t'])>=0).all();replay_rows(engine,m,data);offset+=n
        if offset%2000000==0:print('NF behavioral prefix',offset,'elapsed',round(perf_counter()-wall,1),flush=True)
        if offset==stop:break
    assert offset==stop
    with (O/'nf_prefix_state.pkl').open('wb') as f:pickle.dump(engine,f,pickle.HIGHEST_PROTOCOL)
    report=dict(source_rows=offset,targets=save_targets(data,'ton'),state=engine.diagnostics(),
        wall_s=perf_counter()-wall,cpu_s=process_time()-cpu,peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),
        arrival_assumption='Completion time is an idealized ordered receipt-time proxy. No export/collector receipt timestamps are supplied.',
        cache_columns=FEATURE_NAMES+['log_global_record_rate10s','log_global_record_rate60s'],labels_used_for_state=False)
    out.write_text(json.dumps(report,indent=2));print('COMPLETE NF behavioral prefix cache',report['wall_s'],flush=True)

def unsw():
    out=O/'unsw_cache_audit.json'
    if out.exists():print('UNSW caches already complete');return
    audit=json.loads((U.parent/'development_audit.json').read_text());n_allowed=audit['bounds'][-2]
    protected=np.load(U/'protected_row_ids.npy');excluded=np.zeros(audit['rows'],bool);excluded[protected]=True
    pieces=[];offset=0
    for frame in pd.read_csv(U.parent/'NF-UNSW-NB15-v3.csv',usecols=META_COLUMNS,chunksize=100000):
        m=metadata(frame,offset);keep=~excluded[m['row']];pieces.append(m[keep]);offset+=len(frame)
    m=np.concatenate(pieces);del pieces;assert len(m)==n_allowed
    m=m[np.argsort(m['t'],kind='stable')];np.save(O/'unsw_allowed_metadata.npy',m)
    data=targets(U,['train','calibration','selection','development_holdout']);engine=BehavioralObservation();wall=perf_counter();cpu=process_time()
    # One shared lookup avoids rebuilding a390k-row map per chunk.
    wanted={int(row):(name,j) for name,d in data.items() for j,row in enumerate(d['ids'])}
    for i,event in enumerate(zip(*event_columns(m))):
        row=int(m['row'][i]);query=row in wanted;v=engine.observe(*event,query=query)
        if query:
            name,j=wanted[row];data[name]['values'][j]=(*v,*engine.global_features());data[name]['meta'][j]=m[i];data[name]['seen'][j]=True
    report=dict(source_rows=len(m),protected_records_processed=0,targets=save_targets(data,'unsw'),state=engine.diagnostics(),
        wall_s=perf_counter()-wall,cpu_s=process_time()-cpu,peak_resident_bytes=psutil.Process().memory_info()._asdict().get('peak_wset'),
        sort='Stable completion time within permitted80% prefix only; idealized arrival proxy.',labels_used_for_state=False)
    out.write_text(json.dumps(report,indent=2));print('COMPLETE UNSW behavioral cache',report['wall_s'],flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',choices=['ton','unsw']);args=ap.parse_args();O.mkdir(exist_ok=True)
    (nf if args.source=='ton' else unsw)()
if __name__=='__main__':main()
