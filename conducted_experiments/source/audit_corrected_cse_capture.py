"""Full-file temporal, label and rate-unit audit of a CRC-verified corrected capture."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--member',default='Wednesday-28-02-2018.csv');args=parser.parse_args()
    path=ROOT/'data/cse_corrected'/args.member
    metadata=json.loads(Path(str(path)+'.download.json').read_text())
    assert metadata['status']=='complete_crc_verified' and path.stat().st_size==metadata['bytes']
    cols=['Timestamp','Flow Duration','Label','Attempted Category','Protocol','Total Fwd Packet','Total Bwd packets','Flow Packets/s','Fwd Packets/s','Bwd Packets/s','Active Max','Idle Max','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags']
    counts=Counter();labels=Counter();attempts=Counter();dates=Counter();bins=defaultdict(lambda:[0,0]);last=None
    times_min=None;times_max=None;durmax=0.;invariants={k:Counter() for k in ('total','forward','backward')}
    for frame in pd.read_csv(path,usecols=cols,chunksize=150000,low_memory=False):
        counts['rows']+=len(frame)
        for col in ('Active Max','Idle Max'):
            values=pd.to_numeric(frame[col],errors='coerce')
            counts[col+'_negative']+=int((values<0).sum())
            counts[col+'_above_1e12']+=int((values>1e12).sum())
        for col in ('Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags'):
            counts[col+'_above_one']+=int((pd.to_numeric(frame[col],errors='coerce')>1).sum())
        lab=frame['Label'].astype(str).str.strip();labels.update(lab)
        attempt=pd.to_numeric(frame['Attempted Category'],errors='coerce')
        attempts.update(frame['Attempted Category'].astype(str))
        counts['invalid_attempted_category']+=int(attempt.isna().sum())
        counts['missing_label']+=int(frame['Label'].isna().sum())
        benign=lab.str.casefold().eq('benign');attempted=attempt.ne(-1)&attempt.notna()
        primary_attack=(~benign)&(~attempted)&attempt.notna()&frame['Label'].notna()
        counts['primary_attack']+=int(primary_attack.sum());counts['primary_benign']+=int((benign|attempted).sum())
        counts['attempted_flows']+=int(attempted.sum());counts['label_nonbenign_before_attempted_mapping']+=int((~benign).sum())
        start=pd.to_datetime(frame.Timestamp,format='mixed',errors='coerce')
        duration=pd.to_numeric(frame['Flow Duration'],errors='coerce')
        counts['invalid_start']+=int(start.isna().sum());counts['negative_duration']+=int((duration<0).sum())
        counts['nonfinite_duration']+=int((~np.isfinite(duration)).sum());counts['zero_duration']+=int((duration==0).sum())
        if np.isfinite(duration).any():durmax=max(durmax,float(duration[np.isfinite(duration)].max()))
        valid=start.notna()&np.isfinite(duration)&(duration>=0)
        completion=start[valid]+pd.to_timedelta(duration[valid],unit='us')
        ns=completion.to_numpy(dtype='datetime64[ns]').astype('int64')
        if len(ns):
            counts['completion_descents_in_file']+=int((np.diff(ns)<0).sum())+int(last is not None and ns[0]<last)
            last=int(ns[-1]);lo=int(ns.min());hi=int(ns.max())
            times_min=lo if times_min is None else min(times_min,lo);times_max=hi if times_max is None else max(times_max,hi)
        dates.update(start.dropna().dt.strftime('%Y-%m-%d'))
        minute=(ns//(60*10**9))
        yy=primary_attack[valid].to_numpy(dtype=int)
        tab=pd.DataFrame({'minute':minute,'attack':yy}).groupby('minute')['attack'].agg(['sum','count'])
        for idx,row in tab.iterrows():
            bins[int(idx)][0]+=int(row['count']-row['sum']);bins[int(idx)][1]+=int(row['sum'])
        f=pd.to_numeric(frame['Total Fwd Packet'],errors='coerce').to_numpy()
        b=pd.to_numeric(frame['Total Bwd packets'],errors='coerce').to_numpy()
        du=duration.to_numpy()
        for name,col,packets in [('total','Flow Packets/s',f+b),('forward','Fwd Packets/s',f),('backward','Bwd Packets/s',b)]:
            rate=pd.to_numeric(frame[col],errors='coerce').to_numpy()
            ok=(du>0)&np.isfinite(du)&np.isfinite(rate)&np.isfinite(packets)
            match=np.isclose(rate[ok]*du[ok]/1e6,packets[ok],rtol=1e-6,atol=1e-6)
            invariants[name]['eligible']+=int(ok.sum());invariants[name]['matches']+=int(match.sum())
        if counts['rows']%1500000==0:print('audited rows',counts['rows'],flush=True)
    supported=sum(min(v)>=100 for v in bins.values());both=sum(min(v)>0 for v in bins.values())
    out=dict(member=args.member,provenance=metadata,counts=dict(counts),labels=dict(labels),attempted_categories=dict(attempts),dates=dict(dates),
             duration_unit_evidence='microsecond rate/count invariants, not exporter-arrival validation',max_duration=durmax,
             rate_invariants={k:dict(v) for k,v in invariants.items()},
             completion_min_ns=times_min,completion_max_ns=times_max,minute_bins=len(bins),both_class_minute_bins=both,supported_minute_bins=supported,
             primary_mapping='BENIGN or Attempted Category != -1 -> benign; other valid labels -> attack',
             limitations=['Single corrected capture, not independent confirmation.','Rows are not assumed chronological; replay requires completion-time sorting.',
                          'Completion time is an event-time surrogate, not verified flow-export time.','No old/new row correspondence or 77-feature compatibility established.'])
    dest=ROOT/'results/corrected_cse_gate'
    (dest/(args.member+'.full_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')
    (dest/(args.member+'.minute_bins.json')).write_text(json.dumps(bins),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
