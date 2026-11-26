"""Compare pinned CICFlowMeter flow-timeout outputs with official IoTID20 scan flows."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
INV=ROOT/'results/iot_nid_raw_capture_inventory_20261102.json'
OFF=ROOT/'data/provenance/IoTID20_official_original.csv'
CAPTURE='scan-portos-1-dec.pcap'
CFMDIR=ROOT/'results/iot_nid_cfm_v3_capture_batch'
OUT=ROOT/'results/iot_nid_scan_portos1_timeout_identity_20261103.json'
TIMEOUTS=(120_000_000,100_000,500)

def canon(df, sip, sport, dip, dport, proto):
    a=df[sip].astype(str)+'|'+pd.to_numeric(df[sport],errors='coerce').fillna(-1).astype('int64').astype(str)
    b=df[dip].astype(str)+'|'+pd.to_numeric(df[dport],errors='coerce').fillna(-1).astype('int64').astype(str)
    left=a.where(a<=b,b); right=b.where(a<=b,a)
    return left+'<>'+right+'|'+pd.to_numeric(df[proto],errors='coerce').fillna(-1).astype('int64').astype(str)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    inv=json.loads(INV.read_text()); c=next(x for x in inv['captures'] if x['file']==CAPTURE)
    start=pd.Timestamp(c['first_packet_utc'])-pd.Timedelta(seconds=1); end=pd.Timestamp(c['last_packet_utc'])+pd.Timedelta(seconds=1)
    off=pd.read_csv(OFF,low_memory=False)
    off['_t']=pd.to_datetime(off.Timestamp,format='%d/%m/%Y %I:%M:%S %p',errors='coerce',utc=True)+pd.Timedelta(hours=4)
    off=off.loc[off._t.between(start,end,inclusive='both')].copy()
    off['_key']=canon(off,'Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol')
    off['_sec']=off['_t'].dt.floor('s'); off['_oid']=np.arange(len(off),dtype='int64')
    records=[]
    for timeout in TIMEOUTS:
        p=CFMDIR/f'{CAPTURE}_flowTimeout_{timeout}.csv'
        cfm=pd.read_csv(p,low_memory=False)
        cfm['_t']=pd.to_datetime(cfm.Timestamp,format='%d/%m/%Y %H:%M:%S',errors='coerce',utc=True)
        cfm['_key']=canon(cfm,'Src IP','Src Port','Dst IP','Dst Port','Protocol')
        cfm['_sec']=cfm['_t'].dt.floor('s'); cfm['_cid']=np.arange(len(cfm),dtype='int64')
        available=set(off._key.dropna())
        same_tuple=cfm._key.isin(available)
        pairs=[]
        for shift in (-1,0,1):
            right=off[['_key','_oid']].copy()
            right['_sec']=off['_sec']+pd.to_timedelta(shift,unit='s')
            pairs.append(cfm[['_key','_sec','_cid']].merge(right,on=['_key','_sec'],how='inner'))
        matched=pd.concat(pairs,ignore_index=True).drop_duplicates(['_cid','_oid'])
        # Strict identity requires a unique original Flow ID and second-resolution start time in both tables.
        cfm['_idkey']=cfm['Flow ID'].astype(str)+'|'+cfm['_t'].astype(str)
        off['_idkey']=off['Flow_ID'].astype(str)+'|'+off['_t'].astype(str)
        lc=cfm['_idkey'].value_counts(); lo=off['_idkey'].value_counts()
        common=set(lc.index)&set(lo.index)
        one={k for k in common if lc[k]==1 and lo[k]==1}
        cm=cfm.loc[cfm['_idkey'].isin(one)].set_index('_idkey').sort_index()
        om=off.loc[off['_idkey'].isin(one)].set_index('_idkey').sort_index()
        cfm_features={str(col).replace(' ','_'):col for col in cfm.columns if str(col).replace(' ','_') in off.columns and str(col).replace(' ','_') not in {'Flow_ID','Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Timestamp','Label','Cat','Sub_Cat','_t','_key','_sec','_cid','_idkey'}}
        feature_results={}; exact_all=np.ones(len(one),dtype=bool)
        for name,cname in cfm_features.items():
            x=pd.to_numeric(cm[cname],errors='coerce').to_numpy(dtype='float64'); y=pd.to_numeric(om[name],errors='coerce').to_numpy(dtype='float64')
            eq=(x==y)|(np.isnan(x)&np.isnan(y)); feature_results[name]=int(eq.sum()); exact_all &= eq
        records.append({'timeout_us':timeout,'generated_rows':len(cfm),'official_rows_in_capture_window':len(off),'generated_rows_sharing_official_bidirectional_tuple':int(same_tuple.sum()),'generated_rows_with_same_tuple_and_start_within_1s':int(matched._cid.nunique()),'official_rows_matched_by_tuple_and_start_within_1s':int(matched._oid.nunique()),'strict_common_flowid_second_keys':len(common),'strict_one_to_one_flowid_second_pairs':len(one),'comparable_network_features':len(cfm_features),'exact_full_feature_vectors':int(exact_all.sum()),'exact_feature_agreements':feature_results,'cfm_csv_sha256':sha(p),'official_label_counts':off.Sub_Cat.value_counts().to_dict()})
    out={'status':'complete','capture':CAPTURE,'cfm_commit':'ea19b0783cd6647ee6cd07ad58365b506b947771','activity_timeout_us':5_000_000,'comparison':'bidirectional endpoint/protocol tuple and start-time difference <=1 second; duplicate candidate pairs collapsed by original row identifiers','official_csv_sha256':sha(OFF),'records':records,'interpretation':'Timeout changes generated flow counts. A count match alone does not reproduce feature values, row identity, or the publisher setting.'}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()


