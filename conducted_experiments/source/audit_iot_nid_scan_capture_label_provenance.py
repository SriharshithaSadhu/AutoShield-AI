"""Audit scan-event packets against pinned CICFlowMeter and published IoTID20 rows."""
from __future__ import annotations
import csv, hashlib, json, re, subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import audit_iot_nid_tcp_packet_flow_correspondence as match

ROOT=Path(__file__).resolve().parent
FILTER=ROOT/'results/iot_nid_attack_filter_audit_20261103.json'
INVENTORY=ROOT/'results/iot_nid_raw_capture_inventory_20261102.json'
OFFICIAL=ROOT/'data/provenance/IoTID20_official_original.csv'
CFM=ROOT/'results/iot_nid_cfm_v3_capture_batch'
OUT=ROOT/'results/iot_nid_scan_capture_label_provenance_20261103.json'
REPORT=ROOT/'results/IOTNID_SCAN_CAPTURE_LABEL_PROVENANCE.md'
TIMEOUT=120_000_000
FILES={*(f'scan-hostport-{i}-dec.pcap' for i in range(1,7)),*(f'scan-portos-{i}-dec.pcap' for i in range(1,7))}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 fa=json.loads(FILTER.read_text()); inv=json.loads(INVENTORY.read_text())
 invmap={x['file']:x for x in inv['captures']}
 events=[e for e in fa['filter_events'] if e['file'] in FILES]
 cols=['Flow_ID','Timestamp','Flow_Duration','Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Label','Cat','Sub_Cat']
 allf=pd.read_csv(OFFICIAL,usecols=cols,low_memory=False)
 allf['_time']=pd.to_datetime(allf.Timestamp,format='%d/%m/%Y %I:%M:%S %p',errors='coerce',utc=True)+pd.Timedelta(hours=4)
 rec=[]
 capture_all={}; capture_matched={}
 for ev in events:
  cap=invmap[ev['file']]; pcap=match.PCAPS/ev['file']
  run=subprocess.run(['wsl','-e','tshark','-r',str(pcap),'-Y',ev['filter'],'-T','fields','-E','separator=\t','-e','frame.time_epoch','-e','ip.src','-e','tcp.srcport','-e','ip.dst','-e','tcp.dstport','-e','ip.proto'],capture_output=True,text=True,timeout=300,check=False)
  if run.returncode: raise RuntimeError(f"TShark failed {ev['file']}: {run.stderr[-1000:]}")
  total=0; packets=[]
  for row in csv.reader(run.stdout.splitlines(),delimiter='\t'):
   total+=1
   if len(row)<6 or not row[0] or not row[1] or not row[3]: continue
   packets.append({'time':datetime.fromtimestamp(float(row[0]),tz=timezone.utc),'key':match.tuple_key(row[1],row[2],row[3],row[4],row[5])})
  cpath=CFM/f'{ev["file"]}_flowTimeout_{TIMEOUT}.csv'
  cfm=pd.read_csv(cpath,low_memory=False)
  cidx=match.indexed_flows(cfm,'Src IP','Src Port','Dst IP','Dst Port','Protocol','Timestamp','Flow Duration',False)
  crows,cp,_,_=match.links(packets,cidx)
  st=datetime.fromisoformat(cap['first_packet_utc'].replace('Z','+00:00')); en=datetime.fromisoformat(cap['last_packet_utc'].replace('Z','+00:00'))
  off=allf.loc[allf._time.between(st-timedelta(seconds=1),en+timedelta(seconds=1),inclusive='both')].copy()
  oidx=match.indexed_flows(off,'Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Timestamp','Flow_Duration',True)
  orows,op,_,_=match.links(packets,oidx)
  same_tuple=sum(bool(oidx.get(pkt['key'])) for pkt in packets)
  capture_all.setdefault(ev['file'],set()).update(int(i) for i in off.index)
  capture_matched.setdefault(ev['file'],set()).update(int(i) for i in orows)
  mcounts=Counter(off.loc[list(orows),'Sub_Cat'].astype(str)) if orows else Counter()
  ucounts=Counter(off.loc[~off.index.isin(orows),'Sub_Cat'].astype(str))
  ips=re.findall(r'ip\.(?:src|dst)\s*==\s*([0-9.]+)',ev['filter'])
  scanner=ips[0] if len(ips)>0 else None; target=ips[1] if len(ips)>1 else None
  srcs=off['Src_IP'].astype(str); dsts=off['Dst_IP'].astype(str)
  endpoint_audit={'filter_source_ip':scanner,'filter_target_ip':target,'official_rows_touching_filter_source':int(((srcs==scanner)|(dsts==scanner)).sum()) if scanner else None,'official_rows_touching_filter_target':int(((srcs==target)|(dsts==target)).sum()) if target else None,'official_source_ip_top5':srcs.value_counts().head(5).to_dict(),'official_destination_ip_top5':dsts.value_counts().head(5).to_dict()}
  rec.append({'capture':ev['file'],'category':ev['category'],'event_subcategory':ev['subcategory'],'capture_sha256':cap['sha256'],'cfm_csv_sha256':sha(cpath),'expected_packets':ev['expected_attack_packets'],'tshark_filtered_packets':total,'tuple_eligible_packets':len(packets),'non_ip_or_missing_tuple_packets':total-len(packets),'cfm_flows':len(cfm),'official_interval_flows':len(off),'cfm_packet_match_count':cp,'official_packet_match_count':op,'official_same_tuple_packet_count':same_tuple,'official_same_tuple_outside_time_count':same_tuple-op,'official_matched_flow_rows':len(orows),'official_unmatched_flow_rows':len(off)-len(orows),'matched_flow_subcategories':dict(mcounts),'unmatched_flow_subcategories':dict(ucounts),'endpoint_audit':endpoint_audit})
  print(f"{ev['file']} | {ev['subcategory']}: packets={total}, tuple={len(packets)}, CFM linked={cp}, official linked={op}, rows={len(off)}, unmatched={len(off)-len(orows)}",flush=True)
 out={'status':'complete' if all(x['expected_packets']==x['tshark_filtered_packets'] for x in rec) else 'filter_count_mismatch','purpose':'Bounded packet-to-flow correspondence audit for all creator-filtered host/port/OS scan events. No official labels changed.','cfm_version':'CICFlowMeter-V3 3.0','cfm_commit':'ea19b0783cd6647ee6cd07ad58365b506b947771','flow_timeout_microseconds':TIMEOUT,'matching':'bidirectional IP/port/protocol tuple plus packet time within flow interval, one-second tolerance','csv_to_pcap_offset_hours':4,'records':rec,'aggregate':{'event_count':len(rec),'expected_packets':sum(x['expected_packets'] for x in rec),'tshark_packets':sum(x['tshark_filtered_packets'] for x in rec),'tuple_eligible_packets':sum(x['tuple_eligible_packets'] for x in rec),'non_ip_or_missing_tuple_packets':sum(x['non_ip_or_missing_tuple_packets'] for x in rec),'cfm_linked_packets':sum(x['cfm_packet_match_count'] for x in rec),'official_linked_packets':sum(x['official_packet_match_count'] for x in rec),'official_same_tuple_packets':sum(x['official_same_tuple_packet_count'] for x in rec),'official_same_tuple_outside_time':sum(x['official_same_tuple_outside_time_count'] for x in rec),'official_interval_flows':sum(x['official_interval_flows'] for x in rec),'official_matched_flow_rows':sum(x['official_matched_flow_rows'] for x in rec),'official_unmatched_flow_rows':sum(x['official_unmatched_flow_rows'] for x in rec),'official_unique_interval_flow_rows':sum(len(v) for v in capture_all.values()),'official_unique_matched_flow_rows':sum(len(v) for v in capture_matched.values()),'official_unique_unmatched_flow_rows':sum(len(capture_all[k]-capture_matched.get(k,set())) for k in capture_all)},'limitations':['The creator display filters define selected event packets, not necessarily every relevant flow or the original label aggregation.','ARP host-discovery packets have no IP 5-tuple; they are counted but excluded from tuple matching.','Capture time alignment is empirical (+4h), not independently documented by the dataset publisher.','A tuple/time match is not packet-level proof when tuple reuse and rounded timestamps are present.','Pinned CICFlowMeter default timeout is not known to be the publisher configuration.']}
 OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
 lines=['# IoT-NID scan packet-to-flow/label audit (2026-11-03)','',f"Validated {out['aggregate']['tshark_packets']:,}/{out['aggregate']['expected_packets']:,} creator-filter packet selections across {len(rec)} host/port/OS scan events. {out['aggregate']['tuple_eligible_packets']:,} selections had usable IP tuple fields; ARP and other non-IP selections are retained as unmatched packet evidence. The one-second tuple/time rule linked {out['aggregate']['official_linked_packets']:,} packets to official rows versus {out['aggregate']['cfm_linked_packets']:,} links in pinned CFM-V3 output. This is coverage, not label accuracy.",'','| Capture | Event | Filter packets | Tuple eligible | Official interval rows | Packet-linked rows | Unmatched rows | matched subcategory counts |','|---|---|---:|---:|---:|---:|---:|---|']
 for x in rec: lines.append(f"| {x['capture']} | {x['event_subcategory']} | {x['tshark_filtered_packets']:,} | {x['tuple_eligible_packets']:,} | {x['official_interval_flows']:,} | {x['official_matched_flow_rows']:,} | {x['official_unmatched_flow_rows']:,} | `{x['matched_flow_subcategories']}` |")
 lines+=['','## Interpretation','',f"Across the 12 unique capture windows, {out['aggregate']['official_unique_interval_flow_rows']:,} source rows are present; {out['aggregate']['official_unique_matched_flow_rows']:,} match any selected event packet and {out['aggregate']['official_unique_unmatched_flow_rows']:,} do not. Two filters per capture make the per-event total a double-counted diagnostic. A recent cross-dataset paper explicitly describes IoTID20 labels as capture-scenario-level ([Rodriguez et al., 2026](https://doi.org/10.1016/j.iot.2026.102030)); unmatched event-filter packets therefore do not establish individual label errors. No labels were modified.",'',"The feature-export path remains unresolved. On scan-portos-1, the official window has 20,310 rows while pinned CFM-V3 generates 1,292, 3,261, or 17,623 rows at 120 s, 100 ms, or 500 µs; none of 80, 90, or 104 one-to-one Flow-ID/second pairs matches all 76 comparable numeric features. See `audit_iot_nid_scan_timeout_identity.py`. This is a one-capture exporter diagnostic, not a publisher-configuration recovery, general audit novelty, or IDS performance claim.",'',"Reproduction: run `audit_iot_nid_scan_capture_label_provenance.py` after the pinned 12-capture CFM batch. Machine output: `iot_nid_scan_capture_label_provenance_20261103.json`.",'']
 REPORT.write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(out['aggregate'],indent=2))
if __name__=='__main__': main()
