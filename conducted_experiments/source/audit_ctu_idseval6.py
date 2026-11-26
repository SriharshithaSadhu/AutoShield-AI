from pathlib import Path
import collections, hashlib, json
ROOT=Path(__file__).resolve().parent/'data'/'ctu_idseval_6'
EXPECTED={'README.md':('3dad0e775de07f8854f7f5d01c795cfb',2868),'CTU-IDSEVAL-6-summary.csv':('e408b8431e38428e8f9dfd0538986254',519),'labels.zip':('a017b6f47fe0603595cec7d643f7e4a2',17486),'zeek.zip':('33b57c813053eed12ec3738b65f27ef0',7291205)}
files={}
for name,(md5,size) in EXPECTED.items():
 p=ROOT/name; data=p.read_bytes(); got=hashlib.md5(data).hexdigest(); files[name]={'bytes':len(data),'md5':got,'expected_md5':md5,'checksum_ok':got==md5 and len(data)==size}
summary=[]
for p in sorted((ROOT/'zeek'/'zeek').glob('*.log')):
 fields=None; n=0; bad=0; labels=collections.Counter(); details=collections.Counter(); ts=[]; missing=collections.Counter()
 with p.open(encoding='utf-8',errors='replace') as f:
  for line in f:
   if line.startswith('#fields\t'): fields=line.rstrip('\n').split('\t')[1:]; continue
   if line.startswith('#') or not line.strip(): continue
   row=line.rstrip('\n').split('\t'); n+=1
   if fields is None or len(row)!=len(fields): bad+=1; continue
   d=dict(zip(fields,row)); labels[d.get('label','')]+=1; details[d.get('detailedlabel','')]+=1
   try: ts.append(float(d['ts']))
   except (ValueError,KeyError): bad+=1
   for k in ('duration','orig_bytes','resp_bytes'): missing[k]+=d.get(k,'') in ('-','(empty)','')
 summary.append({'file':p.name,'rows':n,'columns':len(fields or []),'fields':fields,'row_or_timestamp_errors':bad,'labels':dict(labels),'detailed_top':details.most_common(8),'timestamp_min':min(ts) if ts else None,'timestamp_max':max(ts) if ts else None,'unset_count':dict(missing),'capture_open':next((x.split('\t',1)[1] for x in p.read_text(errors='replace').splitlines() if x.startswith('#open\t')),None)})
actual=sum(x['rows'] for x in summary)
out={'dataset':'CTU-IDSEVAL-6','zenodo_record':'https://zenodo.org/records/21027042','license':'CC BY 4.0 (record metadata)','files':files,'summary_csv':(ROOT/'CTU-IDSEVAL-6-summary.csv').read_text(encoding='utf-8').splitlines(),'capture_audits':summary,'total_rows_actual':actual,'total_rows_claimed_summary_csv':286704,'row_count_delta_actual_minus_summary':actual-286704,'interpretation':{'temporal_replay':'reject: capture-local timestamps are mixed relative/epoch scales; captures are separate behavior groups and span multiple years; no continuous labelled stream','external_static_validation':'not accepted for headline model validation without a capture-held-out protocol; only one benign capture and label rules are capture/IP-specific; Zeek feature schema differs from CICFlowMeter model input','label_semantics':'three labels include Background; preserve as a separate/unknown class or exclude explicitly, never merge silently with benign','provenance':'label configs use source/destination IP and ports, creating risk that naive models learn capture/rule artifacts'}}
(ROOT/'audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'files':files,'rows':[(x['file'],x['rows'],x['labels'],x['timestamp_min'],x['timestamp_max'],x['unset_count']) for x in summary],'actual_total':actual,'summary_total':286704,'delta':actual-286704},indent=2))
