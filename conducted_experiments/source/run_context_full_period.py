"""Frozen full-period diagnostic: actual 1000-flow batch scoring, no adaptation.

NF-v3 was inspected previously; this expands development evaluation to every
flow in the final chronological period and is not independent confirmation.
"""
from pathlib import Path
from time import perf_counter,process_time
from collections import defaultdict
import json,hashlib
import numpy as np,pandas as pd,joblib,psutil
from threadpoolctl import threadpool_limits
from causal_flow_context import CausalFlowContext
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation';O=R/'results/context_full_period'
START=18360007;N=27520260

def add(target,y,p,f):
 target['benign']+=int((y==0).sum());target['attacks']+=int(y.sum());target['fp']+=int(p[y==0].sum());target['tp']+=int(p[y==1].sum())
 for family in np.unique(f[y==1]):
  mask=(f==family)&(y==1);bucket=target['family'].setdefault(str(family),dict(n=0,tp=0));bucket['n']+=int(mask.sum());bucket['tp']+=int(p[mask].sum())
def bucket():return dict(benign=0,attacks=0,fp=0,tp=0,family={})

def main():
 O.mkdir(exist_ok=True);features=json.loads((D/'manifest.json').read_text())['features'];meta=['FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_DST_PORT'];cols=features+meta+['Label','Attack']
 ab=json.loads((R/'results/causal_context_ablation/results.json').read_text())['results'];models={};policies={};frozen={}
 for context in (False,True):
  path=R/'results/causal_context_ablation'/f'model_context{context}_balancedFalse_seed42.joblib';models[context]=joblib.load(path)
  frozen[str(context)]=dict(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),policies={})
  for policy in ('pooled','worst_block'):
   r=next(v for v in ab if v['context']==context and not v['family_balanced'] and v['seed']==42 and v['cap']==.001 and v['policy']==policy)
   policies[(context,policy)]=r['threshold'];frozen[str(context)]['policies'][policy]=r['threshold']
 (O/'frozen_configuration.json').write_text(json.dumps(dict(start_row=START,stop_row=N,batch_flows=1000,models=frozen,scope=__doc__),indent=2))
 stats={str(k):bucket() for k in policies};flow_windows={str(k):defaultdict(bucket) for k in policies};time_windows={str(k):defaultdict(bucket) for k in policies}
 state=CausalFlowContext();offset=0;first_time=None;timings=[];read_s=0.;compute_cpu=0.;start=perf_counter();rss_peak=psutil.Process().memory_info().rss
 reader=pd.read_csv(R/'data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=cols,chunksize=100000)
 while True:
  read_start=perf_counter()
  try:frame=next(reader)
  except StopIteration:break
  read_s+=perf_counter()-read_start;x=frame[features].to_numpy(dtype=np.float32);metadata=list(frame[meta].itertuples(index=False,name=None));labels=frame['Label'].to_numpy(dtype=np.uint8);family=frame['Attack'].astype(str).to_numpy()
  for lo in range(0,len(frame),1000):
   hi=min(lo+1000,len(frame));wall=perf_counter();cpu=process_time();cx=np.asarray([state.observe(*v) for v in metadata[lo:hi]],dtype=np.float32)
   first=max(lo,START-offset)
   if first>=hi:continue
   sel=slice(first,hi);relative=first-lo;xx=np.ascontiguousarray(x[sel]);cc=cx[relative:];y=labels[sel];f=family[sel]
   times=np.asarray([v[0] for v in metadata[first:hi]],dtype=np.int64)
   if first_time is None:first_time=int(times[0])
   fw=(np.arange(offset+first,offset+hi)-START)//20000;tw=(times-first_time)//600000
   scores={False:models[False].booster_.predict(xx,num_threads=1),True:models[True].booster_.predict(np.column_stack((xx,cc)),num_threads=1)}
   for key,threshold in policies.items():
    name=str(key);pred=scores[key[0]]>threshold;add(stats[name],y,pred,f)
    for wid in np.unique(fw):
     m=fw==wid;add(flow_windows[name][int(wid)],y[m],pred[m],f[m])
    for wid in np.unique(tw):
     m=tw==wid;add(time_windows[name][int(wid)],y[m],pred[m],f[m])
   timings.append((perf_counter()-wall)*1000);compute_cpu+=process_time()-cpu
  offset+=len(frame);rss_peak=max(rss_peak,psutil.Process().memory_info().rss)
  if offset%2000000==0:
   print('full period processed raw rows',offset,flush=True);(O/'progress.json').write_text(json.dumps(dict(raw_rows=offset,stats=stats),indent=2))
 assert offset==N
 result=dict(scope=__doc__,raw_rows=offset,evaluated_rows=N-START,stats=stats,flow_windows=flow_windows,time_windows_600s=time_windows,
  resources=dict(wall_seconds=perf_counter()-start,csv_read_wall_s=read_s,evaluation_batch_compute_cpu_s=compute_cpu,actual_batch_compute_p95_ms=float(np.quantile(timings,.95)),actual_batch_compute_p99_ms=float(np.quantile(timings,.99)),rss_peak_observed=rss_peak,max_live_context_flows=state.max_live),
  caveats=['One seed, two fixed models, four fixed policies. Context extraction uses all past unlabelled flows.','Batch time includes history extraction, two model predictions, four threshold policies and metric bookkeeping; CSV parsing is separate.','This batch benchmark does not measure network transport, queue waiting, alert delivery or Raspberry Pi performance.'])
 (O/'results.json').write_text(json.dumps(result,indent=2));print('complete full-period',result['stats'],flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
