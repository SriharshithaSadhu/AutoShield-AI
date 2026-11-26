"""Causal one-second flow context. Endpoint values are keys, never predictors."""
from collections import Counter,deque,defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/nf_ton_iot_v3/detection_foundation"

def main():
 selected={}
 arrays={}
 for name in ("train","tune","calibration","evaluation"):
  with np.load(DATA/(name+".npz")) as z:ids=z["row"]
  arrays[name]=np.zeros((len(ids),4),dtype=np.float32)
  selected.update({int(row):(name,i) for i,row in enumerate(ids)})
 queue=deque();sources=Counter();destinations=Counter();source_ports=defaultdict(Counter)
 pending=[];last_time=None;row=0
 cols=["FLOW_END_MILLISECONDS","IPV4_SRC_ADDR","IPV4_DST_ADDR","L4_DST_PORT"]
 def insert(items):
  for t,s,d,p in items:
   queue.append((t,s,d,p));sources[s]+=1;destinations[d]+=1;source_ports[s][p]+=1
 for frame in pd.read_csv(ROOT/"data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv",
                           usecols=cols,chunksize=200000):
  for t,s,d,p in frame[cols].itertuples(index=False,name=None):
   if t!=last_time:
    if last_time is not None:assert t>last_time
    insert(pending);pending=[];last_time=t
    while queue and queue[0][0]<t-1000:
     _,os,od,op=queue.popleft()
     sources[os]-=1;destinations[od]-=1;source_ports[os][op]-=1
     if not source_ports[os][op]:del source_ports[os][op]
     if not sources[os]:
      del sources[os];del source_ports[os]
     if not destinations[od]:del destinations[od]
   hit=selected.get(row)
   if hit:
    name,i=hit
    arrays[name][i]=[sources.get(s,0),destinations.get(d,0),
                      len(source_ports.get(s,{})),len(queue)]
   pending.append((t,s,d,p));row+=1
  if row%2000000==0:print("Context rows",row,flush=True)
 assert row==27520260
 for name,values in arrays.items():
  np.save(DATA/(name+"_causal_context.npy"),np.log1p(values))
 report=dict(features=["log1p_source_completed_flows_prior_1s","log1p_destination_completed_flows_prior_1s",
                       "log1p_source_distinct_destination_ports_prior_1s","log1p_total_completed_flows_prior_1s"],
             scope="Strictly earlier completion timestamps in preceding second, same-time events excluded. No labels, absolute time, or endpoint values in predictors. These behavioral features may still encode testbed artifacts.")
 (DATA/"causal_context_manifest.json").write_text(json.dumps(report,indent=2))
 print(report)
if __name__=="__main__":main()
