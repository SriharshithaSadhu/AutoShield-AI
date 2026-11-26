"""Desktop-only completed-flow state benchmark; includes Python state, excludes CSV IO."""
from pathlib import Path
from time import perf_counter,process_time
import json,threading
import pandas as pd,numpy as np,psutil
from causal_flow_context import CausalFlowContext
R=Path(__file__).resolve().parent
cols=['FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_DST_PORT']
f=pd.read_csv(R/'data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=cols,nrows=200000)
records=list(f[cols].itertuples(index=False,name=None));state=CausalFlowContext();rss0=psutil.Process().memory_info().rss
rss=[rss0];stop=threading.Event()
def monitor():
 while not stop.wait(.01):rss.append(psutil.Process().memory_info().rss)
thread=threading.Thread(target=monitor);thread.start();timings=[];wall=perf_counter();cpu=process_time()
for i,row in enumerate(records):
 if i%1000==0:
  t=perf_counter();state.observe(*row);timings.append((perf_counter()-t)*1e6)
 else:state.observe(*row)
cpu=process_time()-cpu;wall=perf_counter()-wall;stop.set();thread.join()
out=dict(scope='Desktop first 200000 real chronological NF-v3 flows. CSV IO and input storage excluded from extraction timer. Every 1000th flow sampled for per-flow wall latency; process RSS includes Python imports and input records.',rows=len(records),wall_s=wall,cpu_s=cpu,flows_per_wall_second=len(records)/wall,mean_cpu_us_per_flow=cpu/len(records)*1e6,sampled_p95_us=float(np.quantile(timings,.95)),rss_before=rss0,peak_observed_rss=max(rss),incremental_rss=max(rss)-rss0,max_live_flows=state.max_live)
(R/'results/detection_foundation/causal_context_desktop_resources.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
