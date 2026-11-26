"""Completed-flow history with strictly earlier timestamps and no label access."""
from collections import Counter,defaultdict,deque
import numpy as np
class CausalFlowContext:
 def __init__(self,window_ms=1000):
  if window_ms<=0:raise ValueError('positive window required')
  self.window_ms=window_ms;self.queue=deque();self.pending=[];self.last_time=None
  self.sources=Counter();self.destinations=Counter();self.ports=defaultdict(Counter)
  self.max_live=0
 def observe(self,t,source,destination,destination_port):
  if self.last_time is not None and t<self.last_time:raise ValueError('completion timestamps must be ordered')
  if t!=self.last_time:
   for item in self.pending:
    tt,s,d,p=item;self.queue.append(item);self.sources[s]+=1;self.destinations[d]+=1;self.ports[s][p]+=1
   self.pending=[];self.last_time=t
   while self.queue and self.queue[0][0]<t-self.window_ms:
    _,s,d,p=self.queue.popleft();self.sources[s]-=1;self.destinations[d]-=1;self.ports[s][p]-=1
    if not self.ports[s][p]:del self.ports[s][p]
    if not self.sources[s]:del self.sources[s];del self.ports[s]
    if not self.destinations[d]:del self.destinations[d]
  values=(self.sources.get(source,0),self.destinations.get(destination,0),len(self.ports.get(source,{})),len(self.queue))
  self.pending.append((t,source,destination,destination_port));self.max_live=max(self.max_live,len(self.queue)+len(self.pending))
  return np.log1p(np.asarray(values,dtype=np.float32))
