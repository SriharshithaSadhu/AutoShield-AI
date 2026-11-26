"""Label-free deployment adapter for frozen experimental behavioral models.

Caller supplies completed bidirectional flow records in collector arrival order.
No future sorting, endpoint-valued predictors, training, labels or controllers.
This adapter does not make a detector deployment-approved or control its FPR.
"""
from dataclasses import dataclass
import math
import numpy as np
from behavioral_observation import BehavioralObservation
from behavioral_models import scores
from multienvironment_features import transform

RAW_FIELDS=('PROTOCOL','FLOW_DURATION_MILLISECONDS','IN_PKTS','OUT_PKTS','IN_BYTES','OUT_BYTES',
            'RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_PKTS','SHORTEST_FLOW_PKT','LONGEST_FLOW_PKT','TCP_FLAGS')

@dataclass(frozen=True)
class CompletedFlow:
    arrival_ms:int
    source:int
    destination:int
    destination_port:int
    raw:tuple

    def validate(self):
        for name,minimum,maximum in [('source',0,2**32-1),('destination',0,2**32-1),('destination_port',0,65535)]:
            value=getattr(self,name)
            if not minimum<=value<=maximum or int(value)!=value:raise ValueError(name)
        if int(self.arrival_ms)!=self.arrival_ms or len(self.raw)!=len(RAW_FIELDS):raise ValueError('arrival or physical schema')
        # Full feature vector may contain NaNs handled by the tree. Behavioral
        # update fields require actual observed finite nonnegative quantities.
        for i in [0,1,2,3,10]:
            if not math.isfinite(self.raw[i]) or self.raw[i]<0:raise ValueError(RAW_FIELDS[i])
        if self.raw[0]>255 or int(self.raw[0])!=self.raw[0] or self.raw[10]>65535 or int(self.raw[10])!=self.raw[10]:raise ValueError('protocol or flags')

    def event(self):
        return (self.arrival_ms,self.source,self.destination,self.destination_port,self.raw[0],self.raw[10],self.raw[2],self.raw[3],self.raw[1])

class BehavioralRuntime:
    METHODS=('flow14','behavior24','quantile_tail_anomaly','hybrid38','behavior26_global_ablation')
    def __init__(self,model,method,threshold,max_hosts=4096,contact_cap=64,max_batch=1000):
        if method not in self.METHODS or not math.isfinite(threshold):raise ValueError('method or threshold')
        if not isinstance(max_batch,int) or max_batch<1:raise ValueError('positive batch cap')
        self.model=model;self.method=method;self.threshold=float(threshold)
        self.max_batch=max_batch
        self.observation=BehavioralObservation(max_hosts,contact_cap) if method!='flow14' else None
        self.last_arrival=None

    def score_batch(self,records):
        """Return scores/alarms for current records, preserving caller order.

        Batch construction deadlines and collector IO belong to the caller;
        this function performs actual feature extraction and native inference.
        Prevalidate the whole batch before mutating history on invalid input.
        """
        if not records:return dict(scores=np.empty(0),alerts=np.empty(0,dtype=bool))
        if len(records)>self.max_batch:raise ValueError('batch exceeds configured memory cap')
        last=self.last_arrival
        for r in records:
            r.validate()
            if last is not None and r.arrival_ms<last:raise ValueError('collector arrival order must be monotone')
            last=r.arrival_ms
        raw=np.array([r.raw for r in records],dtype=np.float32)
        if self.observation is not None:
            values=[]
            for r in records:
                v=self.observation.observe(*r.event())
                values.append((*v,*self.observation.global_features()) if self.method=='behavior26_global_ablation' else v)
            behavior=np.asarray(values,dtype=np.float32)
        if self.method=='flow14':x=transform(raw,list(RAW_FIELDS),'physical14')
        elif self.method=='hybrid38':x=np.column_stack([transform(raw,list(RAW_FIELDS),'physical14'),behavior])
        else:x=behavior
        s=scores(self.model,x);self.last_arrival=last
        return dict(scores=s,alerts=s>self.threshold)
