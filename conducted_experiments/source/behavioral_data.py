"""Label-free exporter metadata conversion; IPs remain grouping keys only."""
import ipaddress
import numpy as np,pandas as pd
META_COLUMNS=['FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_DST_PORT','PROTOCOL',
              'TCP_FLAGS','IN_PKTS','OUT_PKTS','FLOW_DURATION_MILLISECONDS']
META_DTYPE=np.dtype([('t','i8'),('source','u4'),('destination','u4'),('port','u2'),('protocol','u2'),
                     ('flags','u2'),('in_packets','f8'),('out_packets','f8'),('duration','f8'),('row','i8')])

def addresses(values):
    codes,unique=pd.factorize(values,sort=False)
    if (codes<0).any():raise ValueError('Missing endpoint key')
    decoded=np.array([int(ipaddress.IPv4Address(str(v))) for v in unique],dtype=np.uint32)
    return decoded[codes]

def metadata(frame,offset):
    out=np.empty(len(frame),dtype=META_DTYPE)
    out['t']=frame.FLOW_END_MILLISECONDS.to_numpy(dtype=np.int64)
    out['source']=addresses(frame.IPV4_SRC_ADDR);out['destination']=addresses(frame.IPV4_DST_ADDR)
    for field,column,maximum in [('port','L4_DST_PORT',65535),('protocol','PROTOCOL',255),('flags','TCP_FLAGS',65535)]:
        v=frame[column].to_numpy();assert np.isfinite(v).all() and ((v>=0)&(v<=maximum)&(v==np.floor(v))).all();out[field]=v
    for field,column in [('in_packets','IN_PKTS'),('out_packets','OUT_PKTS'),('duration','FLOW_DURATION_MILLISECONDS')]:
        v=frame[column].to_numpy(dtype=float);assert np.isfinite(v).all() and (v>=0).all();out[field]=v
    out['row']=np.arange(offset,offset+len(frame));return out

def event_columns(m):
    return [m[name].tolist() for name in ['t','source','destination','port','protocol','flags','in_packets','out_packets','duration']]
