"""Causal common-exporter representations. No endpoint, clock or dataset identifiers."""
import numpy as np
PHYSICAL_NAMES=['protocol','log_duration_ms','log_packets','log_bytes','log_forward_bytes_per_packet','log_backward_bytes_per_packet','backward_packet_fraction','backward_byte_fraction','forward_retransmitted_packet_fraction','backward_retransmitted_packet_fraction','log_shortest_packet','log_longest_packet','syn_present','ack_present']
def transform(x,features,mode):
 x=np.asarray(x,dtype=np.float32)
 if mode=='raw46':return np.ascontiguousarray(np.where(np.isfinite(x),x,np.nan))
 if mode not in ('physical14','harmonized14'):raise ValueError(mode)
 def c(name):return x[:,features.index(name)]
 def log(v):return np.log1p(np.where(v>=0,v,np.nan))
 def divide(a,b):
  out=np.zeros_like(a);np.divide(a,b,out=out,where=b>0);out[~np.isfinite(a)|~np.isfinite(b)]=np.nan;return out
 fpk,bpk=c('IN_PKTS'),c('OUT_PKTS');fb,bb=c('IN_BYTES'),c('OUT_BYTES');flags=c('TCP_FLAGS')
 flags_int=np.where(np.isfinite(flags),flags,0).astype(np.int64)
 syn=((flags_int&2)>0).astype(np.float32);ack=((flags_int&16)>0).astype(np.float32)
 syn[~np.isfinite(flags)]=np.nan;ack[~np.isfinite(flags)]=np.nan
 duration=c('FLOW_DURATION_MILLISECONDS')
 if mode=='harmonized14':duration=np.where(duration>=0,np.maximum(duration,1),np.nan)
 columns=[c('PROTOCOL'),log(duration),log(fpk+bpk),log(fb+bb),log(divide(fb,fpk)),log(divide(bb,bpk)),divide(bpk,fpk+bpk),divide(bb,fb+bb),divide(c('RETRANSMITTED_IN_PKTS'),fpk),divide(c('RETRANSMITTED_OUT_PKTS'),bpk),log(c('SHORTEST_FLOW_PKT')),log(c('LONGEST_FLOW_PKT')),syn,ack]
 a=np.column_stack(columns).astype(np.float32);a[~np.isfinite(a)]=np.nan;return np.ascontiguousarray(a)
