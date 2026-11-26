"""Reconstruct frozen model inputs from their sufficient audit signature."""
import numpy as np
from network_context_features import canonicalize,protocol_context
from multienvironment_features import transform as physical

def reconstruct(raw,signature,features,variant):
    if variant=='physical14':return physical(raw,features,'physical14')
    if variant!='context29':raise ValueError(variant)
    xx=canonicalize(raw,features,dict(source_service_hint=signature[:,1].astype(bool)))
    base=physical(xx,features,'physical14');onehot=np.column_stack([signature[:,0]==i for i in range(1,10)])
    known=(signature[:,1]>0)|(signature[:,2]>0)
    return np.ascontiguousarray(np.column_stack([base,onehot,known,protocol_context(xx,features)]),dtype=np.float32)
