"""Connect the new raw-vector scanning certificate to prior raw-text verification."""
import json,hashlib
import numpy as np
from run_network_context_study import R,D,O,development_inputs

def keys(x):
    x=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf),dtype=np.float32)
    return x.view(np.dtype((np.void,x.dtype.itemsize*x.shape[1]))).ravel()

def mixed_keys(x,y):
    v,inv=np.unique(keys(x),return_inverse=True);a=np.bincount(inv,weights=y);b=np.bincount(inv,weights=1-y)
    return set(v[(a>0)&(b>0)].tolist())

def main():
    with np.load(D/'train.npz') as z:old=mixed_keys(z['x'],z['y'])
    train,_,_=development_inputs();ix=(train['env']==0)&((train['y']==0)|(train['family']=='scanning'))
    new=mixed_keys(train['x'][ix],train['y'][ix]);assert new<=old
    raw=R/'results/detection_foundation/raw_conflict_audit.json';precision=R/'results/detection_foundation/float32_precision_changes.json'
    a=json.loads(raw.read_text());b=json.loads(precision.read_text())
    assert len(old)==a['cached_conflicting_vectors']==a['raw_numeric_conflicting_groups']==a['exact_text_conflicting_groups']==518
    assert b['rows_with_any_precision_change']==0 and b['rows']==a['cached_conflict_rows']
    (O/'raw_precision_support_audit.json').write_text(json.dumps(dict(scope=__doc__,
        previously_verified_training_vectors=len(old),scanning_conflicting_raw_vectors=len(new),
        scanning_mixed_vectors_subset_of_prior_verified_raw_vectors=True,
        prior_conflicting_rows_with_float32_value_changes=0,
        prior_sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw,precision]},
        caveat='Uses the prior raw-text/float64 verification of the superset; no claim that all derived log/ratio representations are free of rounding collisions.'),indent=2))
    print('PASS raw precision provenance:',len(new),'scanning conflict groups lie within the518 raw-verified groups.')
if __name__=='__main__':main()
