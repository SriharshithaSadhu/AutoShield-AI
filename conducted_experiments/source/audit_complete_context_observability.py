"""Correct role-complete sample audit and verify sufficiency for frozen variants."""
import json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from run_network_context_study import D,O,development_inputs
from network_context_features import transform,VARIANTS
from context_observability_keys import full_observation_matrix
from audit_context_information_bound import information_bound

def binary_keys(x):
    v=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf),dtype=np.float32)
    return v.view(np.dtype((np.void,v.dtype.itemsize*v.shape[1]))).ravel()

def main():
    train,_,_=development_inputs();ix=(train['env']==0)&((train['y']==0)|(train['family']=='scanning'))
    raw=train['x'][ix];ports=train['ports'][ix];y=train['y'][ix];features=json.loads((D/'manifest.json').read_text())['features']
    full=full_observation_matrix(raw,ports,features);_,g=np.unique(binary_keys(full),return_inverse=True);order=np.argsort(g,kind='stable');same=g[order][1:]==g[order][:-1]
    checks=[]
    for variant in VARIANTS:
        keys=binary_keys(transform(raw,ports,features,variant))[order]
        assert not (keys[1:][same]!=keys[:-1][same]).any(),variant
        checks.append(dict(variant=variant,complete_audit_key_determines_model_inputs=True))
    bound=information_bound(full,y);ng=int(g.max())+1;a=np.bincount(g,weights=y,minlength=ng).astype(int);b=np.bincount(g,weights=1-y,minlength=ng).astype(int)
    free=int(a[b==0].sum());budget=bound['false_positive_budget'];good=(a>0)&(b>0)&(b<=budget);av,bv=a[good],b[good]
    solver=milp(-av.astype(float),integrality=np.ones(len(av)),bounds=Bounds(0,1),constraints=LinearConstraint(bv[None,:],-np.inf,budget),options=dict(time_limit=60,mip_rel_gap=0.))
    assert solver.success and solver.status==0
    pred=np.rint(solver.x).astype(int);tp=free+int(av@pred);assert tp==bound['label_assisted_optimal_tp'] and int(bv@pred)<=budget
    out=dict(scope=__doc__,definition='46 aggregates + coarse category + source/destination service-hint bits. This is an audit signature, not another fitted representation.',
        bound=bound,sufficiency_checks=checks,independent_milp=dict(status=0,mip_gap=float(solver.mip_gap),optimal_tp=tp,witness_fp=int(bv@pred)),
        correction='Earlier all_aggregates_and_frozen_service_hints audit included category/known-role only, not side. That bound is preserved as a category-only result and is not used as the full-context ceiling.',
        caveat='Fixed sample, deterministic vector classifiers. No overall-recall impossibility, population guarantee or annotation-error claim.')
    (O/'complete_context_observability_audit.json').write_text(json.dumps(out,indent=2));print('COMPLETE role-complete sample certificate',bound)
if __name__=='__main__':main()
