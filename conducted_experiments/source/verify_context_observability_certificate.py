"""Independently verify the scanning information bound with HiGHS MILP.

The model class is deterministic functions of these recorded per-flow vectors.
This certificate is empirical, not a statement about every possible network
context, future data, randomized/stateful detectors, or annotation correctness.
"""
from fractions import Fraction
import json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from run_network_context_study import D,O,development_inputs
from network_context_features import transform
from multienvironment_features import transform as old

def main():
    d,_,_=development_inputs();ix=(d['env']==0)&((d['y']==0)|(d['family']=='scanning'))
    raw=d['x'][ix];ports=d['ports'][ix];y=d['y'][ix];f=json.loads((D/'manifest.json').read_text())['features']
    previously=json.loads((O/'family_information_audit.json').read_text())['results'];results=[]
    for variant in ['raw46','context29','all_aggregates_and_frozen_service_hints']:
        c=transform(raw,ports,f,'context29')
        x=old(raw,f,'raw46') if variant=='raw46' else c if variant=='context29' else np.column_stack([old(raw,f,'raw46'),c[:,14:24]])
        x=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf));keys=x.view(np.dtype((np.void,x.dtype.itemsize*x.shape[1]))).ravel()
        _,g=np.unique(keys,return_inverse=True);a=np.bincount(g,weights=y).astype(int);b=np.bincount(g,weights=1-y).astype(int)
        budget=int(np.floor(.001*(y==0).sum()));free=int(a[b==0].sum());eligible=(a>0)&(b>0)&(b<=budget)
        av,bv=a[eligible],b[eligible]
        # Independent solver uses all eligible groups; no DP dominance pruning.
        result=milp(c=-av.astype(float),integrality=np.ones(len(av)),bounds=Bounds(0,1),
            constraints=LinearConstraint(bv[None,:].astype(float),-np.inf,budget),
            options=dict(time_limit=60,mip_rel_gap=0.))
        assert result.success and result.status==0,result.message
        chosen=np.rint(result.x).astype(int);assert np.isin(chosen,[0,1]).all()
        fp=int(bv@chosen);tp=free+int(av@chosen);assert fp<=budget
        reference=next(r for r in previously if r['environment']=='ton_training' and r['variant']==variant and r['family']=='scanning')
        assert tp==reference['label_assisted_optimal_tp']
        # Exact fractional-knapsack upper bound, independently reconstructed.
        ordered=sorted(zip(av.tolist(),bv.tolist()),key=lambda z:Fraction(z[0],z[1]),reverse=True)
        remaining=budget;upper=Fraction(free)
        for attack,cost in ordered:
            take=min(remaining,cost);upper+=Fraction(attack*take,cost);remaining-=take
            if remaining==0:break
        assert Fraction(tp)<=upper
        results.append(dict(variant=variant,attacks=int(y.sum()),benign=int((y==0).sum()),
            fp_budget=budget,optimal_tp=tp,witness_fp=fp,optimal_recall=tp/int(y.sum()),
            independently_verified_by='scipy.optimize.milp / HiGHS, zero optimality gap',
            solver_groups=len(av),solver_status=int(result.status),mip_gap=float(result.mip_gap),
            exact_fractional_upper=dict(numerator=upper.numerator,denominator=upper.denominator),
            fractional_upper_recall=float(upper/int(y.sum()))))
    (O/'scanning_observability_certificate.json').write_text(json.dumps(dict(scope=__doc__,results=results,
        statement='On this fixed training sample, any deterministic classifier using only the stated vector must assign the same decision to identical vectors. Its TP count under the empirical FP budget cannot exceed the verified optimum.',
        exclusions=['Not a future-risk bound.','Not a universal impossibility claim for IDS.','Not evidence of annotation errors.',
                    'Does not cover raw endpoint identity, richer payload, or temporal/relational state.','Does not prove overall recall50% impossible; this certificate concerns the scanning family.']),indent=2))
    print('COMPLETE independent observability certificates',results)
if __name__=='__main__':main()
