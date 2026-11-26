"""In-sample label-assisted upper bound for deterministic vector classifiers.

Not a trained model, not a deployment threshold and not a generalization bound.
An exact bounded 0/1 knapsack maximizes attacks detected while allowing <=floor
(0.001 * benign rows) false positives. Identical input vectors share one decision.
"""
import json
import numpy as np
from run_network_context_study import D,O,development_inputs,holdouts
from network_context_features import transform

def information_bound(x,y,cap=.001):
    values=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf),dtype=np.float32)
    keys=values.view(np.dtype((np.void,values.dtype.itemsize*values.shape[1]))).ravel()
    _,inv=np.unique(keys,return_inverse=True)
    n=int(inv.max())+1
    attack=np.bincount(inv,weights=y,minlength=n).astype(int)
    benign=np.bincount(inv,weights=1-y,minlength=n).astype(int)
    mixed=(attack>0)&(benign>0);budget=int(np.floor(cap*(y==0).sum()))
    free=int(attack[benign==0].sum());dp=np.zeros(budget+1,dtype=np.int64)
    for cost in range(1,budget+1):
        # All selected groups with the same cost are interchangeable except value.
        candidates=np.sort(attack[(benign==cost)&(attack>0)])[::-1][:budget//cost]
        for value in candidates:dp[cost:]=np.maximum(dp[cost:],dp[:-cost]+value)
    tp=free+int(dp.max())
    return dict(rows=len(y),distinct_vectors=n,mixed_vectors=int(mixed.sum()),
        rows_in_mixed_vectors=int((attack[mixed]+benign[mixed]).sum()),
        attacks_in_mixed_vectors=int(attack[mixed].sum()),
        benign_in_mixed_vectors=int(benign[mixed].sum()),false_positive_budget=budget,
        label_assisted_optimal_tp=tp,label_assisted_optimal_recall=tp/int(y.sum()),
        free_attack_only_vectors_tp=free,
        scope='Optimistic in-sample exact lookup/knapsack diagnostic. Uses these period labels. Cannot be deployed or interpreted as independent test performance.')

def main():
    train,_,_=development_inputs();h=holdouts();f=json.loads((D/'manifest.json').read_text())['features'];out=[]
    for env,d in [('training',train),*h.items()]:
        for variant in ['physical14','direction15','service26','protocol19','context29','harmonized_context29']:
            x=transform(d['x'],d['ports'],f,variant)
            out.append(dict(environment=env,variant=variant,**information_bound(x,d['y'])))
        print('COMPLETE representation information diagnostic',env,flush=True)
    (O/'information_bound_audit.json').write_text(json.dumps(dict(scope=__doc__,results=out),indent=2))
if __name__=='__main__':main()
