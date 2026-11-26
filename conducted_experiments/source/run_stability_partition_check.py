"""Fixed exploratory replication: five group partitions, unchanged q10 selector."""
import json
from functools import lru_cache
from pathlib import Path
from time import process_time
import numpy as np
from scipy.stats import binom
from run_tolerance_threshold_audit import tolerance_threshold

ROOT=Path(__file__).resolve().parent;BANK=ROOT/'results/matched_selection_budget_forests';OUT=ROOT/'results/selection_stability'
@lru_cache(None)
def allowed(n,cap):
    lo,hi=0,n-1
    if binom.cdf(0,n,cap)>.05:raise ValueError('insufficient support')
    while lo<hi:
        mid=(lo+hi+1)//2
        if binom.cdf(mid,n,cap)<=.05:lo=mid
        else:hi=mid-1
    return lo

def weighted_threshold(scores,weight,cap):
    order=np.argsort(scores);cum=np.cumsum(weight[order]);n=int(cum[-1])
    return scores[order[np.searchsorted(cum,n-allowed(n,cap),side='left')]]

def main():
    protocol=dict(status='exploratory replication after original development results',partition_seeds=list(range(100,105)),
        bootstrap_replicates=200,quantile=.1,cap=.001,minimum_recall=.5,
        note='Same frozen models and development evaluation; no new network or confirmation. No tuning of the lower-decile rule.')
    (OUT/'partition_protocol.json').write_text(json.dumps(protocol,indent=2))
    source=json.loads((BANK/'results.json').read_text())
    with np.load(ROOT/'results/expanded_benign_calibration/benign_partitions.npz') as z:h=z['calibration_hashes']
    hashes=np.concatenate((h[h%4==0],h[h%4==2]));groups=np.unique(hashes)
    records=[];start=process_time()
    for partition_seed in protocol['partition_seeds']:
        rng=np.random.default_rng(partition_seed);chosen=rng.permutation(groups)[:len(groups)//2]
        sel=np.isin(hashes,chosen);_,inv=np.unique(hashes[sel],return_inverse=True);g=int(inv.max()+1)
        weights=[np.bincount(rng.integers(g,size=g),minlength=g)[inv] for _ in range(200)]
        for run in source['runs']:
            seed=run['seed'];candidates=[];scorebank={}
            for name in run['eligible']:
                with np.load(BANK/f'{name}_seed{seed}_scores.npz') as z:
                    calibration=np.concatenate((z['selection_benign'],z['final_benign']))
                    benign=calibration[sel];attack=np.sort(z['tuning_attack'])
                    scorebank[name]=(calibration[~sel],z['evaluation_benign'].copy(),z['evaluation_attack'].copy())
                order=np.argsort(benign);ordered=benign[order];recalls=[]
                for weight in weights:
                    cum=np.cumsum(weight[order]);n=int(cum[-1]);t=ordered[np.searchsorted(cum,n-allowed(n,.001),side='left')]
                    recalls.append(float(1-np.searchsorted(attack,t,side='right')/len(attack)))
                # Check weighted rank shortcut against expanded reference once per candidate/partition.
                assert weighted_threshold(benign,weights[0],.001)==tolerance_threshold(np.repeat(benign,weights[0]),.001,.05)['threshold']
                t=tolerance_threshold(benign,.001,.05)['threshold']
                original=next(c for c in run['decisions'][0]['candidates'] if c['name']==name)
                candidates.append(dict(name=name,bytes=original['bytes'],point=float((attack>t).mean()),q10=float(np.quantile(recalls,.1))))
            selected={method:max(candidates,key=lambda c:(c[key],-c['bytes'],c['name']))['name'] for method,key in [('point','point'),('stability','q10')]}
            # Choice fixed before thresholding evaluation scores; data previously inspected, exploratory.
            for method,name in selected.items():
                cal,benign,attack=scorebank[name];t=tolerance_threshold(cal,.001,.05)['threshold']
                fpr=float((benign>t).mean());recall=float((attack>t).mean())
                records.append(dict(partition_seed=partition_seed,model_seed=seed,method=method,name=name,
                    selection_n=int(sel.sum()),final_n=int((~sel).sum()),fpr=fpr,recall=recall,
                    cap_pass=fpr<=.001,joint_pass=fpr<=.001 and recall>=.5))
        print('finished partition',partition_seed,flush=True)
    summary=[]
    for method in ('point','stability'):
        rr=[r for r in records if r['method']==method]
        summary.append(dict(method=method,n=len(rr),cap_pass=sum(r['cap_pass'] for r in rr),joint_pass=sum(r['joint_pass'] for r in rr),
            zero_recall=sum(r['recall']==0 for r in rr),mean_recall=float(np.mean([r['recall'] for r in rr])),worst_fpr=max(r['fpr'] for r in rr)))
    out=dict(protocol=protocol,records=records,summary=summary,cpu_seconds=process_time()-start)
    (OUT/'partition_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
