"""Label-assisted diagnostic of deleting model-input conflicts; never adopted.

No row is removed from a training/evaluation cache, and no model is refitted.
This quantifies why aggregate aliasing must not be silently treated as noise.
"""
import json
import numpy as np,joblib
from run_network_context_study import R,D,O,development_inputs,holdouts
from network_context_features import transform
from run_multienvironment_training import metrics

def main():
    train,_,_=development_inputs();stress=holdouts()['ton_stress'];f=json.loads((D/'manifest.json').read_text())['features'];out=[]
    for period,d in [('pooled_training',train),('ton_stress_development',stress)]:
        for variant in ['physical14','context29']:
            x=transform(d['x'],d['ports'],f,variant);vv=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf))
            keys=vv.view(np.dtype((np.void,vv.dtype.itemsize*vv.shape[1]))).ravel();_,g=np.unique(keys,return_inverse=True)
            a=np.bincount(g,weights=d['y']);b=np.bincount(g,weights=1-d['y']);drop=((a>0)&(b>0))[g]
            c=json.loads((O/f'{variant}_seed42_selection.json').read_text())['selected']['shared_window'];m=joblib.load(R/c['artifact']);score=m.booster_.predict(x,num_threads=1)
            fam={str(v):dict(total=int((d['family']==v).sum()),
                    counterfactual_removed=int(((d['family']==v)&drop).sum())) for v in np.unique(d['family'])}
            groups={}
            for name,mask in [('all_rows',np.ones(len(x),bool)),('would_be_retained',~drop),('would_be_removed',drop)]:
                if not mask.any():continue
                groups[name]=metrics(d['y'][mask],score[mask],c['thresholds']['shared'],d['family'][mask],windows=3)
            out.append(dict(period=period,variant=variant,rows=len(x),counterfactual_removed_rows=int(drop.sum()),family=fam,metrics=groups))
    (O/'conflict_filter_counterfactual.json').write_text(json.dumps(dict(scope=__doc__,results=out,
        warning='These label-assisted partitions are explanatory counterfactuals, not approved data cleaning, validation sets, deployment metrics or evidence of annotation errors. Original full-row results remain primary.'),indent=2))
    print('COMPLETE no-deletion counterfactual',[(r['period'],r['variant'],r['counterfactual_removed_rows']) for r in out])
if __name__=='__main__':main()
