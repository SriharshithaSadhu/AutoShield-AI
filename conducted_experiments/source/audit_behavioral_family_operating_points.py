"""Undeployable benign-label-assisted family ranking diagnostic.

Never adopts these thresholds. Uses development benign labels to identify what
the fitted score ranking can achieve, separating coverage from calibration.
"""
import json
import numpy as np
from run_behavioral_comparison import O,D,load_part,inputs

def main():
    train,_,_=inputs();d=load_part(D,'evaluation','ton');y=d['y'];family=d['family'];out=[]
    seen=set(train['family'][(train['env']==0)&(train['y']==1)])
    for c in json.loads((O/'selections.json').read_text()):
        s0=c['selected']['shared_pooled']
        with np.load(O/(c['identity']+'_shared_pooled_scores.npz')) as z:s=z['ton_stress']
        benign=np.sort(s[y==0]);allowed=int(np.floor(.001*len(benign)));tau=float(benign[-allowed-1]);pred=s>tau
        assert int(pred[y==0].sum())<=allowed
        for f in np.unique(family[y==1]):
            ix=(y==1)&(family==f)
            out.append(dict(method=c['method'],seed=c['seed'],family=str(f),n=int(ix.sum()),ton_fitting_family_name_seen=f in seen,
                frozen_recall=float((s[ix]>s0['thresholds']['shared']).mean()),oracle_cap_recall=float(pred[ix].mean()),
                oracle_fpr=float(pred[y==0].mean()),oracle_threshold=tau))
    (O/'family_operating_point_diagnostic.json').write_text(json.dumps(dict(scope=__doc__,results=out,
        caveat='Thresholds use evaluation benign labels and are unusable for prospective deployment. Family-name absence is not proof of an unseen technique across taxonomies.'),indent=2))
    print('COMPLETE family operating-point diagnostic',len(out),flush=True)
if __name__=='__main__':main()
