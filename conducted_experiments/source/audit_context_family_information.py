"""Family-specific in-sample observability diagnostics, using development only."""
import json
import numpy as np
from run_network_context_study import D,O,development_inputs,holdouts
from network_context_features import transform,service_hints,SERVICE_NAMES
from multienvironment_features import transform as original
from audit_context_information_bound import information_bound

def main():
    train,_,_=development_inputs();stress=holdouts()['ton_stress'];f=json.loads((D/'manifest.json').read_text())['features'];out=[]
    source={k:v[train['env']==0] for k,v in train.items()}
    for name,d in [('ton_training',source),('ton_stress',stress)]:
        for variant in ['raw46','physical14','context29','all_aggregates_and_frozen_service_hints']:
            if variant=='all_aggregates_and_frozen_service_hints':
                # Information audit only, never fitted or proposed as another
                # feature-search candidate. Checks whether compression hides
                # the information already present in the frozen inputs.
                context=transform(d['x'],d['ports'],f,'context29')
                x=np.column_stack([original(d['x'],f,'raw46'),context[:,14:24]])
            else:x=original(d['x'],f,'raw46') if variant=='raw46' else transform(d['x'],d['ports'],f,variant)
            for family in np.unique(d['family'][d['y']==1]):
                ix=(d['y']==0)|(d['family']==family)
                out.append(dict(environment=name,variant=variant,family=str(family),
                    **information_bound(x[ix],d['y'][ix])))
        print('COMPLETE family observability diagnostic',name,flush=True)
    (O/'family_information_audit.json').write_text(json.dumps(dict(scope=__doc__,results=out,
        caveat='Each family is contrasted with all benign rows separately. Label-assisted optimistic limits, not learned models, and not a single jointly feasible classifier for every family.'),indent=2))
    print('SCANNING',[(r['variant'],r['label_assisted_optimal_recall']) for r in out if r['family']=='scanning'])
if __name__=='__main__':main()
