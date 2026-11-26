"""Training-only source contradictions and fitted-score discrimination audit."""
import json
import numpy as np,joblib
from run_network_context_study import R,D,O,development_inputs
from network_context_features import transform
from multienvironment_features import transform as old_transform
from run_multienvironment_training import metrics,ranking_metrics

def collision_counts(x,y,env):
    v=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf),dtype=np.float32)
    k=v.view(np.dtype((np.void,v.dtype.itemsize*v.shape[1]))).ravel();_,inv=np.unique(k,return_inverse=True);ng=int(inv.max())+1
    counts=np.zeros((ng,2,2),dtype=np.int64)
    np.add.at(counts,(inv,env,y),1)
    total=counts.sum(axis=1);mixed=(total[:,0]>0)&(total[:,1]>0)
    within=(counts[:,:,0]>0)&(counts[:,:,1]>0)
    cross_only=mixed&~within.any(axis=1)
    return dict(vectors=ng,mixed_vectors=int(mixed.sum()),mixed_rows=int(total[mixed].sum()),
        ton_internal_mixed_vectors=int(within[:,0].sum()),unsw_internal_mixed_vectors=int(within[:,1].sum()),
        only_cross_environment_conflict_vectors=int(cross_only.sum()),
        only_cross_environment_conflict_rows=int(total[cross_only].sum()),
        note='Conflicts mean identical recorded aggregates have different labels. They do not prove annotation error.')

def main():
    train,_,_=development_inputs();f=json.loads((D/'manifest.json').read_text())['features'];conflicts=[];performance=[]
    for variant in ['raw46','physical14','context29']:
        x=old_transform(train['x'],f,'raw46') if variant=='raw46' else transform(train['x'],train['ports'],f,variant)
        conflicts.append(dict(variant=variant,**collision_counts(x,train['y'],train['env'])))
        if variant=='raw46':continue
        chosen=json.loads((O/f'{variant}_seed42_selection.json').read_text())['selected']['shared_window'];m=joblib.load(R/chosen['artifact']);s=m.booster_.predict(x,num_threads=1)
        for e in [0,1]:
            ix=train['env']==e
            performance.append(dict(variant=variant,training_environment='ton' if e==0 else 'unsw',
                ranking=ranking_metrics(train['y'][ix],s[ix]),
                **metrics(train['y'][ix],s[ix],chosen['thresholds']['shared'],train['family'][ix],windows=3)))
    (O/'training_conflict_audit.json').write_text(json.dumps(dict(scope=__doc__,collisions=conflicts,
        fitted_training_performance=performance,
        caveat='Training scores are in-sample diagnostics, not validation results. No rows are dropped or relabeled.'),indent=2))
    print('COMPLETE training conflict audit',conflicts)
if __name__=='__main__':main()
