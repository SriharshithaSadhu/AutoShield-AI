"""Training-only context audit; domain predictability is not proof of label leakage."""
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from network_context_features import transform,service_hints,SERVICES
from run_network_context_study import R,D,U,O,part

def summarize(x,ports,f):
    h=service_hints(x,ports,f)
    return dict(rows=len(x),known_fraction=float(h['known'].mean()),
        source_service_fraction=float(h['source_service_hint'].mean()),
        destination_service_fraction=float(h['destination_service_hint'].mean()),
        category_counts={str(i):int((h['category']==i).sum()) for i in range(10)})

def main():
    f=json.loads((D/'manifest.json').read_text())['features'];rng=np.random.default_rng(20261121)
    raw=[];ports=[];env=[];stats={}
    for e,(name,folder) in enumerate([('ton',D),('unsw',U)]):
        d=part(folder,'train');stats[name]={str(y):summarize(d['x'][d['y']==y],d['ports'][d['y']==y],f) for y in [0,1]}
        ids=np.sort(rng.choice(np.flatnonzero(d['y']==0),20000,False))
        raw.append(d['x'][ids]);ports.append(d['ports'][ids]);env.append(np.full(len(ids),e))
    raw=np.concatenate(raw);ports=np.concatenate(ports);env=np.concatenate(env);domains=[]
    keys=np.ascontiguousarray(np.nan_to_num(raw,nan=-1)).view(np.dtype((np.void,raw.dtype.itemsize*raw.shape[1]))).ravel()
    _,groups=np.unique(keys,return_inverse=True)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=42).split(raw,env,groups))
    for variant in ['physical14','direction15','service26','protocol19','context29','harmonized_context29']:
        full=transform(raw,ports,f,variant)
        arrays={'all_features':full}
        if full.shape[1]>14:arrays['context_only']=full[:,14:]
        for scope,x in arrays.items():
            # Same raw-aggregate group split for every feature ablation. Holding
            # out entire coarse categories instead would test unseen-category
            # extrapolation, not domain predictability of known service hints.
            if len(np.unique(env[tr]))<2 or len(np.unique(env[te]))<2:
                domains.append(dict(variant=variant,scope=scope,auc=None,reason='Group-disjoint split lacks one domain.'));continue
            m=LGBMClassifier(n_estimators=50,num_leaves=15,min_child_samples=50,n_jobs=1,verbosity=-1,random_state=42)
            m.fit(x[tr],env[tr]);p=m.booster_.predict(x[te],num_threads=1)
            domains.append(dict(variant=variant,scope=scope,auc=float(roc_auc_score(env[te],p)),
                train_rows=len(tr),test_rows=len(te),unique_feature_vectors=len(np.unique(groups))))
    target=O/'training_context_audit.json'
    if target.exists() and not (O/'training_context_audit_pre_split_correction.json').exists():
        (O/'training_context_audit_pre_split_correction.json').write_bytes(target.read_bytes())
    target.write_text(json.dumps(dict(scope=__doc__,
        split='One fixed GroupShuffleSplit of raw46 aggregate fingerprints for all variants and context-only controls; categories may occur in both splits.',
        selected_services=SERVICES,by_environment_and_class=stats,benign_domain_classification=domains,
        caveat='High domain AUC shows network distributions remain distinguishable. It does not prove capture-identity leakage or that a context feature is invalid.'),indent=2))
    print('COMPLETE training-only service/domain audit',domains,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
