"""Reassess existing negative history experiment; no new feature or model trial."""
import json,hashlib
import numpy as np
from run_network_context_study import D,O,development_inputs
from audit_context_information_bound import information_bound

def main():
    train,_,_=development_inputs();ix=(train['env']==0)&((train['y']==0)|(train['family']=='scanning'))
    with np.load(D/'train.npz') as z:ids=z['row']
    order=np.argsort(ids);pos=np.searchsorted(ids[order],train['row'][ix]);aligned=order[pos]
    assert np.array_equal(ids[aligned],train['row'][ix])
    path=D/'train_causal_context.npy';history=np.load(path)[aligned]
    x=np.column_stack([train['x'][ix],history])
    result=information_bound(x,train['y'][ix])
    (O/'previous_history_observability_audit.json').write_text(json.dumps(dict(scope=__doc__,
        existing_history_features=json.loads((D/'causal_context_manifest.json').read_text())['features'],
        existing_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=result,
        earlier_negative_result_report='results/generalization_study_2026-11-19/REPORT.md',
        interpretation='Prior history separates many training vectors, but earlier chronological history classifiers still failed badly under shift. In-sample observability is necessary information, not evidence of generalization. Do not repeat the same history feature search or claim this is a successful detector.'),indent=2))
    print('Existing history diagnostic, NOT deployment performance:',result['label_assisted_optimal_recall'])
if __name__=='__main__':main()
