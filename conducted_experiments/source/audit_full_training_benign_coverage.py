"""Do later false alarms match benign vectors present in the full training pool?

Pure explanatory lookup: no new feature, sample selection, classifier or policy.
Exact matching is not geometric OOD detection and does not show retraining helps.
"""
from collections import defaultdict
import json,sqlite3,zlib
import numpy as np
from run_network_context_study import D,O,development_inputs,holdouts
from network_context_features import transform
from context_from_audit_signature import reconstruct

def keys(x):
    a=np.ascontiguousarray(np.nan_to_num(x,nan=-np.inf),dtype=np.float32)
    return a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel().tolist()

def main():
    folder=O/'full_training_role_text_audit';census=json.loads((folder/'results.json').read_text());features=census['protocol']['features']
    conn=sqlite3.connect('file:'+str((folder/'groups.sqlite').resolve())+'?mode=ro',uri=True)
    cursor=conn.execute('SELECT k,benign FROM groups WHERE benign>0');full={v:defaultdict(int) for v in ['physical14','context29']};processed=0
    while rows:=cursor.fetchmany(10000):
        raw=[];sig=[];counts=[]
        for blob,n in rows:
            text,signature=zlib.decompress(blob).rsplit(b'|context=',1)
            raw.append(text.decode('utf-8').split('|'));sig.append(list(signature));counts.append(n)
        raw=np.array(raw,dtype=np.float32);sig=np.array(sig,dtype=np.uint8)
        assert raw.shape[1]==46 and sig.shape[1]==3
        for variant in full:
            x=reconstruct(raw,sig,features,variant)
            for key,n in zip(keys(x),counts):full[variant][key]+=int(n)
        processed+=len(rows)
    conn.close()
    assert all(sum(table.values())==census['benign'] for table in full.values())
    train,_,_=development_inputs();h=holdouts();records=[]
    for variant in full:
        source_benign=(train['env']==0)&(train['y']==0)
        sample_keys=set(keys(transform(train['x'][source_benign],train['ports'][source_benign],features,variant)))
        for environment in ['ton_future','ton_stress']:
            d=h[environment];query=keys(transform(d['x'],d['ports'],features,variant))
            choice=json.loads((O/f'{variant}_seed42_selection.json').read_text())['selected']['shared_window']
            with np.load(O/f'{variant}_seed42_shared_window_scores.npz') as z:s=z[environment]
            pred=s>choice['thresholds']['shared'];benign=d['y']==0
            groups={}
            in_sample=np.array([k in sample_keys for k in query]);in_full=np.array([k in full[variant] for k in query])
            # Additional full-pool keys may collapse after representation; any
            # sample benign key must therefore also occur in the full census.
            assert all(k in full[variant] for k in sample_keys)
            for name,mask in [('all_benign',benign),('false_positive_benign',benign&pred),('missed_attacks',(d['y']==1)&~pred)]:
                groups[name]=dict(rows=int(mask.sum()),exact_sample_benign_match=int((mask&in_sample).sum()),
                    no_sample_match_but_full_benign_match=int((mask&~in_sample&in_full).sum()),
                    no_full_benign_match=int((mask&~in_full).sum()))
            records.append(dict(variant=variant,environment=environment,groups=groups))
    (O/'full_training_benign_coverage.json').write_text(json.dumps(dict(scope=__doc__,
        full_benign_population=census['benign'],source_groups_read=processed,
        full_representation_vector_counts={v:len(t) for v,t in full.items()},results=records,
        caveat='A matching full-training benign descriptor need not be label-pure. Other attacks and temporal/relational meaning may share it. This lookup does not authorize hard-negative retraining or dropping attacks, and exact nonmatches are not proof of being outside geometric training support.'),indent=2))
    print('COMPLETE full benign-pool coverage diagnostic',records)
if __name__=='__main__':main()
