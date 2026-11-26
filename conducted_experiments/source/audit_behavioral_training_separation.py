"""Exact cached-vector conflicts, fitting data only. Not a generalization bound.

Temporal floating-point features naturally distinguish records. Removing exact
collisions is not evidence of transferable attack semantics or a novel method.
"""
import json
import numpy as np
from run_behavioral_comparison import D,O,inputs,representation

def conflicts(x,y,family):
    a=np.array(x,dtype=np.float32,order='C',copy=True);a[a==0]=0.;a[np.isnan(a)]=np.float32(np.nan)
    keys=a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel();_,inv,count=np.unique(keys,return_inverse=True,return_counts=True)
    attacks=np.bincount(inv,weights=y,minlength=len(count));bad=(attacks>0)&(attacks<count);affected=bad[inv]
    return dict(rows=len(y),distinct_vectors=len(count),conflicting_vectors=int(bad.sum()),affected_rows=int(affected.sum()),
        affected_benign=int((affected&(y==0)).sum()),affected_attacks=int((affected&(y==1)).sum()),
        attack_families={str(f):int((affected&(y==1)&(family==f)).sum()) for f in np.unique(family[y==1])})

def main():
    train,_,_=inputs();f=json.loads((D/'manifest.json').read_text())['features'];out=[]
    for method in ['raw46','flow14','behavior24','hybrid38']:
        x=train['x'] if method=='raw46' else representation(train,method,f)
        for network,mask in [('pooled',np.ones(len(train['y']),bool)),('ton',train['env']==0),('unsw',train['env']==1)]:
            out.append(dict(representation=method,environment=network,**conflicts(x[mask],train['y'][mask],train['family'][mask])))
    (O/'training_separation_audit.json').write_text(json.dumps(dict(scope=__doc__,results=out,
        caveat='Float32 caches of the fixed200k fitting sample only. This does not supersede the earlier raw-text full-prefix audit and does not use evaluation labels.'),indent=2))
    print('TRAINING COLLISION AUDIT',[(r['representation'],r['environment'],r['affected_rows']) for r in out],flush=True)
if __name__=='__main__':main()
