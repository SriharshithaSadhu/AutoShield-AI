"""Training-only consistency and overhead audit for iterative weight updates."""
from pathlib import Path
from time import process_time,perf_counter
import json,numpy as np,joblib
from threadpoolctl import threadpool_limits
from run_multienvironment_training import R,O,D,U,read,weights
from multienvironment_features import transform

def main():
 t=read(D/'train.npz');u=read(U/'train.npz');rng=np.random.default_rng(20261121);it=np.sort(rng.choice(len(t['y']),150000,False));iu=np.sort(rng.choice(len(u['y']),50000,False));raw=np.r_[t['x'][it],u['x'][iu]];y=np.r_[t['y'][it],u['y'][iu]];env=np.r_[np.zeros(len(it),int),np.ones(len(iu),int)];base,gi,gc=weights(np.asarray([f'{e}:{v}' for e,v in zip(env,y)]));features=json.loads((D/'manifest.json').read_text())['features'];records=[]
 for rep in ('raw46','physical14'):
  x=transform(raw,features,rep)
  for method in ('group_dro','hard_negative'):
   for seed in (42,43,44):
    original=json.loads((O/f'{rep}_{method}_seed{seed}_selection.json').read_text());q=np.ones(len(gc))/len(gc);current=base.copy();cpu=0.;wall=0.
    for record in original['candidates']:
     w=base*np.minimum(q[gi]*len(gc),20) if method=='group_dro' else current.copy();w/=w.mean()
     assert abs(w.min()-record['weight_min'])<1e-10 and abs(w.max()-record['weight_max'])<1e-10
     model=joblib.load(R/record['artifact']);c=process_time();s=perf_counter();p=np.clip(model.booster_.predict(x,num_threads=1),1e-7,1-1e-7)
     if method=='group_dro':
      loss=-(y*np.log(p)+(1-y)*np.log1p(-p));gl=np.bincount(gi,weights=loss)/gc;q*=np.exp(.5*gl);q/=q.sum();assert np.allclose(q,record['next_group_mass'],rtol=0,atol=1e-12)
     else:current=base*np.where(y==0,1+4*p,1);current/=current.mean()
     cpu+=process_time()-c;wall+=perf_counter()-s
    records.append(dict(representation=rep,method=method,seed=seed,update_cpu_s_remeasured=cpu,update_wall_s_remeasured=wall,original_model_fit_cpu_s=sum(v['fit_cpu_s'] for v in original['candidates']),weight_reconstruction='exact within 1e-10; GroupDRO q within 1e-12'))
 (O/'training_weight_audit.json').write_text(json.dumps(dict(scope='Independent reconstruction using training inputs/labels only. Weight-update component timing is a remeasurement, not the original complete-run CPU time. Validation/calibration and artifact IO excluded.',records=records),indent=2),encoding='utf-8');print('PASS',len(records),'training-only update reconstructions')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
