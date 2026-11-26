"""Shared static calibration/selection data; protected suffix is not an input."""
from pathlib import Path
import numpy as np,json
from run_multienvironment_training import R,D,U,read,ton_temporal_boundaries

def calibration_selection():
 rng=np.random.default_rng(20261121)
 with np.load(D/'train.npz') as z:nt=len(z['row'])
 with np.load(U/'train.npz') as z:nu=len(z['row'])
 rng.choice(nt,150000,False);rng.choice(nu,50000,False)
 tune=read(D/'tune.npz');uc=read(U/'calibration.npz');us=read(U/'selection.npz');a,b=ton_temporal_boundaries();tc={k:v[:a] for k,v in tune.items()};cal={}
 for name,d in [('ton',tc),('unsw',uc)]:
  ix=np.sort(rng.choice(np.flatnonzero(d['y']==0),20000,False));cal[name]=d['x'][ix]
 return cal,{'ton':{k:v[a:b] for k,v in tune.items()},'unsw':us},json.loads((D/'manifest.json').read_text())['features']
def development_holdouts():
 tune=read(D/'tune.npz');_,b=ton_temporal_boundaries()
 return {'ton_future':{k:v[b:] for k,v in tune.items()},'unsw_future':read(U/'development_holdout.npz'),'ton_stress':read(D/'evaluation.npz')}
def raw_score(model,x):
 x=np.ascontiguousarray(x)
 if hasattr(model,'booster_'):return model.booster_.predict(x,raw_score=True,num_threads=1)
 return model.get_booster().inplace_predict(x,predict_type='margin')
