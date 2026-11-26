"""Attach completion times for evaluation metrics only, never model inputs."""
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation'
with np.load(R/'data/nf_ton_iot_v3/core6_development/stream.npz') as z:t=z['completion_ns']
for name in ('train','tune','calibration','evaluation'):
 with np.load(D/(name+'.npz')) as z:ids=z['row']
 np.save(D/(name+'_completion_ns.npy'),t[ids])
print('Saved strictly ordered completion timestamps for metrics; predictor arrays unchanged')
