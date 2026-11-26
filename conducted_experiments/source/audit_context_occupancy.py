from pathlib import Path
import json,numpy as np
R=Path('.');D=R/'data/nf_ton_iot_v3/detection_foundation'
with np.load(R/'data/nf_ton_iot_v3/core6_development/stream.npz') as z:t=z['completion_ns']
ends=np.r_[np.flatnonzero(t[:-1]!=t[1:]),len(t)-1];counts=np.diff(np.r_[-1,ends]);i=int(np.argmax(counts))
# Exact max completed flows over each one-second timestamp bucket. Window includes earlier events only.
u=t[ends];starts=np.searchsorted(t,u-1000000000,side='left');prior=np.searchsorted(t,u,side='left');active=prior-starts
out=dict(rows=len(t),max_simultaneous_completions=int(counts[i]),max_prior_one_second_flows=int(active.max()),scope='Full unlabelled completion-time trace; bounds observed state occupancy, not Python object memory or Pi speed.')
(R/'results/detection_foundation/context_full_trace_occupancy.json').write_text(json.dumps(out,indent=2));print(out)
