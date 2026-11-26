"""Account for every full-period row, label and per-family/window outcome."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent

def main():
 d=json.loads((R/'results/context_full_period/results.json').read_text())
 with np.load(R/'data/nf_ton_iot_v3/core6_development/stream.npz') as z:y=z['labels'][18360007:]
 expected_a=int(y.sum());expected_b=int((y==0).sum());checks=0
 for name,s in d['stats'].items():
  assert s['attacks']==expected_a and s['benign']==expected_b
  assert sum(v['n'] for v in s['family'].values())==s['attacks']
  assert sum(v['tp'] for v in s['family'].values())==s['tp']
  for key in ('flow_windows','time_windows_600s'):
   w=d[key][name].values()
   for metric in ('benign','attacks','fp','tp'):assert sum(v[metric] for v in w)==s[metric];checks+=1
  checks+=4
 out=dict(status='pass',evaluation_rows=len(y),attacks=expected_a,benign=expected_b,accounting_checks=checks)
 (R/'results/context_full_period/accounting_validation.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
