"""Causality and boundary checks against an independent brute-force reference."""
import math,random
from causal_flow_context import CausalFlowContext

def main():
 rng=random.Random(42);rows=[];t=0
 for _ in range(2000):
  t+=rng.choice([0,0,1,50,1000,1001]);rows.append((t,rng.randrange(4),rng.randrange(3),rng.randrange(5)))
 state=CausalFlowContext();prior=[]
 for row in rows:
  t,s,d,p=row;history=[v for v in prior if t-1000<=v[0]<t]
  expected=(sum(v[1]==s for v in history),sum(v[2]==d for v in history),len({v[3] for v in history if v[1]==s}),len(history))
  got=state.observe(*row)
  assert all(abs(a-math.log1p(b))<2e-6 for a,b in zip(got,expected)),(row,got,expected)
  prior.append(row)
 try:state.observe(-1,0,0,0)
 except ValueError:pass
 else:raise AssertionError('out-of-order accepted')
 print('PASS: 2000 reference comparisons, same-time exclusion, expiry boundaries, ordering rejection')
if __name__=='__main__':main()
