"""Validate cached causal features and quantify training-only aliasing reduction."""
from pathlib import Path
import json,numpy as np,pandas as pd
from causal_flow_context import CausalFlowContext
R=Path(__file__).resolve().parent;D=R/'data/nf_ton_iot_v3/detection_foundation'
def conflicts(x,y):
 h=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
 t=pd.DataFrame(dict(h=h,y=y)).groupby('h').y.agg(['count','sum']);t=t[(t['sum']>0)&(t['sum']<t['count'])]
 return dict(vectors=len(t),rows=int(t['count'].sum()))
def main():
 with np.load(D/'train.npz') as z:x,y,ids=z['x'],z['y'],z['row']
 c=np.load(D/'train_causal_context.npy');idx={int(row):i for i,row in enumerate(ids) if row<200000}
 cols=['FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_DST_PORT']
 f=pd.read_csv(R/'data/nf_ton_iot_v3/NF-ToN-IoT-v3_endtime_ordered.csv',usecols=cols,nrows=200000)
 state=CausalFlowContext();errors=[]
 for row,record in enumerate(f[cols].itertuples(index=False,name=None)):
  values=state.observe(*record)
  if row in idx:errors.append(float(np.max(np.abs(np.asarray(values)-c[idx[row]]))))
 assert errors and max(errors)==0,max(errors)
 out=dict(reference_compared_rows=len(errors),max_absolute_difference=max(errors),baseline=conflicts(x,y),augmented=conflicts(np.column_stack((x,c)),y),scope='Training rows only for feature-label collision counts; independent class implementation compared against stream-builder cache on first 200000 raw flows.')
 (R/'results/detection_foundation/causal_context_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
