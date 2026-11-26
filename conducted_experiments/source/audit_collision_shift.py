"""Diagnose conditional label shift for exact aggregate-feature collisions."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;D=ROOT/'data/nf_ton_iot_v3/detection_foundation'
def main():
 data={}
 for name in ('train','tune','evaluation'):
  with np.load(D/(name+'.npz')) as z:data[name]=(pd.util.hash_pandas_object(pd.DataFrame(z['x']),index=False).to_numpy(),z['y'],z['family'])
 h,y,_=data['train'];table=pd.DataFrame(dict(hash=h,y=y)).groupby('hash').y.agg(['count','sum']);table['rate']=table['sum']/table['count']
 conflicts=table[(table['sum']>0)&(table['sum']<table['count'])];result={}
 for name in ('tune','evaluation'):
  eh,ey,ef=data[name];ev=pd.DataFrame(dict(hash=eh,y=ey)).groupby('hash').y.agg(['count','sum']);ev['rate']=ev['sum']/ev['count']
  both=conflicts.join(ev,lsuffix='_train',rsuffix='_later',how='inner')
  reversal=((both.rate_train>=.5)!=(both.rate_later>=.5))
  pred=np.array([table.at[v,'rate']>=.5 if v in table.index else False for v in eh]);hit=np.isin(eh,conflicts.index)
  result[name]=dict(shared_conflicting_vectors=len(both),majority_reversals=int(reversal.sum()),
   rows_in_reversed_vectors=int(both.loc[reversal,'count_later'].sum()),weighted_absolute_attack_rate_change=float(np.average(abs(both.rate_train-both.rate_later),weights=both.count_later)),
   training_majority_rule=dict(fpr=float(pred[ey==0].mean()),recall=float(pred[ey==1].mean()),
     collision_rows=int(hit.sum()),collision_fp=int((pred&hit&(ey==0)).sum()),collision_tp=int((pred&hit&(ey==1)).sum())))
 out=ROOT/'results/detection_foundation/conditional_collision_shift.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
