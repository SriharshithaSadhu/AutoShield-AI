"""Evaluator-only attack adjacency audit. Original labels/FPR never redefined.

Ground-truth attack history is unavailable to deployment. This audit tests an
association, not a deployable safeguard, annotation error or causal mechanism.
Strictly earlier completion timestamps exclude all same-time/future attacks.
"""
import json,datetime
from time import perf_counter
import numpy as np,pandas as pd
from behavioral_data import addresses
from run_behavioral_comparison import D,O,load_part

class PriorAttacks:
    def __init__(self):self.table={}
    def before(self,host,t):
        value=self.table.get(host)
        if value is None:return -1
        return value[0] if value[0]<t else value[1]
    def add(self,host,t):
        value=self.table.get(host)
        if value is None:self.table[host]=(t,-1)
        elif value[0]<t:self.table[host]=(t,value[0])
        elif value[0]>t:raise ValueError('not chronological')

def main():
    data=load_part(D,'evaluation','ton');wanted={int(row):j for j,row in enumerate(data['row'])}
    gaps=np.full((len(data['y']),4),-1,np.int64);seen=np.zeros(len(gaps),bool);source=PriorAttacks();destination=PriorAttacks();offset=0;wall=perf_counter()
    # Full original chronology is needed for past-only evaluator context.
    for frame in pd.read_csv(D.parent/'NF-ToN-IoT-v3_endtime_ordered.csv',usecols=['FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','Label'],chunksize=250000):
        tt=frame.FLOW_END_MILLISECONDS.to_numpy(np.int64);ss=addresses(frame.IPV4_SRC_ADDR);dd=addresses(frame.IPV4_DST_ADDR);yy=frame.Label.to_numpy(np.uint8)
        for i,(t,s,d,y) in enumerate(zip(tt.tolist(),ss.tolist(),dd.tolist(),yy.tolist())):
            j=wanted.get(offset+i)
            if j is not None:
                assert data['metadata']['t'][j]==t and int(data['metadata']['source'][j])==s and data['y'][j]==y
                past=[source.before(s,t),destination.before(s,t),source.before(d,t),destination.before(d,t)]
                gaps[j]=[t-p if p>=0 else -1 for p in past];seen[j]=True
            if y:source.add(s,t);destination.add(d,t)
        offset+=len(frame)
    assert seen.all() and offset==27520260
    np.save(O/'stress_probe_prior_attack_gaps_ms.npy',gaps)
    out=[]
    for c in json.loads((O/'selections.json').read_text()):
        if c['seed']!=42:continue
        chosen=c['selected']['shared_pooled']
        with np.load(O/(c['identity']+'_shared_pooled_scores.npz')) as z:score=z['ton_stress']
        pred=score>chosen['thresholds']['shared'];benign=data['y']==0
        for window in [10000,60000]:
            near=np.any((gaps>=0)&(gaps<=window),axis=1)
            for name,mask in [('prior_attack_endpoint',near),('no_prior_attack_endpoint',~near)]:
                b=benign&mask;fp=int(pred[b].sum());out.append(dict(method=c['method'],history_ms=window,group=name,benign=int(b.sum()),fp=fp,
                    conditional_fpr=fp/int(b.sum()) if b.any() else None,share_of_all_false_positives=fp/max(1,int(pred[benign].sum()))))
    result=dict(scope=__doc__,recorded_at=datetime.datetime.now().astimezone().isoformat(),source_rows=offset,scored_probe_rows=len(gaps),wall_s=perf_counter()-wall,results=out,
        caveats=['All labels remain original. Overall FPR and deployment feasibility unchanged.',
                 'Endpoint adjacency is an association; high adjacency may also be prevalent among correctly classified benign flows.',
                 'Completion-time order remains an idealized sensor receipt proxy. No protected network processed.'])
    (O/'false_alarm_attack_adjacency_audit.json').write_text(json.dumps(result,indent=2));print('COMPLETE prior-attack adjacency audit',round(result['wall_s'],2),flush=True)
if __name__=='__main__':main()
