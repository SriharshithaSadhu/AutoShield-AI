"""Frozen-model sensitivity to observation loss and designed exporter jitter.

Uses only permitted UNSW prefix. No synthetic flow values, new labels, tuning,
or protected state updates. These are declared operational counterfactuals,
not measured exporter/collector conditions or new independent datasets.
"""
from pathlib import Path
from time import perf_counter
import json,datetime
import numpy as np,joblib
from behavioral_observation import BehavioralObservation,priority
from behavioral_data import event_columns
from run_behavioral_comparison import R,D,U,O,load_part,representation
from behavioral_models import scores
from run_multienvironment_training import metrics,ranking_metrics

def main():
    protocol=dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        scenarios=['half_visibility','bounded_export_jitter_1000ms'],
        mask='Uniform deterministic row-hash parity, independent of labels. Only observed records update state and are scored.',
        jitter='arrival=end+hash(row)%1001ms, stable receipt-time sort; record fields unchanged. Designed scenario, not measured delay.',
        guard='Jitter scores exclude arrivals in final1000ms of permitted stream, to avoid a boundary needing unobserved future records.',
        model='Frozen seed42 shared_pooled selections; no recalibration or refitting.',
        units='FPR/recall conditional on observed records, plus unconditional attack coverage. Same observed subset for every method.')
    (O/'sensor_sensitivity_protocol.json').write_text(json.dumps(protocol,indent=2))
    m=np.load(O/'unsw_allowed_metadata.npy',mmap_mode='r');d=load_part(U,'development_holdout','unsw')
    wanted={int(row):j for j,row in enumerate(d['row'])};features=json.loads((D/'manifest.json').read_text())['features']
    choices=[c for c in json.loads((O/'selections.json').read_text()) if c['seed']==42];out=[]
    # Vectorized invertible hash matches the state admission mixer, used here
    # solely as a deterministic sampling/arrival scenario, never a predictor.
    h=m['row'].astype(np.uint32);h^=h>>16;h*=np.uint32(0x7feb352d);h^=h>>15;h*=np.uint32(0x846ca68b);h^=h>>16
    for scenario in protocol['scenarios']:
        engine=BehavioralObservation();values=np.zeros((len(d['y']),26),np.float32);seen=np.zeros(len(d['y']),bool);probe_order=[];wall=perf_counter()
        if scenario=='half_visibility':
            order=np.flatnonzero((h&1)==0);arrival=None;guard=None
        else:
            arrival=m['t']+(h.astype(np.int64)%1001);order=np.argsort(arrival,kind='stable');guard=int(m['t'][-1])-1000
        for a in range(0,len(order),100000):
            ix=order[a:a+100000];chunk=np.array(m[ix],copy=True)
            if arrival is not None:chunk['t']=arrival[ix]
            for i,event in enumerate(zip(*event_columns(chunk))):
                row=int(chunk['row'][i]);query=row in wanted and (guard is None or int(chunk['t'][i])<=guard)
                v=engine.observe(*event,query=query)
                if query:
                    j=wanted[row];values[j]=(*v,*engine.global_features());seen[j]=True;probe_order.append(j)
        np.save(O/f'{scenario}_unsw_behavior.npy',values)
        ids=np.array(probe_order,dtype=int);assert len(ids)==int(seen.sum())
        sub={k:v[ids] for k,v in d.items()};sub['behavior']=values[ids]
        for c in choices:
            chosen=c['selected']['shared_pooled'];model=joblib.load(R/chosen['artifact']);x=representation(sub,c['method'],features);s=scores(model,x)
            record=dict(scenario=scenario,method=c['method'],seed=42,observed_probe_rows=len(ids),all_probe_rows=len(d['y']),
                observed_attack_coverage=float(sub['y'].sum()/d['y'].sum()),
                **metrics(sub['y'],s,chosen['thresholds']['shared'],sub['family'],windows=20))
            record['unconditional_recall_including_unobserved']=record['tp']/int(d['y'].sum());out.append(record)
        print('COMPLETE sensor scenario',scenario,'observed probes',len(ids),'wall',round(perf_counter()-wall,2),flush=True)
    (O/'sensor_sensitivity_results.json').write_text(json.dumps(dict(protocol=protocol,results=out,
        caveat='Cross-visibility/jitter shift of models calibrated under the complete ordered-record proxy. Not a matched retraining experiment, live network test or independent validation.'),indent=2))
if __name__=='__main__':main()
