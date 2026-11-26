"""Descriptive conditional-label changes on repeated development vectors.

Uses naturally sampled tune/stress periods, not case-control training samples.
No p-values, causal attribution, threshold tuning, or future-risk guarantee.
"""
import json
import numpy as np
from run_network_context_study import D,O,part,slice_data,holdouts
from run_multienvironment_training import ton_temporal_boundaries
from network_context_features import transform

def main():
    a,_=ton_temporal_boundaries();tune=part(D,'tune');h=holdouts();f=json.loads((D/'manifest.json').read_text())['features']
    data={'earlier_calibration_period':slice_data(tune,slice(0,a)),'nearby_future':h['ton_future'],'later_stress':h['ton_stress']}
    values=[];labels=[];period=[]
    for e,(name,d) in enumerate(data.items()):
        c=transform(d['x'],d['ports'],f,'context29')
        x=np.column_stack([d['x'],c[:,14:24]]).astype(np.float32)
        values.append(x);labels.append(d['y']);period.append(np.full(len(x),e))
    x=np.ascontiguousarray(np.nan_to_num(np.concatenate(values),nan=-np.inf));y=np.concatenate(labels);e=np.concatenate(period)
    key=x.view(np.dtype((np.void,x.dtype.itemsize*x.shape[1]))).ravel();unique,g=np.unique(key,return_inverse=True)
    n=np.zeros((len(unique),3),int);att=np.zeros_like(n);np.add.at(n,(g,e),1);np.add.at(att,(g,e),y)
    comparisons=[];names=list(data)
    for earlier,later in [(0,1),(0,2),(1,2)]:
        supported=(n[:,earlier]>=100)&(n[:,later]>=100)
        rate0=att[supported,earlier]/n[supported,earlier];rate1=att[supported,later]/n[supported,later]
        support=np.minimum(n[supported,earlier],n[supported,later]);change=rate1-rate0
        details=[]
        for idx in np.flatnonzero(supported):
            details.append(dict(vector_group=int(idx),earlier_rows=int(n[idx,earlier]),later_rows=int(n[idx,later]),
                earlier_attack_rate=float(att[idx,earlier]/n[idx,earlier]),later_attack_rate=float(att[idx,later]/n[idx,later])))
        errors={}
        if later==2:
            with np.load(O/'physical14_seed42_shared_window_scores.npz') as z:score=z['ton_stress']
            threshold=json.loads((O/'physical14_seed42_selection.json').read_text())['selected']['shared_window']['thresholds']['shared']
            yy=data['later_stress']['y'];pred=score>threshold;covered=supported[g[e==later]]
            for name,mask in [('repeated_high_support_vectors',covered),('other_vectors',~covered)]:
                benign=(yy==0)&mask;attacks=(yy==1)&mask
                errors[name]=dict(rows=int(mask.sum()),benign=int(benign.sum()),attacks=int(attacks.sum()),
                    fp=int(pred[benign].sum()),tp=int(pred[attacks].sum()))
        comparisons.append(dict(earlier=names[earlier],later=names[later],minimum_rows_per_period=100,
            shared_supported_vectors=int(supported.sum()),earlier_rows_in_support=int(n[supported,earlier].sum()),
            later_rows_in_support=int(n[supported,later].sum()),
            overlap_weighted_absolute_attack_rate_change=float(np.average(np.abs(change),weights=support)) if len(change) else None,
            vectors_with_absolute_change_at_least_point2=int((np.abs(change)>=.2).sum()),
            baseline_error_partition=errors,
            groups=sorted(details,key=lambda z:abs(z['later_attack_rate']-z['earlier_attack_rate']),reverse=True)))
    (O/'conditional_shift_diagnostic.json').write_text(json.dumps(dict(scope=__doc__,
        representation='All46 cached aggregates plus the already-frozen9 service indicators and known-role flag.',
        comparisons=comparisons,
        caveats=['Natural period samples avoid the deliberately class-balanced training-prior confound.',
                 'Differences are empirical conditional label rates for recorded vectors; finite sampling, temporal dependence, projected-out context and label semantics remain relevant.',
                 'Does not prove a population concept-drift mechanism or annotation error.',
                 'Period identity and these labels are not used as model inputs or update decisions.']),indent=2))
    print('COMPLETE conditional shift diagnostic',[(z['earlier'],z['later'],z['shared_supported_vectors'],z['overlap_weighted_absolute_attack_rate_change']) for z in comparisons])
if __name__=='__main__':main()
